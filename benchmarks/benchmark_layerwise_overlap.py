import argparse
import ctypes
import os
import socket
import struct
import threading
import time
from dataclasses import dataclass
from multiprocessing import Process
from typing import Dict, List

import torch

from flexkv.common.config import CacheConfig, ModelConfig
from flexkv.common.debug import flexkv_logger
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.kvmanager import KVManager
from flexkv.kvtask import KVResponseStatus
from flexkv.server.client import KVTPClient
from utils import load_config


flexkv_logger.set_level("INFO")

_LIBC = ctypes.CDLL("libc.so.6", use_errno=True)
_EFD_SEMAPHORE = 0x1


def _eventfd(initval: int = 0, flags: int = 0) -> int:
    fd = _LIBC.eventfd(ctypes.c_uint(initval), ctypes.c_int(flags))
    if fd == -1:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return fd


def _send_fds(sock: socket.socket, fds: List[int], extra_data: bytes) -> None:
    packed = struct.pack(f"{len(fds)}i", *fds)
    sock.sendmsg([extra_data], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, packed)])


class MockEventfdClient:
    def __init__(self, socket_path: str, num_layers: int, num_counters: int = 3):
        self.socket_path = socket_path
        self.num_layers = num_layers
        self.num_counters = num_counters
        self.fds = [_eventfd(0, _EFD_SEMAPHORE) for _ in range(num_layers * num_counters)]
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        sock = None
        for _ in range(180):
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                sock.connect(self.socket_path)
                break
            except (FileNotFoundError, ConnectionRefusedError):
                sock.close()
                sock = None
                time.sleep(0.5)
        if sock is None:
            print(f"[MockEventfdClient] failed to connect to {self.socket_path}")
            return

        metadata = struct.pack("iiii", 0, 1, self.num_layers, self.num_counters)
        sock.sendall(metadata)
        fd_idx = 0
        for counter_id in range(self.num_counters):
            fds = self.fds[fd_idx:fd_idx + self.num_layers]
            fd_idx += self.num_layers
            _send_fds(sock, fds, struct.pack("i", counter_id))
        sock.settimeout(30.0)
        ack = sock.recv(1)
        if ack == b"\x01":
            print(f"[MockEventfdClient] eventfd handshake OK, layers={self.num_layers}")
            self.ready.set()
        else:
            print(f"[MockEventfdClient] unexpected ack={ack!r}")
        # Keep the socket open so fds remain owned by this process.
        while True:
            time.sleep(60)

    def wait_layer(self, counter_id: int, layer_id: int) -> None:
        fd = self.fds[counter_id * self.num_layers + layer_id]
        os.read(fd, 8)


@dataclass
class BenchmarkConfig:
    mode: str
    batch_size: int
    sequence_length: int
    cache_ratio: float
    fake_compute_ms: float
    counter_id: int


def run_tp_client(dp_client_id, tp_rank, gpu_register_port, model_config, cache_config):
    device_id = tp_rank + dp_client_id * model_config.tp_size
    tp_client = KVTPClient(gpu_register_port, dp_client_id, device_id)
    num_gpu_blocks = cache_config.num_gpu_blocks
    gpu_kv_layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST,
        num_layer=model_config.num_layers,
        num_block=num_gpu_blocks,
        tokens_per_block=cache_config.tokens_per_block,
        num_head=model_config.num_kv_heads,
        head_size=model_config.head_size,
        is_mla=model_config.use_mla,
    )
    gpu_blocks_for_tp = [
        torch.empty(size=tuple(gpu_kv_layout.kv_shape[1:]), dtype=model_config.dtype).cuda(device_id)
        for _ in range(model_config.num_layers)
    ]
    tp_client.register_to_server(gpu_blocks_for_tp, gpu_kv_layout)
    while True:
        time.sleep(1)


def shutdown_tp_clients(processes):
    for proc in processes:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=2)


def fake_attention_compute(fake_compute_ms: float) -> None:
    if fake_compute_ms > 0:
        time.sleep(fake_compute_ms / 1000.0)


def benchmark(model_config: ModelConfig, cache_config: CacheConfig, bench_config: BenchmarkConfig):
    if model_config.tp_size * model_config.dp_size > torch.cuda.device_count():
        raise ValueError("not enough GPUs")

    event_client = None
    socket_path = os.environ.get("FLEXKV_LAYERWISE_EVENTFD_SOCKET", "/tmp/flexkv_layerwise_eventfd.sock")
    if bench_config.mode == "layerwise":
        try:
            os.unlink(socket_path)
        except FileNotFoundError:
            pass
        event_client = MockEventfdClient(socket_path, model_config.num_layers)
        event_client.start()

    kvmanager = KVManager(model_config, cache_config)
    kvmanager.start()

    cache_config.num_gpu_blocks = bench_config.sequence_length * bench_config.batch_size // cache_config.tokens_per_block
    print(f"allocate {cache_config.num_gpu_blocks} gpu blocks for benchmark")

    tp_processes = []
    for tp_rank in range(model_config.tp_size):
        proc = Process(
            target=run_tp_client,
            args=(0, tp_rank, kvmanager.gpu_register_port, model_config, cache_config),
            daemon=True,
        )
        proc.start()
        tp_processes.append(proc)

    while not kvmanager.is_ready():
        time.sleep(1)
        flexkv_logger.info("waiting for flexkv to be ready")
    flexkv_logger.info("flexkv is ready")

    batch_sequence_tensor = []
    batch_slot_mapping = []
    cache_length = int(bench_config.sequence_length * bench_config.cache_ratio)
    for i in range(bench_config.batch_size):
        batch_sequence_tensor.append(torch.randint(0, 100000, (bench_config.sequence_length,), dtype=torch.int64))
        batch_slot_mapping.append(
            torch.arange(
                i * bench_config.sequence_length,
                (i + 1) * bench_config.sequence_length,
                dtype=torch.int64,
            )
        )

    put_ids = [
        kvmanager.put_async(seq[:cache_length], slots[:cache_length], token_mask=None)
        for seq, slots in zip(batch_sequence_tensor, batch_slot_mapping)
    ]
    put_start = time.time()
    put_result = kvmanager.wait(put_ids, completely=True)
    put_ms = (time.time() - put_start) * 1000
    put_tokens = sum(r.return_mask.sum().item() for r in put_result.values() if r.status == KVResponseStatus.SUCCESS)
    print(f"put {put_tokens} tokens, time: {put_ms:.2f}ms")

    get_ids = []
    get_start = time.time()
    for seq in batch_sequence_tensor:
        task_id, _ = kvmanager.get_match(seq, token_mask=None)
        get_ids.append(task_id)
    match_ms = (time.time() - get_start) * 1000

    if bench_config.mode == "baseline":
        kvmanager.launch(get_ids, batch_slot_mapping, as_batch=True, layerwise_transfer=False)
        kvmanager.wait(get_ids)
        transfer_done_ms = (time.time() - get_start) * 1000
        compute_start = time.time()
        for _ in range(model_config.num_layers):
            fake_attention_compute(bench_config.fake_compute_ms)
        total_ms = (time.time() - get_start) * 1000
        compute_ms = (time.time() - compute_start) * 1000
    else:
        if event_client is None:
            raise RuntimeError("missing event client")
        kvmanager.launch(
            get_ids,
            batch_slot_mapping,
            as_batch=True,
            layerwise_transfer=True,
            counter_id=bench_config.counter_id,
        )
        transfer_done_ms = -1.0
        compute_start = time.time()
        for layer_id in range(model_config.num_layers):
            event_client.wait_layer(bench_config.counter_id, layer_id)
            fake_attention_compute(bench_config.fake_compute_ms)
        kvmanager.wait(get_ids)
        total_ms = (time.time() - get_start) * 1000
        compute_ms = (time.time() - compute_start) * 1000

    print(
        f"synthetic_inference mode={bench_config.mode}, bs={bench_config.batch_size}, "
        f"seq={bench_config.sequence_length}, fake_compute_ms={bench_config.fake_compute_ms}, "
        f"match_ms={match_ms:.2f}, transfer_done_ms={transfer_done_ms:.2f}, "
        f"compute_loop_ms={compute_ms:.2f}, total_ms={total_ms:.2f}"
    )

    shutdown_tp_clients(tp_processes)
    kvmanager.shutdown()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="benchmarks/local_8l.yml")
    parser.add_argument("--mode", choices=["baseline", "layerwise"], required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--cache-ratio", type=float, default=1.0)
    parser.add_argument("--fake-compute-ms", type=float, default=3.0)
    parser.add_argument("--counter-id", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_config, cache_config = load_config(args.config)
    sequence_length = ((args.sequence_length - 1) // cache_config.tokens_per_block + 1) * cache_config.tokens_per_block
    benchmark(
        model_config,
        cache_config,
        BenchmarkConfig(
            mode=args.mode,
            batch_size=args.batch_size,
            sequence_length=sequence_length,
            cache_ratio=args.cache_ratio,
            fake_compute_ms=args.fake_compute_ms,
            counter_id=args.counter_id,
        ),
    )
