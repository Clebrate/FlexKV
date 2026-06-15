import argparse
import ctypes
import math
import os
import socket
import struct
import threading
import time
from dataclasses import dataclass
from multiprocessing import Process
from typing import List

import torch

from flexkv.common.config import CacheConfig, GLOBAL_CONFIG_FROM_ENV, ModelConfig
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


class EventfdClient:
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
            print(f"[EventfdClient] failed to connect to {self.socket_path}")
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
            print(f"[EventfdClient] eventfd handshake OK, layers={self.num_layers}")
            self.ready.set()
        else:
            print(f"[EventfdClient] unexpected ack={ack!r}")
        while True:
            time.sleep(60)

    def wait_ready(self, timeout: float = 120.0) -> None:
        if not self.ready.wait(timeout):
            raise TimeoutError("timed out waiting for eventfd handshake")

    def wait_layer(self, counter_id: int, layer_id: int) -> None:
        fd = self.fds[counter_id * self.num_layers + layer_id]
        os.read(fd, 8)


class GpuAttentionCompute:
    def __init__(
        self,
        batch_size: int,
        sequence_length: int,
        num_layers: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
        device: int,
        repeats: int,
    ):
        self.num_layers = num_layers
        self.scale = 1.0 / math.sqrt(head_size)
        self.repeats = repeats
        self.device = torch.device(f"cuda:{device}")
        self.dtype = dtype

        self.q = torch.randn(
            (num_layers, batch_size, num_heads, head_size),
            device=self.device,
            dtype=dtype,
        )
        self.k = torch.randn(
            (batch_size, sequence_length, num_heads, head_size),
            device=self.device,
            dtype=dtype,
        )
        self.v = torch.randn(
            (batch_size, sequence_length, num_heads, head_size),
            device=self.device,
            dtype=dtype,
        )
        self.out = None

        self.run_layer(0)
        torch.cuda.synchronize(self.device)

    def run_layer(self, layer_id: int) -> None:
        q = self.q[layer_id]
        out = None
        for _ in range(self.repeats):
            scores = torch.einsum("bhd,bshd->bhs", q, self.k) * self.scale
            probs = torch.softmax(scores.float(), dim=-1).to(self.dtype)
            out = torch.einsum("bhs,bshd->bhd", probs, self.v)
            q = out
        self.out = out


@dataclass
class BenchmarkConfig:
    mode: str
    batch_size: int
    sequence_length: int
    cache_ratio: float
    counter_id: int
    compute_kind: str
    compute_sequence_length: int
    attention_repeats: int


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


def build_compute(model_config: ModelConfig, bench_config: BenchmarkConfig):
    if bench_config.compute_kind == "none":
        return None
    compute_seq = bench_config.compute_sequence_length or bench_config.sequence_length
    compute_seq = min(compute_seq, bench_config.sequence_length)
    return GpuAttentionCompute(
        batch_size=bench_config.batch_size,
        sequence_length=compute_seq,
        num_layers=model_config.num_layers,
        num_heads=model_config.num_kv_heads,
        head_size=model_config.head_size,
        dtype=model_config.dtype,
        device=0,
        repeats=bench_config.attention_repeats,
    )


def run_compute_layer(compute, layer_id: int) -> None:
    if compute is not None:
        compute.run_layer(layer_id)


def benchmark(model_config: ModelConfig, cache_config: CacheConfig, bench_config: BenchmarkConfig):
    if model_config.tp_size * model_config.dp_size > torch.cuda.device_count():
        raise ValueError("not enough GPUs")

    layerwise_enabled = bench_config.mode == "layerwise"
    os.environ["FLEXKV_ENABLE_LAYERWISE_TRANSFER"] = "1" if layerwise_enabled else "0"
    GLOBAL_CONFIG_FROM_ENV.enable_layerwise_transfer = layerwise_enabled

    event_client = None
    socket_path = os.environ.get("FLEXKV_LAYERWISE_EVENTFD_SOCKET", "/tmp/flexkv_layerwise_eventfd.sock")
    if layerwise_enabled:
        try:
            os.unlink(socket_path)
        except FileNotFoundError:
            pass
        event_client = EventfdClient(socket_path, model_config.num_layers)
        event_client.start()

    kvmanager = KVManager(model_config, cache_config)
    kvmanager.start()

    cache_config.num_gpu_blocks = bench_config.sequence_length * bench_config.batch_size // cache_config.tokens_per_block
    print(f"allocate {cache_config.num_gpu_blocks} gpu blocks for benchmark")

    tp_processes = []
    try:
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

        if event_client is not None:
            event_client.wait_ready()

        compute = build_compute(model_config, bench_config)
        torch.cuda.synchronize()

        batch_sequence_tensor = []
        batch_slot_mapping = []
        cache_length = int(bench_config.sequence_length * bench_config.cache_ratio)
        for i in range(bench_config.batch_size):
            batch_sequence_tensor.append(
                torch.randint(0, 100000, (bench_config.sequence_length,), dtype=torch.int64)
            )
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
        put_tokens = sum(
            r.return_mask.sum().item()
            for r in put_result.values()
            if r.status == KVResponseStatus.SUCCESS
        )
        print(f"put {put_tokens} tokens, time: {put_ms:.2f}ms")

        get_ids = []
        get_start = time.time()
        torch.cuda.nvtx.range_push(f"flexkv.onboard.{bench_config.mode}.e2e")
        try:
            for seq in batch_sequence_tensor:
                task_id, _ = kvmanager.get_match(seq, token_mask=None)
                get_ids.append(task_id)
            match_ms = (time.time() - get_start) * 1000

            if bench_config.mode == "baseline":
                kvmanager.launch(get_ids, batch_slot_mapping, as_batch=True, layerwise_transfer=False)
                kvmanager.wait(get_ids)
                transfer_done_ms = (time.time() - get_start) * 1000
                compute_start = time.time()
                for layer_id in range(model_config.num_layers):
                    run_compute_layer(compute, layer_id)
                torch.cuda.synchronize()
                compute_ms = (time.time() - compute_start) * 1000
                total_ms = (time.time() - get_start) * 1000
            else:
                assert event_client is not None
                kvmanager.launch(
                    get_ids,
                    batch_slot_mapping,
                    as_batch=True,
                    layerwise_transfer=True,
                    counter_id=bench_config.counter_id,
                )
                transfer_done_ms = -1.0
                compute_start = time.time()
                torch.cuda.nvtx.range_push("flexkv.onboard.layerwise.eventfd_wait.total")
                try:
                    for layer_id in range(model_config.num_layers):
                        torch.cuda.nvtx.range_push(
                            f"flexkv.onboard.layerwise.eventfd_wait.layer{layer_id}"
                        )
                        try:
                            event_client.wait_layer(bench_config.counter_id, layer_id)
                        finally:
                            torch.cuda.nvtx.range_pop()
                        run_compute_layer(compute, layer_id)
                finally:
                    torch.cuda.nvtx.range_pop()
                kvmanager.wait(get_ids)
                torch.cuda.synchronize()
                compute_ms = (time.time() - compute_start) * 1000
                total_ms = (time.time() - get_start) * 1000
        finally:
            torch.cuda.nvtx.range_pop()

        print(
            f"mini_inference mode={bench_config.mode}, bs={bench_config.batch_size}, "
            f"seq={bench_config.sequence_length}, compute_kind={bench_config.compute_kind}, "
            f"compute_seq={bench_config.compute_sequence_length or bench_config.sequence_length}, "
            f"attention_repeats={bench_config.attention_repeats}, match_ms={match_ms:.2f}, "
            f"transfer_done_ms={transfer_done_ms:.2f}, compute_loop_ms={compute_ms:.2f}, "
            f"total_ms={total_ms:.2f}"
        )
    finally:
        shutdown_tp_clients(tp_processes)
        kvmanager.shutdown()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="benchmarks/local_8l.yml")
    parser.add_argument("--mode", choices=["baseline", "layerwise"], required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--cache-ratio", type=float, default=1.0)
    parser.add_argument("--counter-id", type=int, default=0)
    parser.add_argument("--compute-kind", choices=["none", "gpu-attention"], default="gpu-attention")
    parser.add_argument("--compute-sequence-length", type=int, default=0)
    parser.add_argument("--attention-repeats", type=int, default=1)
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
            counter_id=args.counter_id,
            compute_kind=args.compute_kind,
            compute_sequence_length=args.compute_sequence_length,
            attention_repeats=args.attention_repeats,
        ),
    )
