#!/usr/bin/env python3
"""
单次运行：在**已设置好** FLEXKV_SERVER_CLIENT_MODE / FLEXKV_SERVER_RECV_PORT 的独立进程中执行。
由 run_compare.py 启动；勿在同一进程内先 import flexkv 再改环境变量。

在 FlexKV 仓库根目录下：
  FLEXKV_SERVER_CLIENT_MODE=0 FLEXKV_SERVER_RECV_PORT=ipc:///tmp/foo \\
    python3 benchmarks/kvmanager_mode_compare/bench_one_run.py
"""
from __future__ import annotations

import argparse
import inspect
import json
import multiprocessing as mp
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

try:
    import nvtx as _nvtx_mod
except ImportError:
    _nvtx_mod = None

# TP 子进程必须用 spawn：父进程已 import torch 后若用默认 fork，易 CUDA 死锁（GPU 一直 0%、进程挂住）
_SPAWN = mp.get_context("spawn")

_FLEXKV_ROOT = Path(__file__).resolve().parents[2]
_BENCHMARKS_ROOT = _FLEXKV_ROOT / "benchmarks"
if str(_BENCHMARKS_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS_ROOT))

import torch

from common.utils import load_config
from flexkv.common.config import GLOBAL_CONFIG_FROM_ENV
from flexkv.common.debug import flexkv_logger
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.kvmanager import KVManager
from flexkv.kvtask import KVResponseStatus
from flexkv.server.client import KVTPClient

flexkv_logger.set_level(os.getenv("FLEXKV_LOG_LEVEL", "OFF"))

_JSON_PREFIX = "FLEXKV_BENCH_JSON "


@contextmanager
def _bench_nvtx(message: str):
    """Nsight Systems 中显示为 NVTX 区间；未安装 nvtx 时为空操作。"""
    if _nvtx_mod is None:
        yield
        return
    _nvtx_mod.push_range(message)
    try:
        yield
    finally:
        _nvtx_mod.pop_range()


def _log(msg: str) -> None:
    print(f"[bench_one_run] {msg}", file=sys.stderr, flush=True)


def _apply_transfer_manager_mode_override(mode: str) -> None:
    mode = mode.strip().lower()
    if mode == "process":
        return
    if mode not in ("thread",):
        raise ValueError(
            f"不支持的 transfer manager mode: {mode}，仅支持 process/thread"
        )

    import flexkv.transfer_manager as transfer_manager_mod

    if getattr(transfer_manager_mod.TransferManagerHandle, "_bench_mode_patched", False):
        return

    original_init = transfer_manager_mod.TransferManagerHandle.__init__

    def patched_init(self, model_config, cache_config, gpu_register_port=None, mode="process", **kwargs):
        selected_mode = mode
        if mode == "process":
            selected_mode = os.getenv("FLEXKV_TRANSFER_MANAGER_MODE", "process").strip().lower()
        return original_init(
            self,
            model_config,
            cache_config,
            gpu_register_port=gpu_register_port,
            mode=selected_mode,
            **kwargs,
        )

    transfer_manager_mod.TransferManagerHandle.__init__ = patched_init
    transfer_manager_mod.TransferManagerHandle._bench_mode_patched = True
    _log(f"patched TransferManagerHandle mode override: process -> {mode}")


def _ensure_global_config_compat() -> None:
    """
    Patch GLOBAL_CONFIG_FROM_ENV for schema differences across FlexKV builds.

    Some builds expose transfer_num_cta_* instead of transfer_sms_*.
    """
    cfg = GLOBAL_CONFIG_FROM_ENV

    if not hasattr(cfg, "transfer_sms_h2d"):
        fallback = getattr(cfg, "transfer_num_cta_h2d", 8)
        setattr(cfg, "transfer_sms_h2d", int(fallback))

    if not hasattr(cfg, "transfer_sms_d2h"):
        fallback = getattr(cfg, "transfer_num_cta_d2h", 8)
        setattr(cfg, "transfer_sms_d2h", int(fallback))

    if not hasattr(cfg, "use_ce_transfer_h2d"):
        setattr(cfg, "use_ce_transfer_h2d", False)

    if not hasattr(cfg, "use_ce_transfer_d2h"):
        setattr(cfg, "use_ce_transfer_d2h", False)

    _log(
        "global config compat: "
        f"use_ce_h2d={getattr(cfg, 'use_ce_transfer_h2d', None)}, "
        f"use_ce_d2h={getattr(cfg, 'use_ce_transfer_d2h', None)}, "
        f"sms_h2d={getattr(cfg, 'transfer_sms_h2d', None)}, "
        f"sms_d2h={getattr(cfg, 'transfer_sms_d2h', None)}"
    )


def _patch_worker_kwargs_compat() -> None:
    """
    Drop optional kwargs that may be unsupported by mixed worker versions.
    """
    import flexkv.transfer.worker as worker_mod

    if getattr(worker_mod.TransferWorkerBase, "_bench_create_worker_patched", False):
        return

    original_create_worker = worker_mod.TransferWorkerBase.create_worker.__func__

    def patched_create_worker(
        cls,
        mp_ctx,
        finished_ops_queue,
        op_buffer_tensor,
        *args,
        **kwargs,
    ):
        # Some runtime worker builds don't accept transfer_sms_* kwargs.
        kwargs.pop("transfer_sms_h2d", None)
        kwargs.pop("transfer_sms_d2h", None)

        # Best-effort trim unknown kwargs by constructor signature.
        try:
            sig = inspect.signature(cls.__init__)
            params = sig.parameters
            accepts_varkw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
            if not accepts_varkw:
                allowed = {k for k in params.keys() if k != "self"}
                kwargs = {k: v for k, v in kwargs.items() if k in allowed}
        except Exception:
            pass

        return original_create_worker(
            cls, mp_ctx, finished_ops_queue, op_buffer_tensor, *args, **kwargs
        )

    worker_mod.TransferWorkerBase.create_worker = classmethod(patched_create_worker)
    worker_mod.TransferWorkerBase._bench_create_worker_patched = True
    _log("patched TransferWorkerBase.create_worker kwargs compat")


def _patch_worker_handle_del_compat() -> None:
    """
    Suppress noisy ValueError in WorkerHandle.__del__ when process already closed.
    """
    import flexkv.transfer.worker as worker_mod

    if getattr(worker_mod.WorkerHandle, "_bench_del_patched", False):
        return

    def patched_del(self) -> None:
        process = getattr(self, "process", None)
        if process is None:
            return
        try:
            alive = process.is_alive()
        except ValueError:
            # multiprocessing process object is already closed
            return
        except Exception:
            return
        if alive:
            try:
                self.shutdown()
            except Exception:
                pass

    try:
        worker_mod.WorkerHandle.__del__ = patched_del
        worker_mod.WorkerHandle._bench_del_patched = True
        _log("patched WorkerHandle.__del__ compat")
    except Exception as e:
        _log(f"patch WorkerHandle.__del__ skipped: {e}")


def _patch_server_subprocess_compat() -> None:
    """
    Ensure server_client subprocess applies the same compat patches.
    """
    import pickle
    import subprocess
    import textwrap
    import tempfile
    import flexkv.server.server as server_mod

    if getattr(server_mod.KVServer, "_bench_create_server_patched", False):
        return

    original_create_server = server_mod.KVServer.create_server.__func__

    def patched_create_server(
        cls,
        model_config,
        cache_config,
        gpu_register_port,
        server_recv_port=None,
        total_clients: int = 0,
        child_env: dict | None = None,
        inherit_env: bool = True,
    ):
        # Keep original behavior unless caller explicitly asks subprocess path.
        if child_env is None and inherit_env:
            return original_create_server(
                cls,
                model_config,
                cache_config,
                gpu_register_port,
                server_recv_port=server_recv_port,
                total_clients=total_clients,
                child_env=child_env,
                inherit_env=inherit_env,
            )

        if server_recv_port is None:
            server_recv_port = f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"

        env = os.environ.copy()
        if child_env:
            env.update(child_env)
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env.update({"FLEXKV_INSTANCE_NUM": str(total_clients // model_config.dp_size)})

        args_data = pickle.dumps(
            (model_config, cache_config, gpu_register_port, server_recv_port, total_clients)
        )
        flexkv_root = str(_FLEXKV_ROOT)
        server_script = textwrap.dedent(
            f"""
            import inspect
            import os
            import pickle
            import sys
            sys.path.insert(0, {flexkv_root!r})

            from flexkv.common.config import GLOBAL_CONFIG_FROM_ENV
            if not hasattr(GLOBAL_CONFIG_FROM_ENV, "transfer_sms_h2d"):
                setattr(
                    GLOBAL_CONFIG_FROM_ENV,
                    "transfer_sms_h2d",
                    int(getattr(GLOBAL_CONFIG_FROM_ENV, "transfer_num_cta_h2d", 8)),
                )
            if not hasattr(GLOBAL_CONFIG_FROM_ENV, "transfer_sms_d2h"):
                setattr(
                    GLOBAL_CONFIG_FROM_ENV,
                    "transfer_sms_d2h",
                    int(getattr(GLOBAL_CONFIG_FROM_ENV, "transfer_num_cta_d2h", 8)),
                )
            if not hasattr(GLOBAL_CONFIG_FROM_ENV, "use_ce_transfer_h2d"):
                setattr(GLOBAL_CONFIG_FROM_ENV, "use_ce_transfer_h2d", False)
            if not hasattr(GLOBAL_CONFIG_FROM_ENV, "use_ce_transfer_d2h"):
                setattr(GLOBAL_CONFIG_FROM_ENV, "use_ce_transfer_d2h", False)

            import flexkv.transfer.worker as worker_mod
            import flexkv.transfer_manager as transfer_manager_mod
            original_create_worker = worker_mod.TransferWorkerBase.create_worker.__func__

            def patched_create_worker(cls, mp_ctx, finished_ops_queue, op_buffer_tensor, *args, **kwargs):
                kwargs.pop("transfer_sms_h2d", None)
                kwargs.pop("transfer_sms_d2h", None)
                try:
                    sig = inspect.signature(cls.__init__)
                    params = sig.parameters
                    accepts_varkw = any(
                        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
                    )
                    if not accepts_varkw:
                        allowed = {{k for k in params.keys() if k != "self"}}
                        kwargs = {{k: v for k, v in kwargs.items() if k in allowed}}
                except Exception:
                    pass
                return original_create_worker(
                    cls, mp_ctx, finished_ops_queue, op_buffer_tensor, *args, **kwargs
                )

            worker_mod.TransferWorkerBase.create_worker = classmethod(patched_create_worker)

            # Make server subprocess honor FLEXKV_TRANSFER_MANAGER_MODE (thread/process).
            original_tm_init = transfer_manager_mod.TransferManagerHandle.__init__

            def patched_tm_init(self, model_config, cache_config, gpu_register_port=None, mode="process", **kwargs):
                selected_mode = mode
                if mode == "process":
                    selected_mode = os.getenv("FLEXKV_TRANSFER_MANAGER_MODE", "process").strip().lower()
                print(
                    f"[bench_one_run:server_subprocess] TransferManagerHandle mode override: {{mode}} -> {{selected_mode}}",
                    file=sys.stderr,
                    flush=True,
                )
                return original_tm_init(
                    self,
                    model_config,
                    cache_config,
                    gpu_register_port=gpu_register_port,
                    mode=selected_mode,
                    **kwargs,
                )

            transfer_manager_mod.TransferManagerHandle.__init__ = patched_tm_init

            from flexkv.server.server import KVServer
            args_data = {args_data!r}
            model_config, cache_config, gpu_register_port, server_recv_port, total_clients = pickle.loads(args_data)
            server = KVServer(model_config, cache_config, gpu_register_port, server_recv_port, total_clients)
            server.run()
            """
        ).strip()

        process = subprocess.Popen([sys.executable, "-c", server_script], env=env)
        return server_mod.KVServerHandle(process)

    server_mod.KVServer.create_server = classmethod(patched_create_server)
    server_mod.KVServer._bench_create_server_patched = True
    _log("patched KVServer.create_server compat for server subprocess")


def _collect_transfer_handle_status(kvmanager: KVManager) -> str:
    kv_task_engine = getattr(kvmanager, "kv_task_engine", None)
    if kv_task_engine is None:
        return "kv_task_engine=none"
    transfer_handles = getattr(kv_task_engine, "transfer_handles", None)
    if not transfer_handles:
        return "transfer_handles=none"

    parts = []
    for idx, handle in enumerate(transfer_handles):
        impl = getattr(handle, "_handle", None)
        if impl is None:
            parts.append(f"h{idx}:impl=none")
            continue
        proc = getattr(impl, "process", None)
        ready_event = getattr(impl, "ready_event", None)
        local_ready = getattr(impl, "_is_ready", None)
        part = [f"h{idx}:{impl.__class__.__name__}"]
        if proc is not None:
            try:
                part.append(f"pid={proc.pid}")
                part.append(f"alive={proc.is_alive()}")
                part.append(f"exitcode={proc.exitcode}")
            except Exception as e:
                part.append(f"proc_err={e}")
        if ready_event is not None:
            try:
                part.append(f"ready_event={ready_event.is_set()}")
            except Exception as e:
                part.append(f"ready_event_err={e}")
        if local_ready is not None:
            part.append(f"local_ready={local_ready}")
        parts.append(",".join(part))
    return "; ".join(parts)


def _has_dead_transfer_process(kvmanager: KVManager) -> bool:
    kv_task_engine = getattr(kvmanager, "kv_task_engine", None)
    if kv_task_engine is None:
        return False
    transfer_handles = getattr(kv_task_engine, "transfer_handles", None) or []
    for handle in transfer_handles:
        impl = getattr(handle, "_handle", None)
        proc = getattr(impl, "process", None) if impl is not None else None
        if proc is not None and (not proc.is_alive()) and proc.exitcode not in (None, 0):
            return True
    return False


def _collect_server_handle_status(kvmanager: KVManager) -> str:
    server_handle = getattr(kvmanager, "server_handle", None)
    if server_handle is None:
        return "server_handle=none"

    proc = getattr(server_handle, "process", None)
    if proc is None:
        return "server_process=none"

    part = [f"server_proc_type={proc.__class__.__name__}"]
    pid = getattr(proc, "pid", None)
    if pid is not None:
        part.append(f"pid={pid}")

    alive = None
    exitcode = None
    try:
        if hasattr(proc, "is_alive"):
            alive = proc.is_alive()
            exitcode = getattr(proc, "exitcode", None)
        elif hasattr(proc, "poll"):
            rc = proc.poll()
            alive = rc is None
            exitcode = rc
    except Exception as e:
        part.append(f"status_err={e}")
        return ",".join(part)

    part.append(f"alive={alive}")
    part.append(f"exitcode={exitcode}")
    return ",".join(part)


def _has_dead_server_process(kvmanager: KVManager) -> bool:
    if not getattr(kvmanager, "server_client_mode", False):
        return False
    server_handle = getattr(kvmanager, "server_handle", None)
    proc = getattr(server_handle, "process", None) if server_handle is not None else None
    if proc is None:
        return False
    try:
        if hasattr(proc, "is_alive"):
            alive = proc.is_alive()
            exitcode = getattr(proc, "exitcode", None)
        elif hasattr(proc, "poll"):
            rc = proc.poll()
            alive = rc is None
            exitcode = rc
        else:
            return False
    except Exception:
        return False
    return (not alive) and (exitcode not in (None, 0))


@dataclass
class BenchArgs:
    batch_size: int
    sequence_length: int
    cache_ratio: float
    config_path: str
    measure_op: str
    prime_cache_for_get: bool
    ready_timeout_s: float
    transfer_manager_mode: str
    # 1.0：GPU block 数 = 序列所需（原行为）;<1：缩小池以促使 spill，get 更可能含下层→GPU 传输
    gpu_block_capacity_ratio: float


def _mode_label() -> str:
    return "server_client" if GLOBAL_CONFIG_FROM_ENV.server_client_mode else "direct"


def run_tp_client(
    dp_client_id: int,
    tp_rank: int,
    gpu_register_port: str,
    model_config,
    cache_config,
) -> None:
    device_id = tp_rank + dp_client_id * model_config.tp_size
    _log(
        f"tp client bootstrap: tp_rank={tp_rank}, dp_client_id={dp_client_id}, "
        f"device_id={device_id}, port={gpu_register_port}"
    )
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
    gpu_blocks_for_tp = []
    for _ in range(model_config.num_layers):
        gpu_blocks_for_tp.append(
            torch.empty(
                size=tuple(gpu_kv_layout.kv_shape[1:]),
                dtype=model_config.dtype,
            ).cuda(device_id)
        )
    _log(f"tp client register_to_server begin: device_id={device_id}")
    tp_client.register_to_server(gpu_blocks_for_tp, gpu_kv_layout)
    _log(f"tp client register_to_server done: device_id={device_id}")
    while True:
        time.sleep(1)


def shutdown_tp_client(processes: list[mp.Process]) -> None:
    for p in processes:
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)
            if p.is_alive():
                p.kill()
                p.join(timeout=2)


def _spawn_tp_clients(
    *,
    tp_processes: list[mp.Process],
    model_config,
    cache_config,
    gpu_register_port: str,
) -> None:
    for tp_rank in range(model_config.tp_size):
        _log(f"spawn TP client tp_rank={tp_rank} ...")
        p = _SPAWN.Process(
            target=run_tp_client,
            args=(0, tp_rank, gpu_register_port, model_config, cache_config),
            daemon=True,
        )
        p.start()
        _log(f"tp client started: tp_rank={tp_rank}, pid={p.pid}")
        tp_processes.append(p)


def _measure_put(
    kvmanager: KVManager,
    batch_put_token_ids: list[torch.Tensor],
    batch_slot_mapping: list[torch.Tensor],
    batch_token_mask: list[torch.Tensor | None],
) -> tuple[dict, int]:
    with _bench_nvtx("flexkv_bench/put_total"):
        t_put0 = time.perf_counter()
        batch_put_ids = []
        control_path_only = all(
            token_mask is not None and int(token_mask.sum().item()) == 0
            for token_mask in batch_token_mask
        )
        with _bench_nvtx("flexkv_bench/put_submit"):
            for i in range(len(batch_put_token_ids)):
                tid = kvmanager.put_async(
                    batch_put_token_ids[i],
                    batch_slot_mapping[i],
                    token_mask=batch_token_mask[i],
                )
                batch_put_ids.append(tid)
        t_put_submit = time.perf_counter()
        with _bench_nvtx("flexkv_bench/put_wait"):
            put_result = kvmanager.wait(batch_put_ids, completely=True) if batch_put_ids else {}
        t_put1 = time.perf_counter()

        put_tokens = 0
        for _, response in put_result.items():
            if response.status == KVResponseStatus.SUCCESS:
                put_tokens += int(response.return_mask.sum().item())

        return (
            {
                "put_submit_ms": (t_put_submit - t_put0) * 1000.0,
                "put_wait_ms": (t_put1 - t_put_submit) * 1000.0,
                "put_total_ms": (t_put1 - t_put0) * 1000.0,
                "put_control_path_only": control_path_only,
                "put_task_count": len(batch_put_ids),
            },
            put_tokens,
        )


def _build_block_sampled_mask(
    *,
    sequence_length: int,
    tokens_per_block: int,
    cache_ratio: float,
) -> torch.Tensor:
    """Build block-level sampled token_mask for benchmark control."""
    if cache_ratio <= 0.0:
        return torch.zeros(sequence_length, dtype=torch.bool)
    if cache_ratio >= 1.0:
        return torch.ones(sequence_length, dtype=torch.bool)

    num_blocks = sequence_length // tokens_per_block
    sampled_blocks = int(round(num_blocks * cache_ratio))
    sampled_blocks = max(1, min(num_blocks, sampled_blocks))

    block_mask = torch.zeros(num_blocks, dtype=torch.bool)
    selected_blocks = torch.randperm(num_blocks)[:sampled_blocks]
    block_mask[selected_blocks] = True
    return block_mask.repeat_interleave(tokens_per_block)


def _prepare_ratio_controlled_inputs(
    *,
    batch_sequence_tensor: list[torch.Tensor],
    batch_slot_mapping: list[torch.Tensor],
    tokens_per_block: int,
    cache_ratio: float,
) -> tuple[
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor | None],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor | None],
    int,
    int,
]:
    """
    Prepare get/put inputs controlled by cache_ratio with block-level sampling.

    - cache_ratio=1.0: all True (real transfer path)
    - 0<cache_ratio<1: random block sampling
    - cache_ratio=0.0: all False (control-path-only, no data transfer)
    """
    put_token_ids: list[torch.Tensor] = []
    put_slot_mapping: list[torch.Tensor] = []
    put_token_mask: list[torch.Tensor | None] = []
    get_token_ids: list[torch.Tensor] = []
    get_slot_mapping: list[torch.Tensor] = []
    get_token_mask: list[torch.Tensor | None] = []
    selected_tokens_total = 0
    total_tokens = 0

    for seq, slot in zip(batch_sequence_tensor, batch_slot_mapping):
        token_mask = _build_block_sampled_mask(
            sequence_length=len(seq),
            tokens_per_block=tokens_per_block,
            cache_ratio=cache_ratio,
        )
        selected_tokens = int(token_mask.sum().item())
        total_tokens += len(seq)
        selected_tokens_total += selected_tokens

        if selected_tokens == 0:
            # PUT control-path: keep token_ids and pass all-false mask.
            # slot_mapping must be empty to satisfy slot_mapping.size == token_mask.sum() == 0.
            put_token_ids.append(seq)
            put_slot_mapping.append(slot[token_mask])
            put_token_mask.append(token_mask)

            # GET control-path: backend asserts on all-false mask with non-empty token_ids.
            # Use an empty request and still run match/launch/wait end-to-end.
            get_token_ids.append(seq[token_mask])
            get_slot_mapping.append(slot[token_mask])
            get_token_mask.append(None)
        elif selected_tokens == len(seq):
            put_token_ids.append(seq)
            put_slot_mapping.append(slot)
            put_token_mask.append(None)
            get_token_ids.append(seq)
            get_slot_mapping.append(slot)
            get_token_mask.append(None)
        else:
            # Pack sampled blocks so backend receives a compact request.
            packed_token_ids = seq[token_mask]
            packed_slot_mapping = slot[token_mask]
            put_token_ids.append(packed_token_ids)
            put_slot_mapping.append(packed_slot_mapping)
            put_token_mask.append(None)
            get_token_ids.append(packed_token_ids)
            get_slot_mapping.append(packed_slot_mapping)
            get_token_mask.append(None)

    return (
        put_token_ids,
        put_slot_mapping,
        put_token_mask,
        get_token_ids,
        get_slot_mapping,
        get_token_mask,
        selected_tokens_total,
        total_tokens,
    )


def _measure_get(
    kvmanager: KVManager,
    batch_get_token_ids: list[torch.Tensor],
    batch_slot_mapping: list[torch.Tensor],
    batch_token_mask: list[torch.Tensor | None],
) -> tuple[dict, int, int]:
    with _bench_nvtx("flexkv_bench/get_total"):
        t_get0 = time.perf_counter()
        batch_get_ids = []
        all_tokens = 0
        with _bench_nvtx("flexkv_bench/get_match"):
            for i in range(len(batch_get_token_ids)):
                mask = batch_token_mask[i]
                if mask is None:
                    all_tokens += len(batch_get_token_ids[i])
                else:
                    all_tokens += int(mask.sum().item())
                tid, _ = kvmanager.get_match(
                    batch_get_token_ids[i],
                    token_mask=mask,
                )
                batch_get_ids.append(tid)
        t_get_match = time.perf_counter()
        with _bench_nvtx("flexkv_bench/get_launch"):
            kvmanager.launch(batch_get_ids, batch_slot_mapping)
        t_launch = time.perf_counter()
        with _bench_nvtx("flexkv_bench/get_wait"):
            get_result = kvmanager.wait(batch_get_ids)
        t_get1 = time.perf_counter()

        cached_tokens = 0
        for _, response in get_result.items():
            if response.status == KVResponseStatus.SUCCESS:
                cached_tokens += int(response.return_mask.sum().item())

        return (
            {
                "get_match_ms": (t_get_match - t_get0) * 1000.0,
                "get_launch_ms": (t_launch - t_get_match) * 1000.0,
                "get_wait_ms": (t_get1 - t_launch) * 1000.0,
                "get_total_ms": (t_get1 - t_get0) * 1000.0,
            },
            cached_tokens,
            all_tokens,
        )


def run_benchmark(model_config, cache_config, ba: BenchArgs) -> dict:
    if model_config.tp_size * model_config.dp_size > torch.cuda.device_count():
        raise RuntimeError(
            f"需要 GPU 数 >= tp_size*dp_size = {model_config.tp_size * model_config.dp_size}，"
            f"当前 {torch.cuda.device_count()}"
        )

    record: dict = {
        "mode": _mode_label(),
        "flexkv_server_client_mode": os.environ.get("FLEXKV_SERVER_CLIENT_MODE", "0"),
        "flexkv_server_recv_port": os.environ.get("FLEXKV_SERVER_RECV_PORT", ""),
        "batch_size": ba.batch_size,
        "sequence_length": ba.sequence_length,
        "cache_ratio": ba.cache_ratio,
        "tp_size": model_config.tp_size,
        "dp_size": model_config.dp_size,
        "measure_op": ba.measure_op,
        "transfer_manager_mode": ba.transfer_manager_mode,
    }
    token_size_bytes = int(model_config.token_size_in_bytes)
    record["token_size_bytes"] = token_size_bytes
    record["token_size_bits"] = token_size_bytes * 8
    record["throughput_legacy_note"] = (
        "put/get_effective_gbps 为历史字段名，单位实际是 GiB/s；"
        "新增 *_effective_gib_per_s 与 *_effective_gbit_per_s 作为明确单位字段。"
    )

    num_required_gpu_blocks = ba.sequence_length * ba.batch_size // cache_config.tokens_per_block
    gbcr = float(ba.gpu_block_capacity_ratio)
    if gbcr >= 1.0:
        cache_config.num_gpu_blocks = num_required_gpu_blocks
    else:
        # 缩小 GPU 可注册 block 数，KV 更易落在 CPU/SSD；direct 与 server_client 使用同一公式（控制变量）
        cache_config.num_gpu_blocks = max(1, int(num_required_gpu_blocks * gbcr))

    record["gpu_block_capacity_ratio"] = gbcr
    record["num_gpu_blocks_required"] = num_required_gpu_blocks
    record["num_gpu_blocks"] = cache_config.num_gpu_blocks

    tp_processes: list[mp.Process] = []
    kvmanager: KVManager | None = None

    try:
        t_ready0 = time.perf_counter()
        _log("KVManager() ...")
        kvmanager = KVManager(model_config, cache_config)

        # Only direct+thread needs pre-spawn to avoid local startup deadlock.
        # In server_client mode, pre-spawn can race with server startup and cause
        # TP registration failure.
        pre_spawn_tp = (
            ba.transfer_manager_mode == "thread"
            and not GLOBAL_CONFIG_FROM_ENV.server_client_mode
        )

        # In thread mode, TransferManagerHandle.start() runs registration on the same process.
        # If we call kvmanager.start() before TP clients are spawned, startup can deadlock
        # waiting for GPU registration that is only sent by TP clients started later.
        if pre_spawn_tp:
            _log("direct+thread mode: spawn TP clients before kvmanager.start()")
            _spawn_tp_clients(
                tp_processes=tp_processes,
                model_config=model_config,
                cache_config=cache_config,
                gpu_register_port=kvmanager.gpu_register_port,
            )

        _log("kvmanager.start() ...")
        kvmanager.start()
        t_after_start = time.perf_counter()

        if not pre_spawn_tp:
            _spawn_tp_clients(
                tp_processes=tp_processes,
                model_config=model_config,
                cache_config=cache_config,
                gpu_register_port=kvmanager.gpu_register_port,
            )

        _log("wait is_ready() ... (若久无下文，检查 GPU 与 IPC)")
        t_ready_wait_start = time.perf_counter()
        last_hb = time.perf_counter()
        with _bench_nvtx("flexkv_bench/is_ready_wait"):
            while not kvmanager.is_ready():
                time.sleep(0.05)
                now = time.perf_counter()
                exited = [p for p in tp_processes if (not p.is_alive())]
                if exited:
                    details = ", ".join(f"pid={p.pid}, exitcode={p.exitcode}" for p in exited)
                    raise RuntimeError(
                        "等待 is_ready() 期间检测到 TP client 已退出，"
                        f"可能是子进程初始化 CUDA/IPC 失败: {details}"
                    )
                if _has_dead_server_process(kvmanager):
                    server_status = _collect_server_handle_status(kvmanager)
                    raise RuntimeError(
                        "等待 is_ready() 期间检测到 KVServer 子进程退出，"
                        f"{server_status}"
                    )
                if _has_dead_transfer_process(kvmanager):
                    status = _collect_transfer_handle_status(kvmanager)
                    raise RuntimeError(
                        "等待 is_ready() 期间检测到 TransferManager 子进程退出，"
                        f"transfer handles: {status}"
                    )
                if ba.ready_timeout_s > 0 and (now - t_ready_wait_start) >= ba.ready_timeout_s:
                    status = ", ".join(
                        f"pid={p.pid}, alive={p.is_alive()}, exitcode={p.exitcode}"
                        for p in tp_processes
                    )
                    transfer_status = _collect_transfer_handle_status(kvmanager)
                    server_status = _collect_server_handle_status(kvmanager)
                    raise TimeoutError(
                        f"is_ready() 超时（>{ba.ready_timeout_s:.1f}s），"
                        f"tp clients: {status}, transfer handles: {transfer_status}, "
                        f"server handle: {server_status}, gpu_register_port={kvmanager.gpu_register_port}"
                    )
                if now - last_hb >= 10.0:
                    _log(
                        "still waiting is_ready() ... "
                        + _collect_transfer_handle_status(kvmanager)
                        + ", "
                        + _collect_server_handle_status(kvmanager)
                    )
                    last_hb = now
        t_ready1 = time.perf_counter()
        _log("is_ready() OK")

        record["mgr_construct_and_start_ms"] = (t_after_start - t_ready0) * 1000.0
        record["tp_register_until_ready_ms"] = (t_ready1 - t_after_start) * 1000.0
        record["ready_total_ms"] = (t_ready1 - t_ready0) * 1000.0

        batch_sequence_tensor = []
        batch_slot_mapping = []

        for i in range(ba.batch_size):
            batch_sequence_tensor.append(
                torch.randint(0, 100000, (ba.sequence_length,), dtype=torch.int64)
            )
            batch_slot_mapping.append(
                torch.arange(i * ba.sequence_length, (i + 1) * ba.sequence_length, dtype=torch.int64)
            )

        (
            batch_put_token_ids,
            batch_put_slot_mapping,
            batch_put_token_mask,
            batch_get_token_ids,
            batch_get_slot_mapping,
            batch_get_token_mask,
            selected_tokens_total,
            total_tokens,
        ) = _prepare_ratio_controlled_inputs(
            batch_sequence_tensor=batch_sequence_tensor,
            batch_slot_mapping=batch_slot_mapping,
            tokens_per_block=cache_config.tokens_per_block,
            cache_ratio=ba.cache_ratio,
        )

        control_path_only = selected_tokens_total == 0
        record["control_path_only"] = control_path_only
        record["selected_tokens_total"] = selected_tokens_total
        record["selected_tokens_percent"] = (
            100.0 * selected_tokens_total / total_tokens if total_tokens else 0.0
        )
        if control_path_only:
            _log(
                "cache_ratio=0：提交全 False token_mask 的控制路径任务，"
                "测 match/launch/wait，不触发真实数据搬运"
            )

        if ba.measure_op in ("both", "put"):
            put_metrics, put_tokens = _measure_put(
                kvmanager=kvmanager,
                batch_put_token_ids=batch_put_token_ids,
                batch_slot_mapping=batch_put_slot_mapping,
                batch_token_mask=batch_put_token_mask,
            )
            record.update(put_metrics)
            record["put_tokens"] = put_tokens
            put_bytes = int(put_tokens * token_size_bytes)
            put_bits = int(put_bytes * 8)
            record["put_transferred_bytes"] = put_bytes
            record["put_transferred_bits"] = put_bits
            if record["put_total_ms"] > 0:
                secs = record["put_total_ms"] / 1000.0
                put_gib_per_s = (put_bytes / (1024**3)) / secs
                put_gbit_per_s = (put_bits / (1000**3)) / secs
                record["put_effective_gib_per_s"] = put_gib_per_s
                record["put_effective_gbit_per_s"] = put_gbit_per_s
                # Keep legacy field for compatibility with existing scripts.
                record["put_effective_gbps"] = put_gib_per_s
            if ba.cache_ratio > 0 and put_tokens <= 0:
                raise RuntimeError(
                    "cache_ratio>0 预期 put 有真实搬运，但 put_tokens=0。"
                    "请检查配置是否仅 CPU 路径、token_mask 或上游输入。"
                )

        if ba.measure_op == "get" and ba.prime_cache_for_get:
            with _bench_nvtx("flexkv_bench/get_prefill_put"):
                prefill_metrics, prefill_tokens = _measure_put(
                    kvmanager=kvmanager,
                    batch_put_token_ids=batch_put_token_ids,
                    batch_slot_mapping=batch_put_slot_mapping,
                    batch_token_mask=batch_put_token_mask,
                )
            record["get_prefill_put_total_ms"] = prefill_metrics["put_total_ms"]
            record["get_prefill_put_tokens"] = prefill_tokens
            record["get_prefill_put_control_path_only"] = prefill_metrics.get(
                "put_control_path_only", False
            )
            prefill_bytes = int(prefill_tokens * token_size_bytes)
            record["get_prefill_put_transferred_bytes"] = prefill_bytes
            record["get_prefill_put_transferred_bits"] = int(prefill_bytes * 8)
            if ba.cache_ratio > 0 and prefill_tokens <= 0:
                raise RuntimeError(
                    "cache_ratio>0 且启用 get prefill 时，预期 prefill put 有真实搬运，"
                    "但 get_prefill_put_tokens=0。"
                )

        if ba.measure_op in ("both", "get"):
            get_metrics, cached_tokens, all_tokens = _measure_get(
                kvmanager=kvmanager,
                batch_get_token_ids=batch_get_token_ids,
                batch_slot_mapping=batch_get_slot_mapping,
                batch_token_mask=batch_get_token_mask,
            )
            record.update(get_metrics)
            record["cached_tokens"] = cached_tokens
            record["cache_hit_percent"] = (100.0 * cached_tokens / all_tokens) if all_tokens else 0.0
            get_bytes = int(cached_tokens * token_size_bytes)
            get_bits = int(get_bytes * 8)
            record["get_transferred_bytes"] = get_bytes
            record["get_transferred_bits"] = get_bits
            if record["get_total_ms"] > 0:
                secs = record["get_total_ms"] / 1000.0
                get_gib_per_s = (get_bytes / (1024**3)) / secs
                get_gbit_per_s = (get_bits / (1000**3)) / secs
                record["get_effective_gib_per_s"] = get_gib_per_s
                record["get_effective_gbit_per_s"] = get_gbit_per_s
                # Keep legacy field for compatibility with existing scripts.
                record["get_effective_gbps"] = get_gib_per_s

            expect_get_transfer = (
                ba.cache_ratio > 0
                and (
                    ba.measure_op == "both"
                    or (ba.measure_op == "get" and ba.prime_cache_for_get)
                )
            )
            if expect_get_transfer and cached_tokens <= 0:
                raise RuntimeError(
                    "cache_ratio>0 且具备预填充条件时，预期 get 有真实搬运，"
                    "但 cached_tokens=0（可能命中链路未建立或被配置绕过）。"
                )

    finally:
        shutdown_tp_client(tp_processes)
        if kvmanager is not None:
            try:
                kvmanager.shutdown()
            except Exception:
                pass

    return record


def main() -> None:
    p = argparse.ArgumentParser(description="单次 KVManager 模式基准（stdout 一行 JSON）")
    p.add_argument(
        "--config",
        type=str,
        default="benchmarks/kvmanager_mode_compare/smoke_config_cpu_only.yml",
    )
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--sequence-length", type=int, default=1024)
    p.add_argument(
        "--cache-ratio",
        type=float,
        default=1.0,
        help=(
            "参与 put 写入的序列比例。1.0: 真实搬运路径（get 更可能含 H2D）；"
            "0.0: 控制路径模式，不做数据搬运但仍提交任务并测 match/launch/wait。"
        ),
    )
    p.add_argument(
        "--measure-op",
        type=str,
        default="both",
        choices=("both", "put", "get"),
        help="both: put+get（原行为）；put/get: 仅测单条路径",
    )
    p.add_argument(
        "--no-prime-cache-for-get",
        action="store_false",
        dest="prime_cache_for_get",
        help="measure-op=get 时不做预填充 put（默认会先预填充，避免全 miss）",
    )
    p.add_argument(
        "--ready-timeout-s",
        type=float,
        default=120.0,
        help="等待 KVManager.is_ready() 的超时秒数；<=0 表示不超时",
    )
    p.add_argument(
        "--transfer-manager-mode",
        type=str,
        default=os.getenv("FLEXKV_TRANSFER_MANAGER_MODE", "thread"),
        choices=("process", "thread"),
        help="benchmark 用的 transfer manager 模式；thread 在容器里通常更稳定",
    )
    p.add_argument(
        "--gpu-block-capacity-ratio",
        type=float,
        default=1.0,
        metavar="R",
        help=(
            "GPU 可注册 block 数相对序列所需块数的比例，(0,1]；"
            "默认 1.0 与原先一致；<1 时缩小池以促使 spill，get 更可能含下层→GPU 传输。"
            "direct 与 server_client 使用同一公式。"
        ),
    )
    p.set_defaults(prime_cache_for_get=True)
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: 需要 CUDA", file=sys.stderr)
        sys.exit(1)

    if args.gpu_block_capacity_ratio <= 0 or args.gpu_block_capacity_ratio > 1.0:
        print(
            "ERROR: --gpu-block-capacity-ratio 须在 (0, 1] 内",
            file=sys.stderr,
        )
        sys.exit(1)

    _log(
        f"mode={_mode_label()} pid={os.getpid()} "
        f"instance_id={GLOBAL_CONFIG_FROM_ENV.instance_id} "
        f"instance_num={GLOBAL_CONFIG_FROM_ENV.instance_num} "
        f"server_client_mode={GLOBAL_CONFIG_FROM_ENV.server_client_mode} "
        f"transfer_manager_mode={args.transfer_manager_mode}"
    )
    os.environ["FLEXKV_TRANSFER_MANAGER_MODE"] = args.transfer_manager_mode
    _ensure_global_config_compat()
    _patch_worker_kwargs_compat()
    _patch_worker_handle_del_compat()
    _patch_server_subprocess_compat()
    _apply_transfer_manager_mode_override(args.transfer_manager_mode)
    os.chdir(_FLEXKV_ROOT)
    model_config, cache_config = load_config(args.config)
    seq_len = args.sequence_length
    tpb = cache_config.tokens_per_block
    seq_len = ((seq_len - 1) // tpb + 1) * tpb

    ba = BenchArgs(
        batch_size=args.batch_size,
        sequence_length=seq_len,
        cache_ratio=args.cache_ratio,
        config_path=args.config,
        measure_op=args.measure_op,
        prime_cache_for_get=args.prime_cache_for_get,
        ready_timeout_s=args.ready_timeout_s,
        transfer_manager_mode=args.transfer_manager_mode,
        gpu_block_capacity_ratio=args.gpu_block_capacity_ratio,
    )

    record = run_benchmark(model_config, cache_config, ba)
    record["config_path"] = args.config
    print(_JSON_PREFIX + json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
