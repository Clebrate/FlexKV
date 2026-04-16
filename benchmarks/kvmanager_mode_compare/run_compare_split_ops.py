#!/usr/bin/env python3
"""
将 get / put 拆开做独立重复实验（默认各 500 轮），避免同一轮串行测量互相影响。

CSV：
  --csv-output：每完成一轮 measure 追加一行（实时 flush），便于 tail 查看进度。
  --csv-aggregate-output：按 operation 两行聚合，每轮后覆盖刷新，中断也可保留最新均值。

用法示例（在 FlexKV 根目录）:
  python3 benchmarks/kvmanager_mode_compare/run_compare_split_ops.py \
    --get-repeats 500 --put-repeats 500
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

_FLEXKV_ROOT = Path(__file__).resolve().parents[2]
_BENCH_ONE = Path(__file__).resolve().parent / "bench_one_run.py"


def _parse_bench_json(stdout: str) -> dict:
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("FLEXKV_BENCH_JSON "):
            return json.loads(line[len("FLEXKV_BENCH_JSON ") :])
    raise ValueError("子进程输出中未找到 FLEXKV_BENCH_JSON 行:\n" + stdout[:2000])


def _percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("empty values")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    w = pos - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


def _summarize_by_mode(rows: list[dict], metrics: list[str], modes: list[str]) -> dict:
    summary: dict = {}
    for mode in modes:
        mode_rows = [r for r in rows if r.get("mode") == mode]
        metric_stats: dict = {}
        for k in metrics:
            vals = [float(r[k]) for r in mode_rows if k in r]
            if not vals:
                continue
            n = len(vals)
            mean_v = sum(vals) / n
            var_v = sum((x - mean_v) ** 2 for x in vals) / n
            metric_stats[k] = {
                "count": n,
                "mean": mean_v,
                "median": _percentile(vals, 0.5),
                "p95": _percentile(vals, 0.95),
                "min": min(vals),
                "max": max(vals),
                "std": math.sqrt(var_v),
            }
        summary[mode] = metric_stats
    return summary


def _print_summary(summary_by_op: dict, modes: list[str], metrics_by_op: dict[str, list[str]]) -> None:
    print("\n# 拆分实验聚合统计（measure 轮）", file=sys.stderr)
    for op in ("get", "put"):
        if op not in summary_by_op:
            continue
        print(f"\n## operation={op}", file=sys.stderr)
        for mode in modes:
            print(f"[{mode}]", file=sys.stderr)
            for k in metrics_by_op[op]:
                s = summary_by_op.get(op, {}).get(mode, {}).get(k)
                if not s:
                    continue
                print(
                    f"  {k}: mean={s['mean']:.3f}, median={s['median']:.3f}, p95={s['p95']:.3f}, n={s['count']}",
                    file=sys.stderr,
                )


def _summary_delta_server_client_minus_direct(summary_by_op: dict, metrics_by_op: dict[str, list[str]]) -> dict:
    delta: dict = {}
    for op, metrics in metrics_by_op.items():
        a = summary_by_op.get(op, {}).get("direct", {})
        b = summary_by_op.get(op, {}).get("server_client", {})
        op_delta: dict = {}
        for k in metrics:
            if k in a and k in b:
                op_delta[k] = b[k]["mean"] - a[k]["mean"]
        if op_delta:
            delta[op] = op_delta
    return delta


def _write_csv_summary_by_operation(
    *,
    csv_path: str,
    cache_ratio: float,
    get_repeats: int,
    put_repeats: int,
    summary_by_op: dict,
    delta: dict,
    metrics_by_op: dict[str, list[str]],
) -> None:
    base_cols = ["cache_ratio", "get_repeats", "put_repeats", "operation"]

    metric_cols: list[str] = []
    for op in ("get", "put"):
        for metric in metrics_by_op.get(op, []):
            metric_cols.extend(
                [
                    f"{metric}_direct(mean)",
                    f"{metric}_server_client(mean)",
                    f"{metric}_delta_server_client_minus_direct",
                ]
            )

    fieldnames = base_cols + metric_cols
    rows: list[dict[str, object]] = []

    for op in ("get", "put"):
        row: dict[str, object] = {
            "cache_ratio": cache_ratio,
            "get_repeats": get_repeats,
            "put_repeats": put_repeats,
            "operation": op,
        }
        for metric in metrics_by_op.get(op, []):
            direct_col = f"{metric}_direct(mean)"
            server_col = f"{metric}_server_client(mean)"
            delta_col = f"{metric}_delta_server_client_minus_direct"

            direct_mean = summary_by_op.get(op, {}).get("direct", {}).get(metric, {}).get("mean")
            server_mean = summary_by_op.get(op, {}).get("server_client", {}).get(metric, {}).get("mean")
            delta_val = delta.get(op, {}).get(metric)

            row[direct_col] = "" if direct_mean is None else direct_mean
            row[server_col] = "" if server_mean is None else server_mean
            row[delta_col] = "" if delta_val is None else delta_val

        rows.append(row)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _csv_run_row_flat() -> list[str]:
    base = [
        "cache_ratio",
        "get_repeats",
        "put_repeats",
        "compare_session",
        "operation",
        "phase",
        "round_index",
        "mode",
        "gpu_block_capacity_ratio",
        "num_gpu_blocks_required",
        "num_gpu_blocks",
    ]
    metrics = [
        "ready_total_ms",
        "put_submit_ms",
        "put_wait_ms",
        "put_total_ms",
        "put_effective_gbps",
        "get_prefill_put_total_ms",
        "get_match_ms",
        "get_launch_ms",
        "get_wait_ms",
        "get_total_ms",
        "cache_hit_percent",
        "get_effective_gbps",
        "mgr_construct_and_start_ms",
        "tp_register_until_ready_ms",
    ]
    return base + metrics


def _append_csv_run_row(
    *,
    csv_path: str,
    cache_ratio: float,
    get_repeats: int,
    put_repeats: int,
    row: dict,
) -> None:
    """Append one measurement row to CSV; write header if file is new or empty."""
    fieldnames = _csv_run_row_flat()
    out_row: dict[str, object] = {}
    out_row["cache_ratio"] = cache_ratio
    out_row["get_repeats"] = get_repeats
    out_row["put_repeats"] = put_repeats
    out_row["compare_session"] = row.get("compare_session", "")
    out_row["operation"] = row.get("operation", "")
    out_row["phase"] = row.get("phase", "")
    out_row["round_index"] = row.get("round_index", "")
    out_row["mode"] = row.get("mode", "")
    out_row["gpu_block_capacity_ratio"] = row.get("gpu_block_capacity_ratio", "")
    out_row["num_gpu_blocks_required"] = row.get("num_gpu_blocks_required", "")
    out_row["num_gpu_blocks"] = row.get("num_gpu_blocks", "")

    _base_len = 11
    for k in fieldnames[_base_len:]:
        v = row.get(k)
        if v is None:
            out_row[k] = ""
        elif isinstance(v, (int, float)):
            out_row[k] = v
        else:
            out_row[k] = str(v)

    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(out_row)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass


def _run_one_mode(
    *,
    mode: str,
    measure_op: str,
    ipc: str,
    flexkv_root: Path,
    config: str,
    batch_size: int,
    sequence_length: int,
    cache_ratio: float,
    prime_cache_for_get: bool,
    ready_timeout_s: float,
    child_timeout_s: float,
    transfer_manager_mode: str,
    gpu_block_capacity_ratio: float,
) -> dict:
    env = os.environ.copy()
    env["FLEXKV_SERVER_CLIENT_MODE"] = "1" if mode == "server_client" else "0"
    env["FLEXKV_SERVER_RECV_PORT"] = ipc
    # Benchmark runs are single-instance by design; avoid inheriting distributed env.
    env["FLEXKV_INSTANCE_NUM"] = "1"
    env["FLEXKV_INSTANCE_ID"] = "0"
    env["FLEXKV_TRANSFER_MANAGER_MODE"] = transfer_manager_mode
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        sys.executable,
        str(_BENCH_ONE),
        "--config",
        config,
        "--batch-size",
        str(batch_size),
        "--sequence-length",
        str(sequence_length),
        "--cache-ratio",
        str(cache_ratio),
        "--measure-op",
        measure_op,
        "--ready-timeout-s",
        str(ready_timeout_s),
        "--transfer-manager-mode",
        transfer_manager_mode,
        "--gpu-block-capacity-ratio",
        str(gpu_block_capacity_ratio),
    ]
    if measure_op == "get" and not prime_cache_for_get:
        cmd.append("--no-prime-cache-for-get")

    tmp_stdout = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f"flexkv_split_{measure_op}_{mode}_",
        suffix=".stdout.log",
        delete=False,
    )
    tmp_stdout_path = Path(tmp_stdout.name)
    tmp_stdout.close()

    success = False
    try:
        with open(tmp_stdout_path, "w", encoding="utf-8") as out_f:
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(flexkv_root),
                    env=env,
                    stdout=out_f,
                    stderr=None,
                    text=True,
                    check=False,
                    timeout=child_timeout_s if child_timeout_s > 0 else None,
                )
            except subprocess.TimeoutExpired as e:
                captured_stdout = tmp_stdout_path.read_text(
                    encoding="utf-8", errors="replace"
                )
                if captured_stdout:
                    sys.stderr.write(captured_stdout)
                raise TimeoutError(
                    f"模式 {mode} op={measure_op} 子进程超时（>{child_timeout_s:.1f}s），"
                    "可能卡在 KVManager.start()/is_ready() 或其子组件初始化。"
                    f"子进程完整 stdout 已保留：{tmp_stdout_path}"
                ) from e

        captured_stdout = tmp_stdout_path.read_text(encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            if captured_stdout:
                sys.stderr.write(captured_stdout)
            raise RuntimeError(
                f"模式 {mode} op={measure_op} 子进程退出码 {proc.returncode}；"
                f"子进程完整 stdout：{tmp_stdout_path}"
            )
        parsed = _parse_bench_json(captured_stdout)
        success = True
        return parsed
    finally:
        if success:
            try:
                tmp_stdout_path.unlink()
            except OSError:
                pass
        else:
            print(
                f"[run_compare_split_ops] 保留子进程 stdout 日志用于排查: {tmp_stdout_path}",
                file=sys.stderr,
            )


def main() -> None:
    p = argparse.ArgumentParser(description="FlexKV direct vs server_client 拆分 get/put 对比")
    p.add_argument(
        "--spec-only",
        action="store_true",
        help="仅提示 measurement_spec.py 路径后退出",
    )
    p.add_argument(
        "--config",
        type=str,
        default="benchmarks/kvmanager_mode_compare/smoke_config.yml",
    )
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--sequence-length", type=int, default=1024)
    p.add_argument(
        "--cache-ratio",
        type=float,
        default=1.0,
        help=(
            "透传给 bench_one_run。1.0: 真实搬运路径；"
            "0.0: 控制路径模式（仍提交任务并测 match/launch/wait）。"
        ),
    )
    p.add_argument("--warmup", type=int, default=0, help="每个 operation 每个 mode 先跑多少轮预热")
    p.add_argument("--get-repeats", type=int, default=500, help="get 计入统计的轮数")
    p.add_argument("--put-repeats", type=int, default=500, help="put 计入统计的轮数")
    p.add_argument(
        "--modes",
        type=str,
        default="direct,server_client",
        help="逗号分隔，例如 direct 或 server_client 或两者",
    )
    p.add_argument(
        "--ipc-base",
        type=str,
        default="ipc:///tmp/flexkv_kvmanager_mode_split",
        help="ZMQ IPC 前缀；每次运行会追加唯一后缀",
    )
    p.add_argument(
        "--no-prime-cache-for-get",
        action="store_false",
        dest="prime_cache_for_get",
        help="get 轮次不做预填充 put（默认预填充，避免全 miss）",
    )
    p.add_argument(
        "--ready-timeout-s",
        type=float,
        default=120.0,
        help="传给 bench_one_run：等待 is_ready() 超时秒数",
    )
    p.add_argument(
        "--child-timeout-s",
        type=float,
        default=300.0,
        help="run_compare_split_ops 级别子进程总超时秒数；用于兜底 KVManager.start() 卡死",
    )
    p.add_argument(
        "--transfer-manager-mode",
        type=str,
        default="thread",
        choices=("process", "thread"),
        help="传给 bench_one_run；thread 在容器环境下通常更稳定",
    )
    p.add_argument(
        "--gpu-block-capacity-ratio",
        type=float,
        default=1.0,
        metavar="R",
        help=(
            "透传 bench_one_run：(0,1] 缩小 GPU block 池以促使 get 含下层→GPU 传输；"
            "1.0 保持原行为；direct/server_client 使用同一值。"
        ),
    )
    p.set_defaults(prime_cache_for_get=True)
    p.add_argument("-o", "--output", type=str, default="", help="可选：写入 JSONL（每行一条）")
    p.add_argument(
        "--summary-output",
        type=str,
        default="",
        help="可选：写入聚合 JSON（含 runs + summary + delta）",
    )
    p.add_argument(
        "--csv-output",
        type=str,
        default="mode_compare_split_ops_runs.csv",
        help=(
            "每完成一轮 measure 即追加一行原始指标（实时写入并 flush）；"
            "设为空字符串可关闭"
        ),
    )
    p.add_argument(
        "--csv-aggregate-output",
        type=str,
        default="mode_compare_split_ops_summary.csv",
        help=(
            "按 operation 写两行聚合（direct/server_client mean 与 delta）；"
            "每轮 measure 后覆盖刷新一次，便于中断后仍保留最新聚合；设为空可关闭"
        ),
    )
    args = p.parse_args()

    if args.spec_only:
        print("测量范围: benchmarks/kvmanager_mode_compare/measurement_spec.py")
        return

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if not modes:
        p.error("--modes 不能为空")
    if args.warmup < 0:
        p.error("--warmup 不能小于 0")
    if args.get_repeats <= 0:
        p.error("--get-repeats 必须大于 0")
    if args.put_repeats <= 0:
        p.error("--put-repeats 必须大于 0")
    if args.gpu_block_capacity_ratio <= 0 or args.gpu_block_capacity_ratio > 1.0:
        p.error("--gpu-block-capacity-ratio 须在 (0, 1] 内")

    for mode in modes:
        if mode not in ("direct", "server_client"):
            p.error(f"未知 mode: {mode}，仅支持 direct / server_client")

    session = uuid.uuid4().hex[:12]
    measured_rows: list[dict] = []

    metrics_by_op: dict[str, list[str]] = {
        "get": [
            "get_prefill_put_total_ms",
            "get_match_ms",
            "get_launch_ms",
            "get_wait_ms",
            "get_total_ms",
            "cache_hit_percent",
            "get_effective_gbps",
        ],
        "put": [
            "put_submit_ms",
            "put_wait_ms",
            "put_total_ms",
            "put_effective_gbps",
        ],
    }

    phases = [("get", args.get_repeats), ("put", args.put_repeats)]
    for operation, repeats in phases:
        total_rounds = args.warmup + repeats
        for ridx in range(total_rounds):
            is_warmup = ridx < args.warmup
            phase = "warmup" if is_warmup else "measure"

            run_modes = list(modes)
            if len(run_modes) == 2 and (ridx % 2 == 1):
                run_modes = list(reversed(run_modes))

            for mode in run_modes:
                ipc = f"{args.ipc_base.rstrip('/')}_{session}_{operation}_{phase}_{ridx}_{mode}"
                print(
                    f"[run_compare_split_ops][{operation}][{phase} {ridx + 1}/{total_rounds}] "
                    f"mode={mode} started; child stderr below; JSON on stdout when done.",
                    file=sys.stderr,
                    flush=True,
                )
                row = _run_one_mode(
                    mode=mode,
                    measure_op=operation,
                    ipc=ipc,
                    flexkv_root=_FLEXKV_ROOT,
                    config=args.config,
                    batch_size=args.batch_size,
                    sequence_length=args.sequence_length,
                    cache_ratio=args.cache_ratio,
                    prime_cache_for_get=args.prime_cache_for_get,
                    ready_timeout_s=args.ready_timeout_s,
                    child_timeout_s=args.child_timeout_s,
                    transfer_manager_mode=args.transfer_manager_mode,
                    gpu_block_capacity_ratio=args.gpu_block_capacity_ratio,
                )
                row["compare_session"] = session
                row["phase"] = phase
                row["round_index"] = ridx
                row["operation"] = operation
                print(
                    f"[run_compare_split_ops][{operation}][{phase} {ridx + 1}/{total_rounds}] 完成: mode={mode}",
                    file=sys.stderr,
                    flush=True,
                )

                if not is_warmup:
                    measured_rows.append(row)
                    line = json.dumps(row, ensure_ascii=False)
                    print(line)
                    if args.output:
                        with open(args.output, "a", encoding="utf-8") as f:
                            f.write(line + "\n")

                    if args.csv_output:
                        _append_csv_run_row(
                            csv_path=args.csv_output,
                            cache_ratio=args.cache_ratio,
                            get_repeats=args.get_repeats,
                            put_repeats=args.put_repeats,
                            row=row,
                        )

                    if args.csv_aggregate_output:
                        summary_by_op_partial: dict = {}
                        for op, metrics in metrics_by_op.items():
                            op_rows = [r for r in measured_rows if r.get("operation") == op]
                            summary_by_op_partial[op] = _summarize_by_mode(
                                op_rows, metrics, modes
                            )
                        delta_partial = _summary_delta_server_client_minus_direct(
                            summary_by_op_partial, metrics_by_op
                        )
                        _write_csv_summary_by_operation(
                            csv_path=args.csv_aggregate_output,
                            cache_ratio=args.cache_ratio,
                            get_repeats=args.get_repeats,
                            put_repeats=args.put_repeats,
                            summary_by_op=summary_by_op_partial,
                            delta=delta_partial,
                            metrics_by_op=metrics_by_op,
                        )

    summary_by_op: dict = {}
    for op, metrics in metrics_by_op.items():
        op_rows = [r for r in measured_rows if r.get("operation") == op]
        summary_by_op[op] = _summarize_by_mode(op_rows, metrics, modes)

    _print_summary(summary_by_op, modes, metrics_by_op)

    delta = _summary_delta_server_client_minus_direct(summary_by_op, metrics_by_op)
    if delta:
        print("\n# 均值差值 (server_client - direct)", file=sys.stderr)
        for op in ("get", "put"):
            if op not in delta:
                continue
            print(f"[{op}]", file=sys.stderr)
            for k in metrics_by_op[op]:
                if k in delta[op]:
                    print(f"  {k}: {delta[op][k]:+.3f}", file=sys.stderr)

    if args.summary_output:
        payload = {
            "compare_session": session,
            "config": args.config,
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "cache_ratio": args.cache_ratio,
            "modes": modes,
            "warmup": args.warmup,
            "get_repeats": args.get_repeats,
            "put_repeats": args.put_repeats,
            "prime_cache_for_get": args.prime_cache_for_get,
            "ready_timeout_s": args.ready_timeout_s,
            "child_timeout_s": args.child_timeout_s,
            "transfer_manager_mode": args.transfer_manager_mode,
            "csv_runs_output": args.csv_output or None,
            "csv_aggregate_output": args.csv_aggregate_output or None,
            "runs": measured_rows,
            "summary_by_operation": summary_by_op,
            "delta_server_client_minus_direct": delta,
        }
        with open(args.summary_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[run_compare_split_ops] 写入 summary: {args.summary_output}", file=sys.stderr)

    if args.csv_output:
        print(
            f"[run_compare_split_ops] 逐轮 CSV（追加）: {args.csv_output}",
            file=sys.stderr,
        )
    if args.csv_aggregate_output:
        print(
            f"[run_compare_split_ops] 聚合 CSV（已随轮次刷新）: {args.csv_aggregate_output}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
