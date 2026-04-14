#!/usr/bin/env python3
"""
在独立子进程中分别跑 direct 与 server_client，汇总 JSONL。

  cd /path/to/FlexKV
  python3 benchmarks/kvmanager_mode_compare/run_compare.py --help
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


def _print_summary(summary: dict, modes: list[str], metrics: list[str]) -> None:
    print("\n# 聚合统计（measure 轮）", file=sys.stderr)
    for mode in modes:
        print(f"[{mode}]", file=sys.stderr)
        for k in metrics:
            s = summary.get(mode, {}).get(k)
            if not s:
                continue
            print(
                f"  {k}: mean={s['mean']:.3f}, median={s['median']:.3f}, p95={s['p95']:.3f}, n={s['count']}",
                file=sys.stderr,
            )


def _summary_delta_server_client_minus_direct(summary: dict, metrics: list[str]) -> dict:
    a = summary.get("direct", {})
    b = summary.get("server_client", {})
    delta: dict = {}
    for k in metrics:
        if k in a and k in b:
            delta[k] = b[k]["mean"] - a[k]["mean"]
    return delta


def _collect_measured_metrics(rows: list[dict]) -> list[str]:
    excluded = {
        "mode",
        "flexkv_server_client_mode",
        "flexkv_server_recv_port",
        "compare_session",
        "phase",
        "round_index",
        "config_path",
        "measure_op",
        "transfer_manager_mode",
        "batch_size",
        "sequence_length",
        "cache_ratio",
        "tp_size",
        "dp_size",
    }
    metrics: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k, v in row.items():
            if k in excluded:
                continue
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)) and k not in seen:
                seen.add(k)
                metrics.append(k)
    return metrics


def _write_csv_summary(
    *,
    csv_path: str,
    cache_ratio: float,
    repeats: int,
    metrics: list[str],
    summary: dict,
    delta: dict,
) -> None:
    row: dict[str, object] = {
        "cache_ratio": cache_ratio,
        "repeats": repeats,
    }
    fieldnames: list[str] = ["cache_ratio", "repeats"]

    for metric in metrics:
        direct_col = f"{metric}_direct(mean)"
        server_col = f"{metric}_server_client(mean)"
        delta_col = f"{metric}_delta_server_client_minus_direct"
        fieldnames.extend([direct_col, server_col, delta_col])

        direct_mean = summary.get("direct", {}).get(metric, {}).get("mean")
        server_mean = summary.get("server_client", {}).get(metric, {}).get("mean")

        row[direct_col] = "" if direct_mean is None else direct_mean
        row[server_col] = "" if server_mean is None else server_mean
        row[delta_col] = "" if metric not in delta else delta[metric]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def _run_one_mode(
    *,
    mode: str,
    ipc: str,
    flexkv_root: Path,
    config: str,
    batch_size: int,
    sequence_length: int,
    cache_ratio: float,
    transfer_manager_mode: str,
    ready_timeout_s: float,
    child_timeout_s: float,
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
        "--ready-timeout-s",
        str(ready_timeout_s),
        "--transfer-manager-mode",
        transfer_manager_mode,
    ]
    # 不再用 PIPE 捕获 stdout：server_client 下孙进程可能继承该 fd，导致 communicate() 等 EOF 卡住
    tmp_stdout = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f"flexkv_bench_{mode}_",
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
                    f"模式 {mode} 子进程超时（>{child_timeout_s:.1f}s），"
                    "可能卡在 KVManager.start()/is_ready() 或其子组件初始化。"
                    f"子进程完整 stdout 已保留：{tmp_stdout_path}"
                ) from e

        captured_stdout = tmp_stdout_path.read_text(encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            if captured_stdout:
                sys.stderr.write(captured_stdout)
            raise RuntimeError(
                f"模式 {mode} 子进程退出码 {proc.returncode}；"
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
                f"[run_compare] 保留子进程 stdout 日志用于排查: {tmp_stdout_path}",
                file=sys.stderr,
            )


def main() -> None:
    p = argparse.ArgumentParser(description="FlexKV direct vs server_client（无 vLLM）对比")
    p.add_argument(
        "--spec-only",
        action="store_true",
        help="仅提示 measurement_spec.py 路径后退出",
    )
    p.add_argument(
        "--config",
        type=str,
        default="benchmarks/kvmanager_mode_compare/smoke_config.yml",
        help="默认用轻量 CPU-only smoke 配置，避免部分环境里的 SSD worker/daemon 初始化冲突",
    )
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--sequence-length", type=int, default=1024)
    p.add_argument("--cache-ratio", type=float, default=0.0)
    p.add_argument(
        "--transfer-manager-mode",
        type=str,
        default="thread",
        choices=("process", "thread"),
        help="传给 bench_one_run；thread 在容器环境下通常更稳定",
    )
    p.add_argument(
        "--ready-timeout-s",
        type=float,
        default=120.0,
        help="透传给 bench_one_run：等待 is_ready() 超时秒数；<=0 表示不超时",
    )
    p.add_argument(
        "--child-timeout-s",
        type=float,
        default=300.0,
        help="run_compare 级别子进程总超时秒数；用于兜底 KVManager.start() 卡死",
    )
    p.add_argument("--warmup", type=int, default=0, help="每个模式先跑多少轮预热（不计入输出统计）")
    p.add_argument("--repeats", type=int, default=1, help="每个模式计入统计的轮数")
    p.add_argument(
        "--modes",
        type=str,
        default="direct,server_client",
        help="逗号分隔，例如 direct 或 server_client 或两者",
    )
    p.add_argument(
        "--ipc-base",
        type=str,
        default="ipc:///tmp/flexkv_kvmanager_mode_cmp",
        help="ZMQ IPC 前缀；每次运行会追加唯一后缀",
    )
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
        default="mode_compare_summary.csv",
        help=(
            "写入聚合 CSV（单行）。列格式：cache_ratio,repeats,"
            "<metric>_direct(mean),<metric>_server_client(mean),<metric>_delta_server_client_minus_direct"
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
    if args.repeats <= 0:
        p.error("--repeats 必须大于 0")

    session = uuid.uuid4().hex[:12]

    for mode in modes:
        if mode not in ("direct", "server_client"):
            p.error(f"未知 mode: {mode}，仅支持 direct / server_client")

    measured_rows: list[dict] = []
    total_rounds = args.warmup + args.repeats
    for ridx in range(total_rounds):
        is_warmup = ridx < args.warmup
        phase = "warmup" if is_warmup else "measure"

        # Alternate order across rounds to reduce drift bias.
        run_modes = list(modes)
        if len(run_modes) == 2 and (ridx % 2 == 1):
            run_modes = list(reversed(run_modes))

        for mode in run_modes:
            ipc = f"{args.ipc_base.rstrip('/')}_{session}_{phase}_{ridx}_{mode}"
            print(
                f"[run_compare][{phase} {ridx + 1}/{total_rounds}] mode={mode} started; child stderr below; JSON on stdout when done.",
                file=sys.stderr,
                flush=True,
            )
            row = _run_one_mode(
                mode=mode,
                ipc=ipc,
                flexkv_root=_FLEXKV_ROOT,
                config=args.config,
                batch_size=args.batch_size,
                sequence_length=args.sequence_length,
                cache_ratio=args.cache_ratio,
                transfer_manager_mode=args.transfer_manager_mode,
                ready_timeout_s=args.ready_timeout_s,
                child_timeout_s=args.child_timeout_s,
            )
            row["compare_session"] = session
            row["phase"] = phase
            row["round_index"] = ridx
            print(
                f"[run_compare][{phase} {ridx + 1}/{total_rounds}] 完成: mode={mode}",
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

    metrics = _collect_measured_metrics(measured_rows)
    summary = _summarize_by_mode(measured_rows, metrics, modes)
    _print_summary(summary, modes, metrics)

    delta = _summary_delta_server_client_minus_direct(summary, metrics)
    if delta:
        print("\n# 均值差值 (server_client - direct)", file=sys.stderr)
        for k in metrics:
            if k in delta:
                print(f"  {k}: {delta[k]:+.3f}", file=sys.stderr)

    if args.summary_output:
        payload = {
            "compare_session": session,
            "config": args.config,
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "cache_ratio": args.cache_ratio,
            "transfer_manager_mode": args.transfer_manager_mode,
            "ready_timeout_s": args.ready_timeout_s,
            "child_timeout_s": args.child_timeout_s,
            "modes": modes,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "runs": measured_rows,
            "summary": summary,
            "delta_server_client_minus_direct": delta,
        }
        with open(args.summary_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[run_compare] 写入 summary: {args.summary_output}", file=sys.stderr)

    if args.csv_output:
        _write_csv_summary(
            csv_path=args.csv_output,
            cache_ratio=args.cache_ratio,
            repeats=args.repeats,
            metrics=metrics,
            summary=summary,
            delta=delta,
        )
        print(f"[run_compare] 写入 CSV: {args.csv_output}", file=sys.stderr)


if __name__ == "__main__":
    main()
