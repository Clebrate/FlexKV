import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional


NVTX_KEYS = {
    "api_get_match_ms": "flexkv.onboard.api.get_match",
    "api_wait_ms": "flexkv.onboard.api.wait",
    "scheduler_set_slot_mappings_ms": "flexkv.onboard.scheduler.set_slot_mappings",
    "scheduler_submit_graphs_ms": "flexkv.onboard.scheduler.submit_graphs",
}


def _run(cmd: List[str], log_file: Optional[Path] = None) -> int:
    if log_file is None:
        return subprocess.run(cmd, text=True).returncode
    with log_file.open("w") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True).returncode


def _find_stats_csv(prefix: Path) -> Optional[Path]:
    candidates = sorted(prefix.parent.glob(prefix.name + "*.csv"))
    return candidates[-1] if candidates else None


def _read_nvtx_stats(csv_path: Path) -> Dict[str, float]:
    """Return summed NVTX durations in milliseconds by range name.

    nsys CSV headers vary a bit by version/report. This parser finds the first
    header row with a total-time column and treats the final string-like column
    as the NVTX range name.
    """
    rows = list(csv.reader(csv_path.open()))
    header_idx = -1
    for i, row in enumerate(rows):
        normalized = [c.strip() for c in row]
        if any("Total Time" in c for c in normalized) and (
            any(c in ("Range", "Name", "Message") for c in normalized)
            or len(normalized) >= 2
        ):
            header_idx = i
            break
    if header_idx < 0:
        return {}

    header = [c.strip() for c in rows[header_idx]]
    total_idx = next(i for i, c in enumerate(header) if "Total Time" in c)

    name_idx = None
    for preferred in ("Range", "Name", "Message"):
        if preferred in header:
            name_idx = header.index(preferred)
            break
    if name_idx is None:
        name_idx = len(header) - 1

    scale = 1e-6
    total_col = header[total_idx].lower()
    if "(us)" in total_col:
        scale = 1e-3
    elif "(ms)" in total_col:
        scale = 1.0
    elif "(s)" in total_col and "(ns)" not in total_col and "(us)" not in total_col:
        scale = 1e3

    result: Dict[str, float] = {}
    for row in rows[header_idx + 1 :]:
        if len(row) <= max(total_idx, name_idx):
            continue
        name = row[name_idx].strip()
        raw_total = row[total_idx].strip().replace(",", "")
        if not name or not raw_total:
            continue
        try:
            total_ms = float(raw_total) * scale
        except ValueError:
            continue
        result[name] = result.get(name, 0.0) + total_ms
    return result


def _sum_prefix(stats: Dict[str, float], prefix: str) -> float:
    return sum(v for k, v in stats.items() if k.startswith(prefix))


def _sum_contains(stats: Dict[str, float], needle: str) -> float:
    return sum(v for k, v in stats.items() if needle in k)


def _extract_benchmark_total_ms(log_file: Path) -> str:
    text = log_file.read_text(errors="replace")
    matches = re.findall(r"total_ms=([-0-9.]+)", text)
    if matches:
        return matches[-1]
    matches = re.findall(r"e2e time:\s*([-0-9.]+)ms", text)
    return matches[-1] if matches else "NA"


def _extract_get_datasize_gb(log_file: Path) -> str:
    text = log_file.read_text(errors="replace")
    matches = re.findall(r"get\s+\d+\s+tokens,\s+data_size:\s*([-0-9.]+)\s+GB", text)
    return matches[-1] if matches else "NA"


def _profile_case(
    args: argparse.Namespace,
    mode: str,
    batch_size: int,
    sequence_length: int,
) -> Dict[str, object]:
    tag = f"{mode}_bs{batch_size}_seq{sequence_length}"
    rep_prefix = args.out_dir / "nsys" / tag
    log_file = args.out_dir / "logs" / f"{tag}.log"
    stats_prefix = args.out_dir / "nvtx_csv" / tag
    final_rep = rep_prefix.with_suffix(".nsys-rep")

    row: Dict[str, object] = {
        "mode": mode,
        "batch_size": batch_size,
        "seq_len": sequence_length,
        "sequence_length": sequence_length,
        "datasize_gb": "NA",
        "e2e_latency_ms": "NA",
        "transfer_latency_ms": "NA",
        "overhead_ms": "NA",
        "effective_bandwidth_gbps": "NA",
        "exit_code": "NA",
        "benchmark_total_ms": "NA",
        "nsys_rep": str(final_rep),
        "log_file": str(log_file),
        "nvtx_csv": "NA",
    }

    profile_cmd = [
        "nsys",
        "profile",
        "--trace-fork-before-exec=true",
        "-t",
        "cuda,nvtx,osrt",
        "--force-overwrite=true",
        "-o",
        str(rep_prefix),
        sys.executable,
        "benchmarks/benchmark_single_batch.py",
        "--config",
        args.config,
        "--batch-size",
        str(batch_size),
        "--sequence-length",
        str(sequence_length),
    ]
    if mode == "layerwise":
        profile_cmd.append("--layerwise-transfer")

    print(f"[SWEEP] start {tag}", flush=True)
    exit_code = _run(profile_cmd, log_file)
    row["exit_code"] = exit_code
    row["benchmark_total_ms"] = _extract_benchmark_total_ms(log_file) if log_file.exists() else "NA"
    row["datasize_gb"] = _extract_get_datasize_gb(log_file) if log_file.exists() else "NA"

    stats: Dict[str, float] = {}
    nvtx_csv = None
    if final_rep.exists():
        stats_cmd = [
            "nsys",
            "stats",
            "--force-export=true",
            "--report",
            "nvtx_sum",
            "--format",
            "csv",
            "--output",
            str(stats_prefix),
            str(final_rep),
        ]
        stats_exit = _run(stats_cmd, args.out_dir / "logs" / f"{tag}.nsys_stats.log")
        row["nsys_stats_exit_code"] = stats_exit
        nvtx_csv = _find_stats_csv(stats_prefix)
        if nvtx_csv is not None:
            row["nvtx_csv"] = str(nvtx_csv)
            stats = _read_nvtx_stats(nvtx_csv)
    else:
        row["nsys_stats_exit_code"] = "NO_REP"

    mode_launch_key = f"flexkv.onboard.{mode}.api.launch"
    mode_build_key = f"flexkv.onboard.scheduler.build_graph.{mode}"
    transfer_prefix = (
        "flexkv.onboard.baseline.transfer_kv_blocks.H2D"
        if mode == "baseline"
        else "flexkv.onboard.layerwise.transfer_group"
    )

    row["api_get_match_ms"] = _sum_contains(stats, NVTX_KEYS["api_get_match_ms"])
    row["api_launch_ms"] = _sum_contains(stats, mode_launch_key)
    row["api_wait_ms"] = _sum_contains(stats, NVTX_KEYS["api_wait_ms"])
    row["scheduler_set_slot_mappings_ms"] = _sum_contains(stats, NVTX_KEYS["scheduler_set_slot_mappings_ms"])
    row["scheduler_build_graph_ms"] = _sum_contains(stats, mode_build_key)
    row["scheduler_submit_graphs_ms"] = _sum_contains(stats, NVTX_KEYS["scheduler_submit_graphs_ms"])
    row["get_match_loop_ms"] = _sum_prefix(stats, f"flexkv.onboard.{mode}.get_match_loop")
    row["launch_call_ms"] = _sum_prefix(stats, f"flexkv.onboard.{mode}.launch_call")
    row["wait_call_ms"] = _sum_prefix(stats, f"flexkv.onboard.{mode}.wait_call")
    row["eventfd_wait_ms"] = _sum_prefix(
        stats, "flexkv.onboard.layerwise.eventfd_wait.total"
    )
    row["eventfd_poll_ms"] = _sum_prefix(
        stats, "flexkv.onboard.layerwise.eventfd_poll.layer"
    )
    row["eventfd_read_ms"] = _sum_prefix(
        stats, "flexkv.onboard.layerwise.eventfd_read.layer"
    )
    row["callback_total_ms"] = _sum_prefix(
        stats, "flexkv.onboard.layerwise.callback.total"
    )
    row["callback_eventfd_write_ms"] = _sum_prefix(
        stats, "flexkv.onboard.layerwise.callback.eventfd_write"
    )
    row["callback_nvtx_bookkeeping_ms"] = _sum_prefix(
        stats, "flexkv.onboard.layerwise.callback.nvtx_bookkeeping"
    )
    row["layer_h2d_ms"] = _sum_prefix(stats, "CPU->GPU Layer[")
    row["transfer_ms"] = _sum_prefix(stats, transfer_prefix)
    row["onboard_total_ms"] = _sum_prefix(stats, f"flexkv.onboard.{mode}.e2e")
    row["e2e_latency_ms"] = row["onboard_total_ms"]
    row["transfer_latency_ms"] = row["transfer_ms"]
    row["overhead_ms"] = max(float(row["onboard_total_ms"]) - float(row["transfer_ms"]), 0.0)
    row["worker_protocol_overhead_ms"] = max(
        float(row["transfer_ms"]) - float(row["layer_h2d_ms"]),
        0.0,
    )
    if row["datasize_gb"] != "NA" and float(row["onboard_total_ms"]) > 0:
        row["effective_bandwidth_gbps"] = (
            float(row["datasize_gb"]) / (float(row["onboard_total_ms"]) / 1000.0)
        )
    row["transfer_ratio"] = (
        float(row["transfer_ms"]) / float(row["onboard_total_ms"])
        if float(row["onboard_total_ms"]) > 0
        else 0.0
    )
    row["overhead_ratio"] = (
        float(row["overhead_ms"]) / float(row["onboard_total_ms"])
        if float(row["onboard_total_ms"]) > 0
        else 0.0
    )
    row["eventfd_wait_ratio"] = (
        float(row["eventfd_wait_ms"]) / float(row["overhead_ms"])
        if float(row["overhead_ms"]) > 0
        else 0.0
    )
    row["wait_after_transfer_ms"] = max(
        float(row["wait_call_ms"]) - float(row["transfer_ms"]),
        0.0,
    )
    accounted_overhead = (
        float(row["get_match_loop_ms"])
        + float(row["launch_call_ms"])
        + float(row["eventfd_wait_ms"])
        + float(row["wait_after_transfer_ms"])
    )
    row["unclassified_overhead_ms"] = max(
        float(row["overhead_ms"]) - accounted_overhead,
        0.0,
    )

    print(
        f"[SWEEP] done {tag} exit={exit_code} "
        f"onboard_ms={float(row['onboard_total_ms']):.3f} "
        f"transfer_ms={float(row['transfer_ms']):.3f} "
        f"ratio={float(row['transfer_ratio']):.3f}",
        flush=True,
    )
    return row


def _write_summary(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_overhead_breakdown(path: Path, rows: List[Dict[str, object]]) -> None:
    fields = [
        "mode",
        "batch_size",
        "seq_len",
        "datasize_gb",
        "e2e_latency_ms",
        "transfer_latency_ms",
        "overhead_ms",
        "get_match_loop_ms",
        "launch_call_ms",
        "scheduler_set_slot_mappings_ms",
        "scheduler_build_graph_ms",
        "scheduler_submit_graphs_ms",
        "eventfd_wait_ms",
        "eventfd_poll_ms",
        "eventfd_read_ms",
        "callback_total_ms",
        "callback_eventfd_write_ms",
        "callback_nvtx_bookkeeping_ms",
        "layer_h2d_ms",
        "worker_protocol_overhead_ms",
        "wait_call_ms",
        "wait_after_transfer_ms",
        "unclassified_overhead_ms",
        "effective_bandwidth_gbps",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "NA") for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config", default="benchmarks/local_8l.yml")
    parser.add_argument("--batch-sizes", default="1,2,4,8,16")
    parser.add_argument("--sequence-lengths", default="1024,2048,4096,8192")
    parser.add_argument("--summary", default="summary_from_nsys.csv")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "logs").mkdir(exist_ok=True)
    (args.out_dir / "nsys").mkdir(exist_ok=True)
    (args.out_dir / "nvtx_csv").mkdir(exist_ok=True)

    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x]
    sequence_lengths = [int(x) for x in args.sequence_lengths.split(",") if x]

    rows: List[Dict[str, object]] = []
    summary_path = args.out_dir / args.summary
    overhead_breakdown_path = args.out_dir / "overhead_breakdown.csv"
    for batch_size in batch_sizes:
        for sequence_length in sequence_lengths:
            for mode in ("baseline", "layerwise"):
                row = _profile_case(args, mode, batch_size, sequence_length)
                rows.append(row)
                _write_summary(summary_path, rows)
                _write_overhead_breakdown(overhead_breakdown_path, rows)

    print(f"[SWEEP] summary {summary_path}", flush=True)
    print(f"[SWEEP] overhead_breakdown {overhead_breakdown_path}", flush=True)


if __name__ == "__main__":
    main()
