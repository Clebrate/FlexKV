#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _group_by_mode(rows: List[dict], modes: List[str]) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = {m: [] for m in modes}
    for r in rows:
        m = r.get("mode")
        if m in grouped:
            grouped[m].append(r)
    return grouped


def _mean(grouped: Dict[str, List[dict]], mode: str, key: str) -> float:
    vals = [float(r[key]) for r in grouped.get(mode, []) if key in r]
    return float(np.mean(vals)) if vals else 0.0


def _std(grouped: Dict[str, List[dict]], mode: str, key: str) -> float:
    vals = [float(r[key]) for r in grouped.get(mode, []) if key in r]
    return float(np.std(vals)) if vals else 0.0


def _make_ready_breakdown(grouped: Dict[str, List[dict]], modes: List[str], out: Path) -> None:
    x = np.arange(len(modes))
    w = 0.6
    mgr = [_mean(grouped, m, "mgr_construct_and_start_ms") for m in modes]
    reg = [_mean(grouped, m, "tp_register_until_ready_ms") for m in modes]
    total = [_mean(grouped, m, "ready_total_ms") for m in modes]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, mgr, w, label="mgr_construct_and_start_ms")
    ax.bar(x, reg, w, bottom=mgr, label="tp_register_until_ready_ms")
    ax.plot(x, total, "ko--", label="ready_total_ms (mean)")

    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylabel("Milliseconds")
    ax.set_title("Ready Phase Breakdown (Mean)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _make_put_get_breakdown(grouped: Dict[str, List[dict]], modes: List[str], out: Path) -> None:
    x = np.arange(len(modes))
    w = 0.6

    put_submit = [_mean(grouped, m, "put_submit_ms") for m in modes]
    put_wait = [_mean(grouped, m, "put_wait_ms") for m in modes]

    get_match = [_mean(grouped, m, "get_match_ms") for m in modes]
    get_launch = [_mean(grouped, m, "get_launch_ms") for m in modes]
    get_wait = [_mean(grouped, m, "get_wait_ms") for m in modes]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.bar(x, put_submit, w, label="put_submit_ms")
    ax.bar(x, put_wait, w, bottom=put_submit, label="put_wait_ms")
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylabel("Milliseconds")
    ax.set_title("Put Breakdown (Mean)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    ax.bar(x, get_match, w, label="get_match_ms")
    btm = np.array(get_match)
    ax.bar(x, get_launch, w, bottom=btm, label="get_launch_ms")
    btm = btm + np.array(get_launch)
    ax.bar(x, get_wait, w, bottom=btm, label="get_wait_ms")
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylabel("Milliseconds")
    ax.set_title("Get Breakdown (Mean)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _make_latency_compare(grouped: Dict[str, List[dict]], modes: List[str], out: Path) -> None:
    metrics = ["ready_total_ms", "put_total_ms", "get_total_ms"]
    x = np.arange(len(metrics))
    w = 0.35

    vals_a = [_mean(grouped, modes[0], m) for m in metrics]
    vals_b = [_mean(grouped, modes[1], m) for m in metrics]
    std_a = [_std(grouped, modes[0], m) for m in metrics]
    std_b = [_std(grouped, modes[1], m) for m in metrics]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, vals_a, w, yerr=std_a, capsize=4, label=modes[0])
    ax.bar(x + w / 2, vals_b, w, yerr=std_b, capsize=4, label=modes[1])
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Milliseconds")
    ax.set_title("Latency Compare (Mean ± Std)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _make_throughput_compare(grouped: Dict[str, List[dict]], modes: List[str], out: Path) -> None:
    metrics = ["put_effective_gbps", "get_effective_gbps"]
    x = np.arange(len(metrics))
    w = 0.35

    vals_a = [_mean(grouped, modes[0], m) for m in metrics]
    vals_b = [_mean(grouped, modes[1], m) for m in metrics]
    std_a = [_std(grouped, modes[0], m) for m in metrics]
    std_b = [_std(grouped, modes[1], m) for m in metrics]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, vals_a, w, yerr=std_a, capsize=4, label=modes[0])
    ax.bar(x + w / 2, vals_b, w, yerr=std_b, capsize=4, label=modes[1])
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("GB/s")
    ax.set_title("Throughput Compare (Mean ± Std)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot mode compare results from summary + jsonl")
    p.add_argument("--summary", type=str, required=True)
    p.add_argument("--jsonl", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    args = p.parse_args()

    summary_path = Path(args.summary)
    jsonl_path = Path(args.jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = _load_json(summary_path)
    all_rows = _load_jsonl(jsonl_path)

    session = summary.get("compare_session")
    modes = summary.get("modes", ["direct", "server_client"])

    # Use same session as summary to avoid historical rows in append-only jsonl.
    rows = [r for r in all_rows if r.get("compare_session") == session]
    if not rows:
        rows = list(summary.get("runs", []))

    grouped = _group_by_mode(rows, modes)

    _make_ready_breakdown(grouped, modes, out_dir / "01_ready_breakdown.png")
    _make_put_get_breakdown(grouped, modes, out_dir / "02_put_get_breakdown.png")
    _make_latency_compare(grouped, modes, out_dir / "03_latency_compare.png")
    _make_throughput_compare(grouped, modes, out_dir / "04_throughput_compare.png")

    manifest = {
        "summary": str(summary_path),
        "jsonl": str(jsonl_path),
        "compare_session": session,
        "rows_used": len(rows),
        "modes": modes,
        "files": [
            "01_ready_breakdown.png",
            "02_put_get_breakdown.png",
            "03_latency_compare.png",
            "04_throughput_compare.png",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

