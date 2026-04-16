#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

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


def _load_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(value: str | float | int | None) -> float | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


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


def _iqr_filter(values: List[float], iqr_multiplier: float) -> Tuple[List[float], float, float, float, float]:
    if not values:
        return [], float("nan"), float("nan"), float("nan"), float("nan")
    arr = np.array(values, dtype=float)
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    iqr = q3 - q1
    if iqr <= 0:
        return list(values), q1, q3, float("-inf"), float("inf")
    lower = q1 - iqr_multiplier * iqr
    upper = q3 + iqr_multiplier * iqr
    kept = [v for v in values if lower <= v <= upper]
    return kept, q1, q3, lower, upper


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


def _build_metric_index(rows: List[dict], phase: str) -> Dict[Tuple[float, str, str, str], List[float]]:
    metrics = [
        "get_match_ms",
        "get_launch_ms",
        "get_wait_ms",
        "get_total_ms",
        "put_submit_ms",
        "put_wait_ms",
        "put_total_ms",
    ]
    index: Dict[Tuple[float, str, str, str], List[float]] = {}
    for r in rows:
        if phase and r.get("phase") != phase:
            continue
        cache_ratio = _to_float(r.get("cache_ratio"))
        mode = r.get("mode")
        operation = r.get("operation")
        if cache_ratio is None or not mode or not operation:
            continue
        for metric in metrics:
            v = _to_float(r.get(metric))
            if v is None:
                continue
            key = (cache_ratio, mode, operation, metric)
            index.setdefault(key, []).append(v)
    return index


def _make_distribution_plot(
    metric_index: Dict[Tuple[float, str, str, str], List[float]],
    cache_ratios: List[float],
    modes: List[str],
    out: Path,
) -> None:
    get_metrics = ["get_match_ms", "get_launch_ms", "get_wait_ms"]
    put_metrics = ["put_submit_ms", "put_wait_ms"]

    fig, axes = plt.subplots(len(cache_ratios), len(modes), figsize=(6.5 * len(modes), 4.8 * len(cache_ratios)), squeeze=False)
    for i, cache_ratio in enumerate(cache_ratios):
        for j, mode in enumerate(modes):
            ax = axes[i][j]
            labels: List[str] = []
            data: List[List[float]] = []
            for metric in get_metrics:
                vals = metric_index.get((cache_ratio, mode, "get", metric), [])
                if vals:
                    labels.append(metric.replace("get_", "").replace("_ms", ""))
                    data.append(vals)
            for metric in put_metrics:
                vals = metric_index.get((cache_ratio, mode, "put", metric), [])
                if vals:
                    labels.append(metric.replace("put_", "").replace("_ms", ""))
                    data.append(vals)

            if not data:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue

            box = ax.boxplot(data, patch_artist=True, showfliers=True)
            for idx, patch in enumerate(box["boxes"]):
                if idx < len(get_metrics):
                    patch.set_facecolor("#79a7d3")
                else:
                    patch.set_facecolor("#eea76e")
                patch.set_alpha(0.8)
            for median in box["medians"]:
                median.set_color("black")
                median.set_linewidth(1.2)
            for flier in box["fliers"]:
                flier.set(marker="o", markerfacecolor="#222222", markeredgecolor="#222222", alpha=0.4, markersize=3)

            n_get = len(metric_index.get((cache_ratio, mode, "get", "get_total_ms"), []))
            n_put = len(metric_index.get((cache_ratio, mode, "put", "put_total_ms"), []))
            ax.set_title(f"cache_ratio={cache_ratio}, mode={mode}\nget={n_get}, put={n_put}")
            ax.set_xticks(np.arange(1, len(labels) + 1))
            ax.set_xticklabels(labels, rotation=20, ha="right")
            ax.set_ylabel("Milliseconds")
            ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Distribution by cache_ratio + mode (raw values)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _compute_trimmed_means(
    metric_index: Dict[Tuple[float, str, str, str], List[float]],
    cache_ratios: List[float],
    modes: List[str],
    iqr_multiplier: float,
) -> Tuple[List[dict], Dict[Tuple[float, str, str], float]]:
    metric_specs = [
        ("get", "get_match_ms"),
        ("get", "get_launch_ms"),
        ("get", "get_wait_ms"),
        ("get", "get_total_ms"),
        ("put", "put_submit_ms"),
        ("put", "put_wait_ms"),
        ("put", "put_total_ms"),
    ]
    rows: List[dict] = []
    trimmed_lookup: Dict[Tuple[float, str, str], float] = {}
    for cache_ratio in cache_ratios:
        for mode in modes:
            for operation, metric in metric_specs:
                raw_values = metric_index.get((cache_ratio, mode, operation, metric), [])
                kept, q1, q3, lower, upper = _iqr_filter(raw_values, iqr_multiplier)
                raw_mean = float(np.mean(raw_values)) if raw_values else float("nan")
                trimmed_mean = float(np.mean(kept)) if kept else float("nan")
                outlier_count = max(0, len(raw_values) - len(kept))
                rows.append(
                    {
                        "cache_ratio": cache_ratio,
                        "mode": mode,
                        "operation": operation,
                        "metric": metric,
                        "unit": "ms",
                        "raw_count": len(raw_values),
                        "kept_count": len(kept),
                        "outlier_count": outlier_count,
                        "outlier_ratio": (outlier_count / len(raw_values)) if raw_values else 0.0,
                        "raw_mean": raw_mean,
                        "trimmed_mean": trimmed_mean,
                        "q1": q1,
                        "q3": q3,
                        "lower_bound": lower,
                        "upper_bound": upper,
                    }
                )
                trimmed_lookup[(cache_ratio, mode, metric)] = trimmed_mean
    return rows, trimmed_lookup


def _write_trimmed_table(path: Path, rows: List[dict]) -> None:
    fields = [
        "cache_ratio",
        "mode",
        "operation",
        "metric",
        "unit",
        "raw_count",
        "kept_count",
        "outlier_count",
        "outlier_ratio",
        "raw_mean",
        "trimmed_mean",
        "q1",
        "q3",
        "lower_bound",
        "upper_bound",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _make_get_breakdown_trimmed(
    trimmed_lookup: Dict[Tuple[float, str, str], float], cache_ratios: List[float], modes: List[str], out: Path
) -> None:
    fig, axes = plt.subplots(1, len(cache_ratios), figsize=(6.5 * len(cache_ratios), 5), squeeze=False)
    for idx, cache_ratio in enumerate(cache_ratios):
        ax = axes[0][idx]
        x = np.arange(len(modes))
        w = 0.62
        match = [trimmed_lookup.get((cache_ratio, m, "get_match_ms"), 0.0) for m in modes]
        launch = [trimmed_lookup.get((cache_ratio, m, "get_launch_ms"), 0.0) for m in modes]
        wait = [trimmed_lookup.get((cache_ratio, m, "get_wait_ms"), 0.0) for m in modes]
        total = [trimmed_lookup.get((cache_ratio, m, "get_total_ms"), 0.0) for m in modes]

        ax.bar(x, match, w, label="get_match_ms", color="#5b8ec7")
        ax.bar(x, launch, w, bottom=match, label="get_launch_ms", color="#87b4de")
        ax.bar(x, wait, w, bottom=np.array(match) + np.array(launch), label="get_wait_ms", color="#b9d6ee")
        ax.plot(x, total, "ko--", label="get_total_ms (trimmed mean)")
        ax.set_xticks(x)
        ax.set_xticklabels(modes)
        ax.set_ylabel("Milliseconds")
        ax.set_title(f"Get breakdown (trimmed), cache_ratio={cache_ratio}")
        ax.grid(axis="y", alpha=0.25)
        if idx == 0:
            ax.legend()

    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _make_put_breakdown_trimmed(
    trimmed_lookup: Dict[Tuple[float, str, str], float], cache_ratios: List[float], modes: List[str], out: Path
) -> None:
    fig, axes = plt.subplots(1, len(cache_ratios), figsize=(6.5 * len(cache_ratios), 5), squeeze=False)
    for idx, cache_ratio in enumerate(cache_ratios):
        ax = axes[0][idx]
        x = np.arange(len(modes))
        w = 0.62
        submit = [trimmed_lookup.get((cache_ratio, m, "put_submit_ms"), 0.0) for m in modes]
        wait = [trimmed_lookup.get((cache_ratio, m, "put_wait_ms"), 0.0) for m in modes]
        total = [trimmed_lookup.get((cache_ratio, m, "put_total_ms"), 0.0) for m in modes]

        ax.bar(x, submit, w, label="put_submit_ms", color="#cf9348")
        ax.bar(x, wait, w, bottom=submit, label="put_wait_ms", color="#e8bc83")
        ax.plot(x, total, "ko--", label="put_total_ms (trimmed mean)")
        ax.set_xticks(x)
        ax.set_xticklabels(modes)
        ax.set_ylabel("Milliseconds")
        ax.set_title(f"Put breakdown (trimmed), cache_ratio={cache_ratio}")
        ax.grid(axis="y", alpha=0.25)
        if idx == 0:
            ax.legend()

    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _run_csv_mode(csv_path: Path, out_dir: Path, phase: str, iqr_multiplier: float) -> dict:
    rows = _load_csv(csv_path)
    metric_index = _build_metric_index(rows, phase=phase)

    cache_ratios = sorted(
        {
            cache_ratio
            for (cache_ratio, _mode, _operation, _metric) in metric_index.keys()
            if cache_ratio is not None
        }
    )
    all_modes = sorted({_mode for (_cache_ratio, _mode, _operation, _metric) in metric_index.keys()})
    preferred = ["direct", "server_client"]
    modes = [m for m in preferred if m in all_modes] + [m for m in all_modes if m not in preferred]

    if not cache_ratios or not modes:
        raise ValueError("CSV 数据中没有可用的 cache_ratio/mode 组合，请检查输入。")

    dist_path = out_dir / "01_distribution_by_cache_ratio_mode.png"
    trim_table_path = out_dir / "02_outlier_trimmed_means.csv"
    get_breakdown_path = out_dir / "03_get_breakdown_trimmed.png"
    put_breakdown_path = out_dir / "04_put_breakdown_trimmed.png"

    _make_distribution_plot(metric_index, cache_ratios, modes, dist_path)
    trimmed_rows, trimmed_lookup = _compute_trimmed_means(metric_index, cache_ratios, modes, iqr_multiplier=iqr_multiplier)
    _write_trimmed_table(trim_table_path, trimmed_rows)
    _make_get_breakdown_trimmed(trimmed_lookup, cache_ratios, modes, get_breakdown_path)
    _make_put_breakdown_trimmed(trimmed_lookup, cache_ratios, modes, put_breakdown_path)

    manifest = {
        "csv": str(csv_path),
        "phase_filter": phase,
        "iqr_multiplier": iqr_multiplier,
        "cache_ratios": cache_ratios,
        "modes": modes,
        "files": [
            dist_path.name,
            trim_table_path.name,
            get_breakdown_path.name,
            put_breakdown_path.name,
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description="Plot mode compare results")
    p.add_argument("--summary", type=str, default=None, help="旧模式：summary JSON 路径（需配合 --jsonl）")
    p.add_argument("--jsonl", type=str, default=None, help="旧模式：jsonl 路径（需配合 --summary）")
    p.add_argument("--csv-input", type=str, default=None, help="新模式：聚合 CSV 路径（如 compare_new.csv）")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--phase", type=str, default="measure", help="CSV 模式过滤 phase，默认 measure")
    p.add_argument("--iqr-multiplier", type=float, default=1.5, help="IQR 离群阈值倍数，默认 1.5")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.csv_input:
        manifest = _run_csv_mode(Path(args.csv_input), out_dir, phase=args.phase, iqr_multiplier=args.iqr_multiplier)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    if not args.summary or not args.jsonl:
        raise ValueError("请提供 --csv-input，或同时提供 --summary 和 --jsonl。")

    summary_path = Path(args.summary)
    jsonl_path = Path(args.jsonl)
    summary = _load_json(summary_path)
    all_rows = _load_jsonl(jsonl_path)

    session = summary.get("compare_session")
    modes = summary.get("modes", ["direct", "server_client"])

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

