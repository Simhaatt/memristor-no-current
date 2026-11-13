"""Summarize and plot imported Kaggle sequence results.

The script accepts one or more ``run_sdc_sequence.py`` JSON outputs and writes
paper-friendly CSV/Markdown summaries plus compact diagnostic figures.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import matplotlib.pyplot as plt


DEFAULT_ORDER = [
    "lstm",
    "lstm_h100",
    "gru",
    "gru_h116",
    "kan_lstm_h32",
    "gauge_kan_lstm_h32",
    "bounded_kan_lstm_h32",
    "gauge_bounded_kan_lstm_h32",
    "clipped_kan_lstm_h32",
    "gauge_clipped_kan_lstm_h32",
]

DISPLAY_NAMES = {
    "lstm": "LSTM",
    "gru": "GRU",
    "kan_lstm_h32": "KAN-LSTM",
    "gauge_kan_lstm_h32": "Gauge KAN-LSTM",
    "bounded_kan_lstm_h32": "Bounded KAN-LSTM",
    "gauge_bounded_kan_lstm_h32": "Gauge Bounded",
    "clipped_kan_lstm_h32": "Clipped KAN-LSTM",
    "gauge_clipped_kan_lstm_h32": "Gauge Clipped",
}


def nested_get(data: dict[str, Any], keys: tuple[str, ...], default: float = float("nan")) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def load_results(paths: list[Path]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text())
        for row in payload.get("results", []):
            row = dict(row)
            row["_source_json"] = str(path)
            results.append(row)
    if not results:
        raise ValueError("No result rows found in input JSON files.")
    return results


def summarize(results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_seed: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in results:
        flat = {
            "variant": row["variant"],
            "seed": row["seed"],
            "epochs": row.get("epochs"),
            "params": nested_get(row, ("complexity", "total_parameters"), None),
            "teacher_r2": nested_get(row, ("final_teacher_forced_val_metrics", "r2")),
            "teacher_rmse": nested_get(row, ("final_teacher_forced_val_metrics", "rmse")),
            "teacher_mae": nested_get(row, ("final_teacher_forced_val_metrics", "mae")),
            "rollout_r2": nested_get(row, ("final_rollout_val_metrics", "rollout_r2")),
            "rollout_rmse": nested_get(row, ("final_rollout_val_metrics", "rollout_rmse")),
            "rollout_mae": nested_get(row, ("final_rollout_val_metrics", "rollout_mae")),
            "rollout_final_state_mae": nested_get(row, ("final_rollout_val_metrics", "rollout_final_state_mae")),
            "seconds_per_epoch": nested_get(row, ("timing", "seconds_per_epoch")),
            "seconds_per_step": nested_get(row, ("timing", "seconds_per_step")),
            "wall_clock_seconds": nested_get(row, ("timing", "wall_clock_seconds")),
            "peak_cuda_memory_mb": nested_get(row, ("timing", "peak_cuda_memory_mb")),
            "projection_overhead_pct": 100.0
            * nested_get(row, ("timing", "projection_wall_clock_fraction"), 0.0),
            "checkpoint_path": row.get("checkpoint_path", ""),
            "source_json": row.get("_source_json", ""),
        }
        per_seed.append(flat)
        grouped[row["variant"]].append(flat)

    order = {name: idx for idx, name in enumerate(DEFAULT_ORDER)}
    per_seed.sort(key=lambda r: (order.get(r["variant"], 999), r["seed"]))

    summary: list[dict[str, Any]] = []
    for variant, rows in grouped.items():
        def values(key: str) -> list[float]:
            return [float(r[key]) for r in rows]

        summary.append(
            {
                "variant": variant,
                "runs": len(rows),
                "epochs": rows[0]["epochs"],
                "params": rows[0]["params"],
                "teacher_r2_mean": mean(values("teacher_r2")),
                "teacher_r2_std": pstdev(values("teacher_r2")),
                "teacher_rmse_mean": mean(values("teacher_rmse")),
                "rollout_r2_mean": mean(values("rollout_r2")),
                "rollout_r2_std": pstdev(values("rollout_r2")),
                "rollout_rmse_mean": mean(values("rollout_rmse")),
                "rollout_final_state_mae_mean": mean(values("rollout_final_state_mae")),
                "seconds_per_epoch_mean": mean(values("seconds_per_epoch")),
                "seconds_per_step_mean": mean(values("seconds_per_step")),
                "wall_clock_minutes_mean": mean(values("wall_clock_seconds")) / 60.0,
                "peak_cuda_memory_mb_max": max(values("peak_cuda_memory_mb")),
                "projection_overhead_pct_mean": mean(values("projection_overhead_pct")),
            }
        )

    summary.sort(key=lambda r: order.get(r["variant"], 999))
    return per_seed, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: list[dict[str, Any]]) -> None:
    lines = [
        "# Sequence Results Summary",
        "",
        "| Variant | Runs | Params | Teacher R2 | Rollout R2 | Rollout RMSE | Sec/epoch | Peak MB | Projection overhead |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| `{row['variant']}` | {row['runs']} | {int(row['params']):,} | "
            f"{row['teacher_r2_mean']:.4f} +/- {row['teacher_r2_std']:.4f} | "
            f"{row['rollout_r2_mean']:.4f} +/- {row['rollout_r2_std']:.4f} | "
            f"{row['rollout_rmse_mean']:.4f} | {row['seconds_per_epoch_mean']:.2f} | "
            f"{row['peak_cuda_memory_mb_max']:.1f} | {row['projection_overhead_pct_mean']:.3f}% |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- Teacher-forced metrics are saturated, so rollout is the more discriminating metric.",
            "- Projection overhead should be interpreted as extra training cost for gauge variants.",
            "- Bounded variants appear automatically when their full-run JSON is included.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def plot_rollout(summary: list[dict[str, Any]], out_dir: Path) -> None:
    labels = [DISPLAY_NAMES.get(r["variant"], r["variant"]) for r in summary]
    means = [r["rollout_r2_mean"] for r in summary]
    errors = [r["rollout_r2_std"] for r in summary]
    colors = ["#6b7280" if "gauge" not in r["variant"] else "#2563eb" for r in summary]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = range(len(summary))
    ax.bar(x, means, yerr=errors, capsize=4, color=colors, alpha=0.9)
    ax.axhline(0.0, color="#111827", linewidth=0.8)
    ax.set_ylabel("Rollout R2")
    ax.set_title("Rollout Reliability Across Sequence Models")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "sequence_rollout_r2.png", dpi=220)
    plt.close(fig)


def plot_teacher_vs_rollout(summary: list[dict[str, Any]], out_dir: Path) -> None:
    labels = [DISPLAY_NAMES.get(r["variant"], r["variant"]) for r in summary]
    x = range(len(summary))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar([i - width / 2 for i in x], [r["teacher_r2_mean"] for r in summary], width, label="Teacher-forced R2", color="#10b981")
    ax.bar([i + width / 2 for i in x], [r["rollout_r2_mean"] for r in summary], width, label="Rollout R2", color="#f59e0b")
    ax.set_ylabel("R2")
    ax.set_title("Teacher-Forced Metrics Saturate, Rollout Separates Models")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_ylim(bottom=min(-0.35, min(r["rollout_r2_mean"] for r in summary) - 0.1), top=1.05)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "sequence_teacher_vs_rollout_r2.png", dpi=220)
    plt.close(fig)


def plot_runtime(summary: list[dict[str, Any]], out_dir: Path) -> None:
    labels = [DISPLAY_NAMES.get(r["variant"], r["variant"]) for r in summary]
    x = range(len(summary))

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x, [r["seconds_per_epoch_mean"] for r in summary], color="#4b5563", alpha=0.9)
    ax.set_ylabel("Seconds per epoch")
    ax.set_title("Training Runtime Per Epoch")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "sequence_seconds_per_epoch.png", dpi=220)
    plt.close(fig)


def write_plots(summary: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_rollout(summary, out_dir)
    plot_teacher_vs_rollout(summary, out_dir)
    plot_runtime(summary, out_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/sequence_analysis"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = load_results(args.inputs)
    per_seed, summary = summarize(results)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "sequence_per_seed.csv", per_seed)
    write_csv(args.out_dir / "sequence_summary.csv", summary)
    write_markdown(args.out_dir / "sequence_summary.md", summary)
    (args.out_dir / "sequence_summary.json").write_text(json.dumps({"rows": summary}, indent=2))
    write_plots(summary, args.out_dir / "figures")
    print(f"Wrote sequence analysis to {args.out_dir}")


if __name__ == "__main__":
    main()
