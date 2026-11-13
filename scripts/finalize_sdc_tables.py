"""Create paper-facing SDC result and complexity tables.

This script intentionally separates locked test metrics from training/runtime
metrics. Test metrics come from saved-checkpoint evaluation outputs, while
runtime and complexity come from the original Kaggle training JSON files.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "final_tables"

TEST_EVALS = [
    ROOT / "results" / "test_evaluation" / "key_models_test_eval.json",
    ROOT / "results" / "test_evaluation" / "kangru_sdc_no_current_test_eval.json",
    ROOT / "results" / "test_evaluation" / "kangru_3seed_sdc_no_current_test_eval.json",
    ROOT / "results" / "test_evaluation" / "clipped_kangru_3seed_sdc_no_current_test_eval.json",
    ROOT / "results" / "test_evaluation" / "fused_clipped_kangru_3seed_sdc_no_current_test_eval.json",
    ROOT / "results" / "test_evaluation" / "hybrid_clipped_kangru_3seed_sdc_no_current_test_eval.json",
    ROOT / "results" / "test_evaluation" / "static_mlp_h200_sdc_no_current_test_eval.json",
]

TRAINING_JSONS = [
    ROOT
    / "results"
    / "sdc_static_mlp_h200_no_current_100_seed0_1_2.json",
    ROOT
    / "results"
    / "kaggle_imported"
    / "no_current_baselines_100epoch_3seed"
    / "kaggle_no_current_baselines_100epoch_3seed.json",
    ROOT
    / "results"
    / "kaggle_imported"
    / "feature_session_a"
    / "kaggle_feature_no_current_clipped_h32_100epoch_3seed.json",
    ROOT
    / "results"
    / "kaggle_imported"
    / "main_h32_100epoch_3seed"
    / "kaggle_sequence_h32_100epoch_3seed.json",
    ROOT
    / "results"
    / "kaggle_imported"
    / "clipped_h32_100epoch_3seed"
    / "kaggle_clipped_h32_100epoch_3seed.json",
    ROOT
    / "results"
    / "kaggle_imported"
    / "bounded_h32_100epoch_3seed"
    / "kaggle_bounded_h32_100epoch_3seed.json",
    ROOT
    / "results"
    / "kaggle_imported"
    / "param_matched_lstm_gru_100epoch_3seed"
    / "kaggle_param_matched_lstm_gru_100epoch_3seed.json",
    ROOT
    / "results"
    / "kaggle_imported"
    / "kangru_sdc_100epoch_seed0"
    / "results"
    / "sdc_kangru_no_current_100_seed0.json",
    ROOT
    / "results"
    / "kaggle_imported"
    / "kangru_sdc_100epoch_seed1_2"
    / "results"
    / "sdc_kangru_no_current_100_seed1_2.json",
    ROOT
    / "results"
    / "kaggle_imported"
    / "clipped_kangru_sdc_100epoch_seed1_2"
    / "results"
    / "sdc_clipped_kangru_no_current_100_seed1_2.json",
    ROOT
    / "results"
    / "kaggle_imported"
    / "fused_clipped_kangru_sdc_100epoch_seed0_1_2"
    / "results"
    / "sdc_fused_clipped_kangru_no_current_100_seed0_1_2.json",
    ROOT
    / "results"
    / "kaggle_imported"
    / "hybrid_clipped_kangru_sdc_100epoch_seed0_1_2"
    / "results"
    / "sdc_hybrid_clipped_kangru_no_current_100_seed0_1_2.json",
]

DISPLAY = {
    "lstm": "LSTM h32",
    "gru": "GRU h32",
    "lstm_h100": "LSTM h100",
    "gru_h116": "GRU h116",
    "static_mlp_h200": "Static MLP h200",
    "kan_lstm_h32": "KAN-LSTM h32",
    "gauge_kan_lstm_h32": "Gauge KAN-LSTM h32",
    "bounded_kan_lstm_h32": "Bounded KAN-LSTM h32",
    "gauge_bounded_kan_lstm_h32": "Gauge Bounded KAN-LSTM h32",
    "clipped_kan_lstm_h32": "Clipped KAN-LSTM h32",
    "gauge_clipped_kan_lstm_h32": "Gauge Clipped KAN-LSTM h32",
    "kan_gru_h32": "KAN-GRU h32",
    "clipped_kan_gru_h32": "Clipped KAN-GRU h32",
    "fused_clipped_kan_gru_h32": "Fused Clipped KAN-GRU h32",
    "hybrid_clipped_kan_gru_h32": "Hybrid Clipped KAN-GRU h32",
}

NO_CURRENT_ORDER = [
    "lstm_h100",
    "gru_h116",
    "static_mlp_h200",
    "kan_lstm_h32",
    "clipped_kan_lstm_h32",
    "kan_gru_h32",
    "clipped_kan_gru_h32",
    "fused_clipped_kan_gru_h32",
    "hybrid_clipped_kan_gru_h32",
]


def fmt_mean_std(mean_value: float, std_value: float, runs: int, digits: int = 4) -> str:
    if runs <= 1:
        return f"{mean_value:.{digits}f}"
    return f"{mean_value:.{digits}f} +/- {std_value:.{digits}f}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_test_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in TEST_EVALS:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("summary", []):
            rows.append(
                {
                    "variant": row["variant"],
                    "model": DISPLAY.get(row["variant"], row["variant"]),
                    "feature_mode": row["feature_mode"],
                    "runs": int(row["runs"]),
                    "params": int(row["params"]),
                    "teacher_r2_mean": float(row["teacher_r2_mean"]),
                    "teacher_r2_std": float(row.get("teacher_r2_std", 0.0)),
                    "teacher_rmse_mean": float(row["teacher_rmse_mean"]),
                    "rollout_r2_mean": float(row["rollout_r2_mean"]),
                    "rollout_r2_std": float(row.get("rollout_r2_std", 0.0)),
                    "rollout_rmse_mean": float(row["rollout_rmse_mean"]),
                    "rollout_final_state_mae_mean": float(row["rollout_final_state_mae_mean"]),
                    "eval_seconds_mean": float(row["eval_seconds_mean"]),
                    "source_json": str(path.relative_to(ROOT)),
                }
            )
    deduped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["variant"], row["feature_mode"])
        existing = deduped.get(key)
        if existing is None or row["runs"] > existing["runs"]:
            deduped[key] = row
    return list(deduped.values())


def load_training_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in TRAINING_JSONS:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("results", []):
            timing = row.get("timing", {})
            complexity = row.get("complexity", {})
            rows.append(
                {
                    "variant": row["variant"],
                    "model": DISPLAY.get(row["variant"], row["variant"]),
                    "feature_mode": row.get("feature_mode", ""),
                    "seed": int(row.get("seed", -1)),
                    "epochs": int(row.get("epochs", 0)),
                    "params": int(complexity.get("total_parameters", 0)),
                    "kan_layers": int(complexity.get("kan_layers", 0) or 0),
                    "kan_edges": int(complexity.get("kan_edges", 0) or 0),
                    "spline_coefficients": int(complexity.get("spline_coefficients", 0) or 0),
                    "seconds_per_epoch": float(timing.get("seconds_per_epoch", 0.0)),
                    "seconds_per_step": float(timing.get("seconds_per_step", 0.0)),
                    "wall_clock_seconds": float(timing.get("wall_clock_seconds", 0.0)),
                    "peak_cuda_memory_mb": float(timing.get("peak_cuda_memory_mb", 0.0)),
                    "projection_overhead_pct": 100.0 * float(timing.get("projection_wall_clock_fraction", 0.0)),
                    "source_json": str(path.relative_to(ROOT)),
                }
            )
    return rows


def summarize_training(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["variant"], row["feature_mode"])].append(row)

    summary: list[dict[str, Any]] = []
    for (variant, feature_mode), items in grouped.items():
        seconds = [r["seconds_per_epoch"] for r in items]
        peak = [r["peak_cuda_memory_mb"] for r in items]
        wall = [r["wall_clock_seconds"] for r in items]
        summary.append(
            {
                "variant": variant,
                "model": DISPLAY.get(variant, variant),
                "feature_mode": feature_mode,
                "runs": len(items),
                "epochs": items[0]["epochs"],
                "params": items[0]["params"],
                "kan_layers": items[0]["kan_layers"],
                "kan_edges": items[0]["kan_edges"],
                "spline_coefficients": items[0]["spline_coefficients"],
                "seconds_per_epoch_mean": mean(seconds),
                "seconds_per_epoch_std": pstdev(seconds),
                "wall_clock_hours_mean": mean(wall) / 3600.0,
                "peak_cuda_memory_mb_max": max(peak),
                "projection_overhead_pct_mean": mean([r["projection_overhead_pct"] for r in items]),
            }
        )
    return summary


def sort_no_current(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {name: idx for idx, name in enumerate(NO_CURRENT_ORDER)}
    return sorted(
        [r for r in rows if r["feature_mode"] == "no_current" and r["variant"] in order],
        key=lambda r: order[r["variant"]],
    )


def write_markdown(test_rows: list[dict[str, Any]], training_rows: list[dict[str, Any]]) -> None:
    no_current_test = sort_no_current(test_rows)
    no_current_training = {r["variant"]: r for r in sort_no_current(training_rows)}
    full_rows = [r for r in test_rows if r["feature_mode"] == "full"]

    lines = [
        "# Final SDC Result Tables",
        "",
        "Dataset: SDC state-characterisation V-I curves.",
        "",
        "Primary no-current input setting:",
        "`[V_t, V_t_minus_1, dV_t, sweep_direction] -> I_t`.",
        "",
        "For no-current models, teacher-forced and rollout metrics are identical because no predicted `I_t-1` is fed back into the next input.",
        "",
        "## Locked Test Results: No-Current Recurrent Models",
        "",
        "| Model | Runs | Params | Test R2 | Test RMSE | Final-state MAE | Eval sec | Notes |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in no_current_test:
        note = "single seed; screening result" if row["runs"] == 1 else "mean +/- std"
        lines.append(
            f"| {row['model']} | {row['runs']} | {row['params']:,} | "
            f"{fmt_mean_std(row['rollout_r2_mean'], row['rollout_r2_std'], row['runs'])} | "
            f"{row['rollout_rmse_mean']:.4f} | "
            f"{row['rollout_final_state_mae_mean']:.4f} | "
            f"{row['eval_seconds_mean']:.1f} | {note} |"
        )

    lines.extend(
        [
            "",
            "## Locked Test Results: Full-History Reference",
            "",
            "Full-history input includes measured current memory:",
            "`[V_t, V_t_minus_1, I_t_minus_1, dV_t, sweep_direction] -> I_t`.",
            "",
            "| Model | Runs | Params | Teacher R2 | Rollout R2 | Rollout RMSE | Final-state MAE |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in full_rows:
        lines.append(
            f"| {row['model']} | {row['runs']} | {row['params']:,} | "
            f"{fmt_mean_std(row['teacher_r2_mean'], row['teacher_r2_std'], row['runs'])} | "
            f"{fmt_mean_std(row['rollout_r2_mean'], row['rollout_r2_std'], row['runs'])} | "
            f"{row['rollout_rmse_mean']:.4f} | "
            f"{row['rollout_final_state_mae_mean']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Timing And Complexity: No-Current Training Runs",
            "",
            "| Model | Runs | Epochs | Params | KAN layers | Spline coeffs | Sec/epoch | Total hours/run | Peak CUDA MB |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for test_row in no_current_test:
        row = no_current_training.get(test_row["variant"])
        if row is None:
            continue
        lines.append(
            f"| {row['model']} | {row['runs']} | {row['epochs']} | {row['params']:,} | "
            f"{row['kan_layers']} | {row['spline_coefficients']:,} | "
            f"{fmt_mean_std(row['seconds_per_epoch_mean'], row['seconds_per_epoch_std'], row['runs'], digits=2)} | "
            f"{row['wall_clock_hours_mean']:.2f} | "
            f"{row['peak_cuda_memory_mb_max']:.1f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Clipped KAN-GRU h32 is the strongest no-current SDC test result so far and is confirmed over three seeds.",
            "- Fused Clipped KAN-GRU h32 is a useful speed ablation: faster than faithful clipped KAN-GRU, but less stable and less accurate on average.",
            "- Hybrid Clipped KAN-GRU h32 is a compact ablation with far fewer parameters, but its SDC accuracy is weaker.",
            "- GRU h116 and LSTM h100 remain the strongest fast dense baselines.",
            "- KAN-GRU h32 has fewer parameters than KAN-LSTM h32, but is slower because the faithful GRU candidate path requires a second KAN evaluation per time step.",
            "- Timing should be reported alongside accuracy: KAN-gated recurrent cells are parameter-efficient, not wall-clock efficient in the current implementation.",
        ]
    )

    (OUT_DIR / "sdc_final_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    test_rows = load_test_rows()
    training_per_seed = load_training_rows()
    training_summary = summarize_training(training_per_seed)

    test_rows = sorted(test_rows, key=lambda r: (r["feature_mode"], r["variant"]))
    training_summary = sorted(training_summary, key=lambda r: (r["feature_mode"], r["variant"]))

    write_csv(OUT_DIR / "sdc_locked_test_results.csv", test_rows)
    write_csv(OUT_DIR / "sdc_training_complexity_timing.csv", training_summary)
    write_markdown(test_rows, training_summary)
    print(f"Wrote SDC final tables to {OUT_DIR}")


if __name__ == "__main__":
    main()
