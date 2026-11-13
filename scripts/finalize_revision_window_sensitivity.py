"""Finalize 128/256/512 SDC window-sensitivity results after GPU runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_sdc_per_curve_significance import (  # noqa: E402
    evaluate_checkpoint_windowed_curves,
    load_checkpoint,
)
from evaluate_sequence_checkpoints import evaluate_one  # noqa: E402


ORIGINAL_CHECKPOINTS = [
    ROOT / "results/kaggle_imported/kangru_sdc_100epoch_seed0/checkpoints/sequence/clipped_kan_gru_h32_no_current_seed0.pt",
    ROOT / "results/kaggle_imported/clipped_kangru_sdc_100epoch_seed1_2/checkpoints/sequence/clipped_kan_gru_h32_no_current_seed1.pt",
    ROOT / "results/kaggle_imported/clipped_kangru_sdc_100epoch_seed1_2/checkpoints/sequence/clipped_kan_gru_h32_no_current_seed2.pt",
    ROOT / "checkpoints/kaggle_imported/no_current_baselines_100epoch_3seed/sequence/gru_h116_no_current_seed0.pt",
    ROOT / "checkpoints/kaggle_imported/no_current_baselines_100epoch_3seed/sequence/gru_h116_no_current_seed1.pt",
    ROOT / "checkpoints/kaggle_imported/no_current_baselines_100epoch_3seed/sequence/gru_h116_no_current_seed2.pt",
]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def infer_window(path: Path, payload: dict[str, Any]) -> int:
    if "window" in payload:
        return int(payload["window"])
    match = re.search(r"window[_-]?(128|256|512)", str(path), re.I)
    if match:
        return int(match.group(1))
    raise ValueError(f"cannot infer window for {path}")


def collect_checkpoints(import_root: Path, device: torch.device) -> dict[tuple[str, int, int], tuple[Path, dict[str, Any]]]:
    paths = [p for p in ORIGINAL_CHECKPOINTS if p.exists()]
    if import_root.exists():
        paths.extend(import_root.rglob("*.pt"))
    selected: dict[tuple[str, int, int], tuple[Path, dict[str, Any]]] = {}
    for path in paths:
        payload = load_checkpoint(path, device)
        variant = payload.get("variant", {})
        model = str(variant.get("name", ""))
        if model not in {"clipped_kan_gru_h32", "gru_h116"}:
            continue
        seed = int(payload["seed"])
        window = infer_window(path, payload)
        selected[(model, window, seed)] = (path, payload)
    return selected


def collect_training_rows(search_roots: list[Path]) -> dict[tuple[str, int, int], dict[str, Any]]:
    indexed: dict[tuple[str, int, int], dict[str, Any]] = {}
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            for row in payload.get("results", []):
                model = str(row.get("variant", ""))
                if model not in {"clipped_kan_gru_h32", "gru_h116"}:
                    continue
                window = int(row.get("window", 256))
                seed = int(row.get("seed", -1))
                epochs = row.get("epoch_summaries", [])
                if not epochs:
                    continue
                best_r2 = max(epochs, key=lambda e: float(e["val_metrics"]["r2"]))
                best_rmse = min(epochs, key=lambda e: float(e["val_metrics"]["rmse"]))
                key = (model, window, seed)
                candidate = {
                    "training_json": str(path),
                    "epochs": int(row.get("epochs", len(epochs))),
                    "train_windows": int(row.get("train_windows", 0)),
                    "validation_windows": int(row.get("eval_windows", 0)),
                    "batch_size": int(row.get("batch_size", 0)),
                    "batches_per_epoch": int(np.ceil(int(row.get("train_windows", 0)) / int(row.get("batch_size", 1)))),
                    "train_stride": int(row.get("stride", window)),
                    "eval_stride": int(row.get("eval_stride", 2 * window)),
                    "total_optimizer_steps": int(row.get("global_steps", 0)),
                    "seconds_per_epoch": float(row["timing"]["seconds_per_epoch"]),
                    "total_training_seconds": float(row["timing"]["wall_clock_seconds"]),
                    "best_validation_r2_epoch": int(best_r2["epoch"]),
                    "best_validation_r2": float(best_r2["val_metrics"]["r2"]),
                    "best_validation_rmse_epoch": int(best_rmse["epoch"]),
                    "best_validation_rmse": float(best_rmse["val_metrics"]["rmse"]),
                    "final_validation_r2": float(epochs[-1]["val_metrics"]["r2"]),
                    "final_validation_rmse": float(epochs[-1]["val_metrics"]["rmse"]),
                }
                previous = indexed.get(key)
                if previous is None or candidate["epochs"] > previous["epochs"]:
                    indexed[key] = candidate
    return indexed


def bootstrap_ci(values: np.ndarray, seed: int = 20260917, repeats: int = 10000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(repeats, dtype=np.float64)
    for idx in range(repeats):
        means[idx] = rng.choice(values, size=len(values), replace=True).mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def save_plot(summary: list[dict[str, Any]], out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6), constrained_layout=True)
    styles = {
        "clipped_kan_gru_h32": ("#0072b2", "o", "Bounded KAN-GRU h32"),
        "gru_h116": ("#d55e00", "s", "Dense GRU h116"),
    }
    for model, (color, marker, label) in styles.items():
        rows = sorted((r for r in summary if r["model"] == model), key=lambda r: r["window_length"])
        x = [r["window_length"] for r in rows]
        axes[0].errorbar(x, [r["test_r2_mean"] for r in rows], yerr=[r["test_r2_std"] for r in rows], color=color, marker=marker, capsize=3, label=label)
        axes[1].errorbar(x, [r["test_rmse_mean"] for r in rows], yerr=[r["test_rmse_std"] for r in rows], color=color, marker=marker, capsize=3, label=label)
    axes[0].set(xlabel="Window length (samples)", ylabel=r"Held-out test $R^2$", xticks=[128, 256, 512])
    axes[1].set(xlabel="Window length (samples)", ylabel="Standardized target RMSE", xticks=[128, 256, 512])
    axes[0].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    paths = []
    for suffix, kwargs in (("png", {"dpi": 350}), ("svg", {})):
        path = out_dir / f"figure_02_window_length_sensitivity.{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paths.append(str(path))
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--import-root", type=Path, default=Path("revision_experiments/window_imports"))
    parser.add_argument("--cache", type=Path, default=Path("data/processed/sdc_state_splits.npz"))
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/sdc_state_metadata.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("revision_experiments"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    checkpoints = collect_checkpoints(args.import_root, device)
    required = {(m, w, s) for m in ("clipped_kan_gru_h32", "gru_h116") for w in (128, 256, 512) for s in (0, 1, 2)}
    missing = sorted(required - set(checkpoints))
    if missing:
        raise RuntimeError(f"missing required checkpoints: {missing}")

    training = collect_training_rows([
        args.import_root,
        ROOT / "results/kaggle_imported/kangru_sdc_100epoch_seed0",
        ROOT / "results/kaggle_imported/clipped_kangru_sdc_100epoch_seed1_2",
        ROOT / "results/kaggle_imported/no_current_baselines_100epoch_3seed",
    ])
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    test_ids = [int(value) for value in metadata["split_curve_indices"]["test"]]
    per_trajectory: list[dict[str, Any]] = []
    pooled: list[dict[str, Any]] = []
    for key in sorted(required):
        model, window, seed = key
        checkpoint, payload = checkpoints[key]
        stride = 2 * window
        curve_rows = evaluate_checkpoint_windowed_curves(
            checkpoint, payload, cache=args.cache, split="test", window=window,
            stride=stride, batch_size=64, device=device, num_workers=0,
        )
        for row in curve_rows:
            per_trajectory.append({
                "window_length": window,
                "model": model,
                "seed": seed,
                "local_test_index": row["curve_index"],
                "trajectory_id": test_ids[int(row["curve_index"])],
                "number_of_evaluated_samples_including_last_window_overlap": row["length"],
                "r2": row["r2"],
                "rmse": row["rmse"],
                "mae": row["mae"],
                "checkpoint_path": str(checkpoint),
            })
        evaluation = evaluate_one(
            checkpoint, cache=args.cache, split="test", window=window, stride=stride,
            batch_size=64, device=device, num_workers=0, rollout_eval=False,
            perturbation="none", perturbation_seed=1234,
        )
        metrics = evaluation["teacher_forced_metrics"]
        meta = training.get(key, {})
        pooled.append({
            "window_length": window,
            "model": model,
            "seed": seed,
            "test_r2": metrics["r2"],
            "test_rmse": metrics["rmse"],
            "test_mae": metrics["mae"],
            "test_windows": evaluation["eval_windows"],
            **meta,
        })

    summary: list[dict[str, Any]] = []
    for model in ("clipped_kan_gru_h32", "gru_h116"):
        for window in (128, 256, 512):
            rows = [r for r in pooled if r["model"] == model and r["window_length"] == window]
            r2 = [float(r["test_r2"]) for r in rows]
            rmse = [float(r["test_rmse"]) for r in rows]
            mae = [float(r["test_mae"]) for r in rows]
            seconds = [float(r["seconds_per_epoch"]) for r in rows if "seconds_per_epoch" in r]
            summary.append({
                "window_length": window,
                "model": model,
                "runs": len(rows),
                "test_r2_mean": mean(r2),
                "test_r2_std": pstdev(r2),
                "test_rmse_mean": mean(rmse),
                "test_rmse_std": pstdev(rmse),
                "test_mae_mean": mean(mae),
                "test_mae_std": pstdev(mae),
                "seconds_per_epoch_mean": mean(seconds) if seconds else float("nan"),
                "total_training_seconds_mean": mean([r["total_training_seconds"] for r in rows if "total_training_seconds" in r]),
                "best_validation_r2_epoch_mean": mean([r["best_validation_r2_epoch"] for r in rows if "best_validation_r2_epoch" in r]),
                "best_validation_r2_mean": mean([r["best_validation_r2"] for r in rows if "best_validation_r2" in r]),
                "train_windows": rows[0].get("train_windows", ""),
                "validation_windows": rows[0].get("validation_windows", ""),
                "test_windows": rows[0]["test_windows"],
                "batches_per_epoch": rows[0].get("batches_per_epoch", ""),
                "epochs": rows[0].get("epochs", ""),
                "batch_size": rows[0].get("batch_size", ""),
                "samples_per_batch": int(rows[0].get("batch_size", 0)) * window,
                "train_stride": rows[0].get("train_stride", ""),
                "eval_stride": rows[0].get("eval_stride", ""),
                "total_optimizer_steps": rows[0].get("total_optimizer_steps", ""),
            })

    paired: list[dict[str, Any]] = []
    for window in (128, 256, 512):
        curve_diffs = []
        for curve in range(9):
            kan = [r["r2"] for r in per_trajectory if r["window_length"] == window and r["model"] == "clipped_kan_gru_h32" and r["local_test_index"] == curve]
            gru = [r["r2"] for r in per_trajectory if r["window_length"] == window and r["model"] == "gru_h116" and r["local_test_index"] == curve]
            curve_diffs.append(float(mean(kan) - mean(gru)))
        values = np.asarray(curve_diffs)
        lo, hi = bootstrap_ci(values)
        paired.append({
            "window_length": window,
            "mean_paired_delta_r2": float(values.mean()),
            "median_paired_delta_r2": float(np.median(values)),
            "bootstrap_95_ci_low": lo,
            "bootstrap_95_ci_high": hi,
            "test_trajectories": 9,
            "kan_wins": int((values > 0).sum()),
            "gru_wins": int((values < 0).sum()),
            "ties": int((values == 0).sum()),
            "bootstrap_seed": 20260917,
            "bootstrap_repeats": 10000,
        })

    write_csv(args.out_dir / "window_sensitivity_per_trajectory.csv", per_trajectory)
    write_csv(args.out_dir / "window_sensitivity_per_seed.csv", pooled)
    write_csv(args.out_dir / "window_sensitivity_summary.csv", summary)
    write_csv(args.out_dir / "window_sensitivity_paired.csv", paired)
    figure_paths = save_plot(summary, args.out_dir / "figures")
    payload = {"summary": summary, "paired": paired, "figure_paths": figure_paths}
    (args.out_dir / "window_sensitivity_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
