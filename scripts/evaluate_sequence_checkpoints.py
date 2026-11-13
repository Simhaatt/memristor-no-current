"""Evaluate saved sequence checkpoints on validation or test splits.

This script is intentionally evaluation-only: it rebuilds the model recorded in
each checkpoint, loads weights, and computes teacher-forced and rollout metrics
without doing any optimizer steps.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from statistics import mean, pstdev
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from memristorkan.complexity import model_complexity

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_sdc_sequence import (
    SequenceVariant,
    WindowedCurveDataset,
    build_sequence_model,
    current_lags_for_mode,
    evaluate_autoregressive_rollout,
    evaluate_teacher_forced,
    feature_names_for_mode,
)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def variant_from_checkpoint(payload: dict[str, Any]) -> SequenceVariant:
    variant = payload.get("variant")
    if not isinstance(variant, dict):
        raise ValueError("checkpoint does not contain a variant dictionary")
    allowed = {"name", "model_name", "hidden_dim", "grid_size", "gauge_every_steps", "moment_mode"}
    kwargs = {key: value for key, value in variant.items() if key in allowed}
    return SequenceVariant(**kwargs)


def evaluate_one(
    checkpoint_path: Path,
    *,
    cache: Path,
    split: str,
    window: int,
    stride: int,
    batch_size: int,
    device: torch.device,
    num_workers: int,
    rollout_eval: bool,
    perturbation: str,
    perturbation_seed: int,
) -> dict[str, Any]:
    payload = load_checkpoint(checkpoint_path, device)
    variant = variant_from_checkpoint(payload)
    feature_mode = str(payload.get("feature_mode", "full"))
    dataset = WindowedCurveDataset(
        cache,
        split,
        window=window,
        stride=stride,
        feature_mode=feature_mode,
    )
    eval_dataset: Dataset = PerturbedSequenceDataset(dataset, perturbation, perturbation_seed)
    loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    model = build_sequence_model(variant, input_dim=dataset.feature_dim).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    start = time.perf_counter()
    teacher_metrics = evaluate_teacher_forced(model, loader, device)
    rollout_metrics: dict[str, float] = {}
    if rollout_eval:
        cache_data = np.load(cache)
        rollout_metrics = evaluate_autoregressive_rollout(
            model,
            loader,
            device,
            previous_current_lags=current_lags_for_mode(feature_mode),
            x_mean_prev_current=float(cache_data["x_mean"][0, 2]),
            x_std_prev_current=float(cache_data["x_std"][0, 2]),
            y_mean=float(cache_data["y_mean"][0, 0]),
            y_std=float(cache_data["y_std"][0, 0]),
        )
    elapsed = time.perf_counter() - start

    return {
        "checkpoint_path": str(checkpoint_path),
        "variant": variant.name,
        "variant_config": asdict(variant),
        "seed": payload.get("seed"),
        "feature_mode": feature_mode,
        "feature_names": payload.get("feature_names", feature_names_for_mode(feature_mode)),
        "split": split,
        "perturbation": perturbation,
        "window": window,
        "stride": stride,
        "batch_size": batch_size,
        "eval_windows": len(dataset),
        "model_input_dim": dataset.feature_dim,
        "complexity": payload.get("complexity", model_complexity(model)),
        "teacher_forced_metrics": teacher_metrics,
        "rollout_metrics": rollout_metrics,
        "timing": {
            "eval_wall_clock_seconds": elapsed,
            "seconds_per_window": elapsed / max(1, len(dataset)),
        },
}


class PerturbedSequenceDataset(Dataset):
    """Apply deterministic input-sequence perturbations while keeping targets fixed."""

    def __init__(self, base: Dataset, perturbation: str, seed: int) -> None:
        self.base = base
        self.perturbation = perturbation
        self.seed = seed

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.base[idx]
        if self.perturbation == "none":
            return x, y
        if self.perturbation == "reverse_time":
            return torch.flip(x, dims=[0]), y
        if self.perturbation == "shuffle_time":
            generator = torch.Generator().manual_seed(self.seed + int(idx))
            perm = torch.randperm(x.shape[0], generator=generator)
            return x[perm], y
        raise ValueError(f"unknown perturbation: {self.perturbation}")


def summarize(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in results:
        grouped.setdefault((str(row["variant"]), str(row["feature_mode"]), str(row["perturbation"])), []).append(row)

    summary: list[dict[str, Any]] = []
    for (variant, feature_mode, perturbation), rows in grouped.items():
        def values(metric_group: str, key: str) -> list[float]:
            return [float(row[metric_group][key]) for row in rows]

        rollout = rows[0].get("rollout_metrics", {})
        summary.append(
            {
                "variant": variant,
                "feature_mode": feature_mode,
                "perturbation": perturbation,
                "runs": len(rows),
                "params": rows[0]["complexity"]["total_parameters"],
                "teacher_r2_mean": mean(values("teacher_forced_metrics", "r2")),
                "teacher_r2_std": pstdev(values("teacher_forced_metrics", "r2")),
                "teacher_rmse_mean": mean(values("teacher_forced_metrics", "rmse")),
                "rollout_r2_mean": mean(values("rollout_metrics", "rollout_r2")) if rollout else float("nan"),
                "rollout_r2_std": pstdev(values("rollout_metrics", "rollout_r2")) if rollout else float("nan"),
                "rollout_rmse_mean": mean(values("rollout_metrics", "rollout_rmse")) if rollout else float("nan"),
                "rollout_final_state_mae_mean": mean(values("rollout_metrics", "rollout_final_state_mae"))
                if rollout
                else float("nan"),
                "eval_seconds_mean": mean([float(row["timing"]["eval_wall_clock_seconds"]) for row in rows]),
            }
        )
    summary.sort(key=lambda row: (row["perturbation"], row["feature_mode"], row["variant"]))
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def flatten_per_seed(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in results:
        tf = row["teacher_forced_metrics"]
        ro = row["rollout_metrics"]
        rows.append(
            {
                "variant": row["variant"],
                "feature_mode": row["feature_mode"],
                "perturbation": row["perturbation"],
                "seed": row["seed"],
                "split": row["split"],
                "params": row["complexity"]["total_parameters"],
                "teacher_r2": tf["r2"],
                "teacher_rmse": tf["rmse"],
                "teacher_mae": tf["mae"],
                "rollout_r2": ro.get("rollout_r2", float("nan")),
                "rollout_rmse": ro.get("rollout_rmse", float("nan")),
                "rollout_mae": ro.get("rollout_mae", float("nan")),
                "rollout_final_state_mae": ro.get("rollout_final_state_mae", float("nan")),
                "eval_seconds": row["timing"]["eval_wall_clock_seconds"],
                "checkpoint_path": row["checkpoint_path"],
            }
        )
    return rows


def write_markdown(path: Path, summary: list[dict[str, Any]], split: str) -> None:
    lines = [
        f"# Sequence Checkpoint Evaluation On {split}",
        "",
        "| Variant | Feature Mode | Perturbation | Runs | Params | Teacher R2 | Rollout R2 | Rollout RMSE | Eval sec |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| `{row['variant']}` | `{row['feature_mode']}` | `{row['perturbation']}` | "
            f"{row['runs']} | {int(row['params']):,} | "
            f"{row['teacher_r2_mean']:.4f} +/- {row['teacher_r2_std']:.4f} | "
            f"{row['rollout_r2_mean']:.4f} +/- {row['rollout_r2_std']:.4f} | "
            f"{row['rollout_rmse_mean']:.4f} | {row['eval_seconds_mean']:.1f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("data/processed/sdc_state_splits.npz"))
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-rollout", action="store_true")
    parser.add_argument("--perturbation", default="none", choices=["none", "shuffle_time", "reverse_time"])
    parser.add_argument("--perturbation-seed", type=int, default=1234)
    parser.add_argument("--out", type=Path, default=Path("results/test_evaluation/sequence_checkpoint_eval.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    results = []
    for checkpoint in args.checkpoints:
        print(f"Evaluating {checkpoint}", flush=True)
        results.append(
            evaluate_one(
                checkpoint,
                cache=args.cache,
                split=args.split,
                window=args.window,
                stride=args.stride,
                batch_size=args.batch_size,
                device=device,
                num_workers=args.num_workers,
                rollout_eval=not args.no_rollout,
                perturbation=args.perturbation,
                perturbation_seed=args.perturbation_seed,
            )
        )

    summary = summarize(results)
    payload = {"split": args.split, "results": results, "summary": summary}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(args.out.with_suffix(".per_seed.csv"), flatten_per_seed(results))
    write_csv(args.out.with_suffix(".summary.csv"), summary)
    write_markdown(args.out.with_suffix(".md"), summary, args.split)
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
