"""Run the first fixed-budget Track A matrix on the SDC V-I dataset."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from memristorkan.metrics import regression_metrics
from memristorkan.complexity import model_complexity
from memristorkan.train import ModelConfig, TrainingConfig, build_model, train_model
from memristorkan.rollout import rollout_sdc_state


@dataclass(frozen=True)
class Variant:
    name: str
    model_name: str
    weight_decay: float = 0.0
    amplitude_decay: float = 0.0
    gauge_every_steps: int = 0
    moment_mode: str = "unchanged"
    wb_decay: str = "standard_optimizer"


VARIANTS = {
    "mlp": Variant("mlp", "mlp"),
    "kan": Variant("kan", "kan"),
    "kan_standard_weight_decay": Variant("kan_standard_weight_decay", "kan", weight_decay=1e-4),
    "gauge_kan": Variant("gauge_kan", "gauge_kan", gauge_every_steps=1),
    "gauge_kan_reset_moments": Variant(
        "gauge_kan_reset_moments",
        "gauge_kan",
        gauge_every_steps=1,
        moment_mode="reset",
    ),
    "gauge_kan_standard_weight_decay": Variant(
        "gauge_kan_standard_weight_decay",
        "gauge_kan",
        weight_decay=1e-4,
        gauge_every_steps=1,
    ),
    "gauge_kan_amplitude_decay": Variant(
        "gauge_kan_amplitude_decay",
        "gauge_kan",
        amplitude_decay=1e-4,
        gauge_every_steps=1,
        wb_decay="none",
    ),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_split(cache: Path, split: str) -> TensorDataset:
    data = np.load(cache)
    x = torch.from_numpy(data[f"{split}_x"].astype(np.float32))
    y = torch.from_numpy(data[f"{split}_y"].astype(np.float32))
    return TensorDataset(x, y)


@torch.no_grad()
def evaluate_raw(model: torch.nn.Module, loader: DataLoader, cache: Path, split: str) -> dict[str, float]:
    data = np.load(cache)
    y_mean = torch.from_numpy(data["y_mean"].astype(np.float32))
    y_std = torch.from_numpy(data["y_std"].astype(np.float32))
    preds = []
    targets = []
    model.eval()
    for x, y in loader:
        preds.append(model(x).cpu() * y_std + y_mean)
        targets.append(y.cpu() * y_std + y_mean)
    metrics = regression_metrics(torch.cat(preds, dim=0), torch.cat(targets, dim=0))
    return {f"{split}_raw_{key}": value for key, value in metrics.items()}


def evaluate_rollout(model: torch.nn.Module, cache: Path, split: str) -> dict[str, float]:
    data = np.load(cache)
    required = [
        f"{split}_curve_x_raw",
        f"{split}_curve_y_raw",
        f"{split}_curve_mask",
        "x_mean",
        "x_std",
        "y_mean",
        "y_std",
    ]
    if not all(key in data for key in required):
        return {}
    metrics = rollout_sdc_state(
        model,
        data[f"{split}_curve_x_raw"],
        data[f"{split}_curve_y_raw"],
        data[f"{split}_curve_mask"],
        data["x_mean"],
        data["x_std"],
        data["y_mean"],
        data["y_std"],
    )
    return {f"{split}_{key}": value for key, value in metrics.items()}


def run_variant(args: argparse.Namespace, variant: Variant, seed: int) -> dict[str, object]:
    set_seed(seed)
    cache = Path(args.cache)
    train = load_split(cache, "train")
    eval_dataset = load_split(cache, args.eval_split)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=generator)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size)
    model = build_model(
        ModelConfig(
            name=variant.model_name,
            input_dim=5,
            output_dim=1,
            hidden_dims=(args.hidden_dim,),
            hidden_dim=args.hidden_dim,
            grid_size=args.grid_size,
            spline_order=args.spline_order,
        )
    )
    history = train_model(
        model,
        train_loader,
        TrainingConfig(
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            optimizer=args.optimizer,
            weight_decay=variant.weight_decay,
            amplitude_decay=variant.amplitude_decay,
            gauge_every_steps=variant.gauge_every_steps,
            moment_mode=variant.moment_mode,
            wb_decay=variant.wb_decay,
        ),
        eval_loader=eval_loader,
    )
    raw_metrics = evaluate_raw(model, eval_loader, cache, args.eval_split)
    rollout_metrics = evaluate_rollout(model, cache, args.eval_split) if args.rollout else {}
    checkpoint_path = None
    if args.save_models:
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"{variant.name}_seed{seed}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "variant_config": asdict(variant),
                "seed": seed,
                "input_dim": 5,
                "output_dim": 1,
                "hidden_dim": args.hidden_dim,
                "grid_size": args.grid_size,
                "spline_order": args.spline_order,
                "complexity": model_complexity(model),
            },
            checkpoint_path,
        )
    timing = {
        "wall_clock_seconds": history.wall_clock_seconds,
        "seconds_per_step": history.wall_clock_seconds / max(1, args.steps),
        "projection_wall_clock_seconds": history.projection_wall_clock_seconds,
        "projection_seconds_per_event": (
            history.projection_wall_clock_seconds / len(history.projection_events)
            if history.projection_events
            else 0.0
        ),
        "projection_wall_clock_fraction": (
            history.projection_wall_clock_seconds / history.wall_clock_seconds
            if history.wall_clock_seconds > 0
            else 0.0
        ),
    }
    return {
        "variant": variant.name,
        "seed": seed,
        "variant_config": asdict(variant),
        "complexity": model_complexity(model),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "grid_size": args.grid_size,
        "spline_order": args.spline_order,
        "lr": args.lr,
        "optimizer": args.optimizer,
        "eval_split": args.eval_split,
        "best_train_loss": min(history.losses),
        "final_train_loss": history.losses[-1],
        "eval_metrics_normalized": history.eval_metrics,
        "eval_metrics_raw": raw_metrics,
        "eval_metrics_rollout": rollout_metrics,
        "projection_events": len(history.projection_events),
        "last_projection_event": history.projection_events[-1] if history.projection_events else None,
        "timing": timing,
        "wall_clock_seconds": history.wall_clock_seconds,
    }


def aggregate(results: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for result in results:
        grouped.setdefault(str(result["variant"]), []).append(result)
    summary = {}
    for name, rows in grouped.items():
        rmse = np.asarray([row["eval_metrics_normalized"]["rmse"] for row in rows], dtype=float)
        r2 = np.asarray([row["eval_metrics_normalized"]["r2"] for row in rows], dtype=float)
        row_summary = {
            "runs": len(rows),
            "rmse_mean": float(rmse.mean()),
            "rmse_std": float(rmse.std(ddof=0)),
            "r2_mean": float(r2.mean()),
            "r2_std": float(r2.std(ddof=0)),
        }
        rollout_rows = [row for row in rows if row.get("eval_metrics_rollout")]
        if rollout_rows:
            rollout_rmse = np.asarray(
                [row["eval_metrics_rollout"][f"{row['eval_split']}_rollout_rmse"] for row in rollout_rows],
                dtype=float,
            )
            rollout_r2 = np.asarray(
                [row["eval_metrics_rollout"][f"{row['eval_split']}_rollout_r2"] for row in rollout_rows],
                dtype=float,
            )
            rollout_final_mae = np.asarray(
                [
                    row["eval_metrics_rollout"][f"{row['eval_split']}_rollout_final_state_mae"]
                    for row in rollout_rows
                ],
                dtype=float,
            )
            row_summary.update(
                {
                    "rollout_rmse_mean": float(rollout_rmse.mean()),
                    "rollout_rmse_std": float(rollout_rmse.std(ddof=0)),
                    "rollout_r2_mean": float(rollout_r2.mean()),
                    "rollout_r2_std": float(rollout_r2.std(ddof=0)),
                    "rollout_final_state_mae_mean": float(rollout_final_mae.mean()),
                    "rollout_final_state_mae_std": float(rollout_final_mae.std(ddof=0)),
                }
            )
        summary[name] = row_summary
    return summary


def write_payload(out: Path, results: list[dict[str, object]]) -> None:
    payload = {
        "note": "Starter CPU Track A run. Validation split is used by default; keep test split for final locked experiments.",
        "results": results,
        "summary": aggregate(results),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="data/processed/sdc_state_splits.npz")
    parser.add_argument("--out", default="results/sdc_track_a_cpu_starter.json")
    parser.add_argument("--eval-split", default="val", choices=["val", "test"])
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()), choices=list(VARIANTS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--spline-order", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--rollout", action="store_true")
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--checkpoint-dir", default="checkpoints/track_a")
    args = parser.parse_args()

    out = Path(args.out)
    results = []
    for seed in args.seeds:
        for name in args.variants:
            print(f"Running {name} seed={seed}", flush=True)
            results.append(run_variant(args, VARIANTS[name], seed))
            write_payload(out, results)
    payload = json.loads(out.read_text(encoding="utf-8"))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
