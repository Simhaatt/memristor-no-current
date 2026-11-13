"""Run first small experiments on the main SDC V-I dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from memristorkan.train import ModelConfig, TrainingConfig, build_model, train_model


def load_split(cache: Path, split: str) -> TensorDataset:
    data = np.load(cache)
    x = torch.from_numpy(data[f"{split}_x"].astype(np.float32))
    y = torch.from_numpy(data[f"{split}_y"].astype(np.float32))
    return TensorDataset(x, y)


def run_one(args: argparse.Namespace, model_name: str) -> dict[str, object]:
    train = load_split(Path(args.cache), "train")
    val = load_split(Path(args.cache), "val")
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=args.batch_size)
    model = build_model(
        ModelConfig(
            name=model_name,
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
            gauge_every_steps=args.gauge_every_steps if model_name.startswith("gauge") else 0,
            moment_mode="reset" if args.reset_adam_moments else "unchanged",
            amplitude_decay=args.amplitude_decay if model_name.startswith("gauge") else 0.0,
            wb_decay=args.wb_decay,
        ),
        eval_loader=val_loader,
    )
    return {
        "model": model_name,
        "steps": args.steps,
        "final_train_loss": history.losses[-1],
        "best_train_loss": min(history.losses),
        "val_metrics_normalized": history.eval_metrics,
        "projection_events": len(history.projection_events),
        "last_projection_event": history.projection_events[-1] if history.projection_events else None,
        "wall_clock_seconds": history.wall_clock_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="data/processed/sdc_state_splits.npz")
    parser.add_argument("--models", nargs="+", default=["kan", "gauge_kan"], choices=["mlp", "kan", "gauge_kan"])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--spline-order", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gauge-every-steps", type=int, default=1)
    parser.add_argument("--reset-adam-moments", action="store_true")
    parser.add_argument("--amplitude-decay", type=float, default=0.0)
    parser.add_argument("--wb-decay", default="standard_optimizer", choices=["standard_optimizer", "none"])
    parser.add_argument("--out", default="results/sdc_smoke_results.json")
    args = parser.parse_args()

    results = [run_one(args, model_name) for model_name in args.models]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
