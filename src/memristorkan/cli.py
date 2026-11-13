"""Command-line entrypoints for local smoke runs and data inspection."""

from __future__ import annotations

import argparse
import json

from torch.utils.data import DataLoader, random_split

from memristorkan.data import (
    SyntheticDriftConfig,
    generate_synthetic_drift,
    load_sdc_retention_sequences,
    load_sdc_state_table,
    make_point_history_dataset,
    make_sequence_dataset,
)
from memristorkan.train import ModelConfig, TrainingConfig, build_model, train_model


def _smoke(args: argparse.Namespace) -> None:
    series = generate_synthetic_drift(
        SyntheticDriftConfig(num_series=args.num_series, series_length=args.series_length, seed=args.seed)
    )
    if args.model in {"lstm", "gru", "kan_lstm", "gauge_kan_lstm"}:
        dataset = make_sequence_dataset(series)
        input_dim = 3
    else:
        dataset = make_point_history_dataset(series)
        input_dim = 2

    train_size = max(1, int(0.8 * len(dataset)))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size)

    model_config = ModelConfig(
        name=args.model,
        input_dim=input_dim,
        hidden_dims=(args.hidden_dim,),
        hidden_dim=args.hidden_dim,
        grid_size=args.grid_size,
        spline_order=args.spline_order,
    )
    train_config = TrainingConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        gauge_every_steps=1 if args.model in {"gauge_kan", "gauge_kan_lstm"} else 0,
        moment_mode="reset" if args.reset_adam_moments else "unchanged",
        amplitude_decay=args.amplitude_decay,
    )
    model = build_model(model_config)
    history = train_model(model, train_loader, train_config, eval_loader=eval_loader)
    print(
        json.dumps(
            {
                "model": args.model,
                "final_loss": history.losses[-1],
                "eval": history.eval_metrics,
                "projection_events": len(history.projection_events),
                "wall_clock_seconds": history.wall_clock_seconds,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _summarize_sdc(args: argparse.Namespace) -> None:
    if args.kind == "state":
        _, summary = load_sdc_state_table(args.root)
    else:
        _, summary = load_sdc_retention_sequences(args.root, crop_first_half=True)
    print(json.dumps(summary.__dict__, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="memristorkan")
    sub = parser.add_subparsers(dest="command", required=True)

    smoke = sub.add_parser("smoke", help="run a synthetic-drift smoke experiment")
    smoke.add_argument("--model", default="gauge_kan", choices=["mlp", "kan", "gauge_kan", "lstm", "gru", "kan_lstm", "gauge_kan_lstm"])
    smoke.add_argument("--steps", type=int, default=20)
    smoke.add_argument("--batch-size", type=int, default=128)
    smoke.add_argument("--hidden-dim", type=int, default=16)
    smoke.add_argument("--grid-size", type=int, default=8)
    smoke.add_argument("--spline-order", type=int, default=3)
    smoke.add_argument("--lr", type=float, default=1e-3)
    smoke.add_argument("--amplitude-decay", type=float, default=0.0)
    smoke.add_argument("--reset-adam-moments", action="store_true")
    smoke.add_argument("--num-series", type=int, default=32)
    smoke.add_argument("--series-length", type=int, default=64)
    smoke.add_argument("--seed", type=int, default=0)
    smoke.set_defaults(func=_smoke)

    summarize = sub.add_parser("summarize-sdc", help="summarize local SDC data")
    summarize.add_argument("root")
    summarize.add_argument("--kind", choices=["state", "retention"], default="state")
    summarize.set_defaults(func=_summarize_sdc)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
