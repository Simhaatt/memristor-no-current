"""Run current recurrent models on the public albertlin11/rram_lstm data."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from memristorkan.complexity import model_complexity
from memristorkan.metrics import regression_metrics

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_sdc_sequence import VARIANTS, build_sequence_model


TASK_FILES = {
    "random_walk": {
        "train": ["CSV_data/ds_rram_test_rnseed1.csv"],
        "val": ["CSV_data/ds_rram_test_rnseed2.csv"],
        "test": ["CSV_data/ds_rram_test_rnseed3.csv", "CSV_data/ds_rram_test_rnseed4.csv", "CSV_data/ds_rram_test_rnseed5.csv"],
        "seq_len": 5,
        "input_dim": 2,
        "target_col": 10,
        "notes": "RRAM random-walk compact-model task; rows are 5 voltage/current history pairs plus next current.",
    },
    "sine": {
        "train": ["CSV_data/ds_rram_sine0.csv"],
        "val": ["CSV_data/ds_rram_sine-1.csv"],
        "test": ["CSV_data/ds_rram_sine1.csv"],
        "seq_len": 5,
        "input_dim": 2,
        "target_col": 10,
        "notes": "RRAM sinusoidal compact-model task; phase split.",
    },
    "anneal": {
        "train": [
            "CSV_data/Anneal/ds0_rram_rndseed1.csv",
            "CSV_data/Anneal/ds300_rram_rndseed1.csv",
            "CSV_data/Anneal/ds400_rram_rndseed1.csv",
            "CSV_data/Anneal/ds500_rram_rndseed1.csv",
        ],
        "val": [
            "CSV_data/Anneal/ds0_rram_rndseed2.csv",
            "CSV_data/Anneal/ds300_rram_rndseed2.csv",
            "CSV_data/Anneal/ds400_rram_rndseed2.csv",
            "CSV_data/Anneal/ds500_rram_rndseed2.csv",
        ],
        "test": [
            "CSV_data/Anneal/ds0_rram_rndseed3.csv",
            "CSV_data/Anneal/ds300_rram_rndseed3.csv",
            "CSV_data/Anneal/ds400_rram_rndseed3.csv",
            "CSV_data/Anneal/ds500_rram_rndseed3.csv",
        ],
        "seq_len": 5,
        "input_dim": 3,
        "target_col": 15,
        "notes": "Process-aware RRAM annealing task; each timestep has temperature, voltage, current.",
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_from_arg(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def read_rows(root: Path, files: list[str], seq_len: int, input_dim: int, target_col: int) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for rel in files:
        path = root / rel
        # Match the original repo scripts: first CSV row is treated as a header.
        df = pd.read_csv(path)
        values = df.to_numpy(dtype=np.float32)
        x = values[:, : seq_len * input_dim].reshape(-1, seq_len, input_dim)
        y = values[:, target_col : target_col + 1]
        xs.append(x)
        ys.append(y)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def normalize(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> dict[str, np.ndarray]:
    x_mean = train_x.reshape(-1, train_x.shape[-1]).mean(axis=0, keepdims=True).astype(np.float32)
    x_std = train_x.reshape(-1, train_x.shape[-1]).std(axis=0, keepdims=True).clip(1e-8).astype(np.float32)
    y_mean = train_y.mean(axis=0, keepdims=True).astype(np.float32)
    y_std = train_y.std(axis=0, keepdims=True).clip(1e-8).astype(np.float32)
    return {
        "train_x": ((train_x - x_mean) / x_std).astype(np.float32),
        "train_y": ((train_y - y_mean) / y_std).astype(np.float32),
        "val_x": ((val_x - x_mean) / x_std).astype(np.float32),
        "val_y": ((val_y - y_mean) / y_std).astype(np.float32),
        "test_x": ((test_x - x_mean) / x_std).astype(np.float32),
        "test_y": ((test_y - y_mean) / y_std).astype(np.float32),
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, *, shuffle: bool, seed: int) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator if shuffle else None)


def predict_last(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    y = model(x)
    if y.ndim == 3:
        return y[:, -1, :]
    return y


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    y_mean: float,
    y_std: float,
) -> dict[str, float]:
    model.eval()
    preds = []
    targets = []
    for x, y in loader:
        x = x.to(device)
        preds.append(predict_last(model, x).cpu())
        targets.append(y.cpu())
    pred = torch.cat(preds, dim=0)
    target = torch.cat(targets, dim=0)
    metrics = regression_metrics(pred, target)
    pred_raw = pred * y_std + y_mean
    target_raw = target * y_std + y_mean
    raw_error = pred_raw.reshape(-1) - target_raw.reshape(-1)
    denom = (target_raw.reshape(-1) - target_raw.reshape(-1).mean()).pow(2).sum().clamp_min(1e-20)
    metrics.update(
        {
            "raw_rmse": float(torch.sqrt(raw_error.pow(2).mean()).item()),
            "raw_mae": float(raw_error.abs().mean().item()),
            "raw_r2": float((1.0 - raw_error.pow(2).sum() / denom).item()),
        }
    )
    return metrics


def run_one(args: argparse.Namespace, task: dict[str, Any], arrays: dict[str, np.ndarray], variant_name: str, seed: int) -> dict[str, Any]:
    set_seed(seed)
    device = device_from_arg(args.device)
    variant = VARIANTS[variant_name]
    train_loader = make_loader(arrays["train_x"], arrays["train_y"], args.batch_size, shuffle=True, seed=seed)
    val_loader = make_loader(arrays["val_x"], arrays["val_y"], args.batch_size, shuffle=False, seed=seed)
    test_loader = make_loader(arrays["test_x"], arrays["test_y"], args.batch_size, shuffle=False, seed=seed)
    model = build_sequence_model(variant, input_dim=int(task["input_dim"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    epoch_summaries = []
    start_all = time.perf_counter()
    global_steps = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        epoch_start = time.perf_counter()
        for x, y in train_loader:
            global_steps += 1
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = predict_last(model, x)
            loss = torch.mean((pred - y) ** 2)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            y_mean=float(arrays["y_mean"][0, 0]),
            y_std=float(arrays["y_std"][0, 0]),
        )
        epoch_summaries.append(
            {
                "epoch": epoch,
                "train_loss_mean": float(np.mean(losses)),
                "train_loss_last": float(losses[-1]),
                "epoch_seconds": time.perf_counter() - epoch_start,
                "val_metrics": val_metrics,
            }
        )

    test_metrics = evaluate(
        model,
        test_loader,
        device,
        y_mean=float(arrays["y_mean"][0, 0]),
        y_std=float(arrays["y_std"][0, 0]),
    )
    checkpoint_path = None
    if args.save_models:
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"{variant.name}_{args.task}_seed{seed}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "variant": asdict(variant),
                "seed": seed,
                "task": args.task,
                "normalization": {
                    "x_mean": arrays["x_mean"].tolist(),
                    "x_std": arrays["x_std"].tolist(),
                    "y_mean": arrays["y_mean"].tolist(),
                    "y_std": arrays["y_std"].tolist(),
                },
                "complexity": model_complexity(model),
            },
            checkpoint_path,
        )
    total_seconds = time.perf_counter() - start_all
    return {
        "variant": variant.name,
        "seed": seed,
        "task": args.task,
        "task_notes": task["notes"],
        "input_dim": int(task["input_dim"]),
        "seq_len": int(task["seq_len"]),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "train_rows": int(arrays["train_x"].shape[0]),
        "val_rows": int(arrays["val_x"].shape[0]),
        "test_rows": int(arrays["test_x"].shape[0]),
        "global_steps": global_steps,
        "complexity": model_complexity(model),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "epoch_summaries": epoch_summaries,
        "final_val_metrics": epoch_summaries[-1]["val_metrics"],
        "test_metrics": test_metrics,
        "timing": {
            "wall_clock_seconds": total_seconds,
            "seconds_per_epoch": total_seconds / max(1, args.epochs),
            "seconds_per_step": total_seconds / max(1, global_steps),
        },
    }


def summarize(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        grouped.setdefault(row["variant"], []).append(row)
    summary = []
    for variant, rows in grouped.items():
        def vals(section: str, key: str) -> list[float]:
            return [float(row[section][key]) for row in rows]

        summary.append(
            {
                "variant": variant,
                "runs": len(rows),
                "params": rows[0]["complexity"]["total_parameters"],
                "val_r2_mean": mean(vals("final_val_metrics", "r2")),
                "val_r2_std": pstdev(vals("final_val_metrics", "r2")),
                "val_raw_rmse_mean": mean(vals("final_val_metrics", "raw_rmse")),
                "test_r2_mean": mean(vals("test_metrics", "r2")),
                "test_r2_std": pstdev(vals("test_metrics", "r2")),
                "test_raw_rmse_mean": mean(vals("test_metrics", "raw_rmse")),
                "seconds_per_epoch_mean": mean([float(row["timing"]["seconds_per_epoch"]) for row in rows]),
            }
        )
    summary.sort(key=lambda row: row["test_raw_rmse_mean"])
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# RRAM LSTM {payload['task']} Results",
        "",
        payload["task_notes"],
        "",
        "| Variant | Runs | Params | Val R2 | Val Raw RMSE (A) | Test R2 | Test Raw RMSE (A) | Sec/epoch |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]:
        lines.append(
            f"| `{row['variant']}` | {row['runs']} | {int(row['params']):,} | "
            f"{row['val_r2_mean']:.4f} +/- {row['val_r2_std']:.4f} | "
            f"{row['val_raw_rmse_mean']:.6g} | "
            f"{row['test_r2_mean']:.4f} +/- {row['test_r2_std']:.4f} | "
            f"{row['test_raw_rmse_mean']:.6g} | {row['seconds_per_epoch_mean']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Claim boundary: this is an external public-code/data comparison aligned with an existing LSTM compact-modelling paper.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/rram_lstm"))
    parser.add_argument("--task", default="random_walk", choices=list(TASK_FILES))
    parser.add_argument("--out", type=Path, default=Path("results/rram_lstm_sequence.json"))
    parser.add_argument("--variants", nargs="+", default=["lstm_h100", "gru_h116", "kan_lstm_h32", "clipped_kan_lstm_h32"], choices=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--checkpoint-dir", default="checkpoints/rram_lstm_sequence")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task = TASK_FILES[args.task]
    train_x, train_y = read_rows(args.root, task["train"], task["seq_len"], task["input_dim"], task["target_col"])
    val_x, val_y = read_rows(args.root, task["val"], task["seq_len"], task["input_dim"], task["target_col"])
    test_x, test_y = read_rows(args.root, task["test"], task["seq_len"], task["input_dim"], task["target_col"])
    arrays = normalize(train_x, train_y, val_x, val_y, test_x, test_y)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in args.seeds:
        for variant_name in args.variants:
            print(f"Running {variant_name} seed={seed} task={args.task}", flush=True)
            results.append(run_one(args, task, arrays, variant_name, seed))
            payload = {
                "note": "RRAM LSTM public dataset run.",
                "task": args.task,
                "task_notes": task["notes"],
                "root": str(args.root),
                "results": results,
                "summary": summarize(results),
            }
            args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            write_csv(args.out.with_suffix(".summary.csv"), payload["summary"])
            write_markdown(args.out.with_suffix(".md"), payload)
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
