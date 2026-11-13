"""RRAM-sine adjusted-grid and distillation GPU batch.

This is a targeted follow-up to the no-GPU RRAM-sine diagnostic. It keeps the
original RRAM train/val/test split and normalization, but lets recurrent KAN
models use a wider fixed spline grid, then optionally distills the best KAN
teacher into dense GRU students.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
SRC = ROOT / "src"
for path in (SRC, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from memristorkan.complexity import model_complexity  # noqa: E402
from memristorkan.metrics import regression_metrics  # noqa: E402
from memristorkan.models import (  # noqa: E402
    ClippedKANGRURegressor,
    ClippedKANLSTMRegressor,
    FusedClippedKANGRURegressor,
    GRURegressor,
    HybridClippedKANGRURegressor,
    KANGRURegressor,
    KANLSTMRegressor,
    LSTMRegressor,
)
from run_rram_lstm_sequence import TASK_FILES, normalize, read_rows  # noqa: E402


@dataclass(frozen=True)
class Variant:
    name: str
    family: str
    hidden_dim: int
    grid_size: int = 4


VARIANTS: dict[str, Variant] = {
    "gru_h116": Variant("gru_h116", "gru", 116),
    "lstm_h100": Variant("lstm_h100", "lstm", 100),
    "kan_gru_h32_wide": Variant("kan_gru_h32_wide", "kan_gru", 32, 4),
    "clipped_kan_gru_h32_wide": Variant("clipped_kan_gru_h32_wide", "clipped_kan_gru", 32, 4),
    "fused_clipped_kan_gru_h32_wide": Variant("fused_clipped_kan_gru_h32_wide", "fused_clipped_kan_gru", 32, 4),
    "hybrid_clipped_kan_gru_h32_wide": Variant("hybrid_clipped_kan_gru_h32_wide", "hybrid_clipped_kan_gru", 32, 4),
    "kan_lstm_h32_wide": Variant("kan_lstm_h32_wide", "kan_lstm", 32, 4),
    "clipped_kan_lstm_h32_wide": Variant("clipped_kan_lstm_h32_wide", "clipped_kan_lstm", 32, 4),
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


def build_variant_model(variant: Variant, input_dim: int, grid_range: tuple[float, float]) -> nn.Module:
    if variant.family == "gru":
        return GRURegressor(input_dim, variant.hidden_dim, 1)
    if variant.family == "lstm":
        return LSTMRegressor(input_dim, variant.hidden_dim, 1)
    if variant.family == "kan_gru":
        return KANGRURegressor(input_dim, variant.hidden_dim, 1, grid_size=variant.grid_size, grid_range=grid_range)
    if variant.family == "clipped_kan_gru":
        return ClippedKANGRURegressor(input_dim, variant.hidden_dim, 1, grid_size=variant.grid_size, grid_range=grid_range)
    if variant.family == "fused_clipped_kan_gru":
        return FusedClippedKANGRURegressor(input_dim, variant.hidden_dim, 1, grid_size=variant.grid_size, grid_range=grid_range)
    if variant.family == "hybrid_clipped_kan_gru":
        return HybridClippedKANGRURegressor(input_dim, variant.hidden_dim, 1, grid_size=variant.grid_size, grid_range=grid_range)
    if variant.family == "kan_lstm":
        return KANLSTMRegressor(input_dim, variant.hidden_dim, 1, grid_size=variant.grid_size, grid_range=grid_range)
    if variant.family == "clipped_kan_lstm":
        return ClippedKANLSTMRegressor(input_dim, variant.hidden_dim, 1, grid_size=variant.grid_size, grid_range=grid_range)
    raise ValueError(f"unknown variant family: {variant.family}")


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, *, shuffle: bool, seed: int) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator if shuffle else None)


def predict_last(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    y = model(x)
    if y.ndim == 3:
        return y[:, -1, :]
    return y


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, *, y_mean: float, y_std: float) -> dict[str, float]:
    model.eval()
    preds = []
    targets = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
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


@torch.no_grad()
def predict_array(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size, shuffle=False)
    model.eval()
    preds = []
    for (batch_x,) in loader:
        preds.append(predict_last(model, batch_x.to(device, non_blocking=True)).cpu().numpy())
    return np.concatenate(preds, axis=0).astype(np.float32)


def load_rram_sine(root: Path) -> dict[str, np.ndarray]:
    task = TASK_FILES["sine"]
    train_x, train_y = read_rows(root, task["train"], task["seq_len"], task["input_dim"], task["target_col"])
    val_x, val_y = read_rows(root, task["val"], task["seq_len"], task["input_dim"], task["target_col"])
    test_x, test_y = read_rows(root, task["test"], task["seq_len"], task["input_dim"], task["target_col"])
    return normalize(train_x, train_y, val_x, val_y, test_x, test_y)


def train_supervised(
    args: argparse.Namespace,
    arrays: dict[str, np.ndarray],
    variant_name: str,
    seed: int,
    *,
    epochs: int,
    device: torch.device,
    checkpoint_dir: Path,
) -> tuple[nn.Module, dict[str, Any]]:
    set_seed(seed)
    variant = VARIANTS[variant_name]
    grid_range = (float(args.grid_range[0]), float(args.grid_range[1]))
    model = build_variant_model(variant, input_dim=2, grid_range=grid_range).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = make_loader(arrays["train_x"], arrays["train_y"], args.batch_size, shuffle=True, seed=seed)
    val_loader = make_loader(arrays["val_x"], arrays["val_y"], args.batch_size, shuffle=False, seed=seed)
    test_loader = make_loader(arrays["test_x"], arrays["test_y"], args.batch_size, shuffle=False, seed=seed)
    y_mean = float(arrays["y_mean"][0, 0])
    y_std = float(arrays["y_std"][0, 0])

    epoch_summaries = []
    global_steps = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start_all = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        epoch_start = time.perf_counter()
        for x, y in train_loader:
            global_steps += 1
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = predict_last(model, x)
            loss = torch.mean((pred - y) ** 2)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        val_metrics = evaluate(model, val_loader, device, y_mean=y_mean, y_std=y_std)
        epoch_summaries.append(
            {
                "epoch": epoch,
                "train_loss_mean": float(np.mean(losses)),
                "train_loss_last": float(losses[-1]),
                "epoch_seconds": time.perf_counter() - epoch_start,
                "val_metrics": val_metrics,
            }
        )
    total_seconds = time.perf_counter() - start_all
    test_metrics = evaluate(model, test_loader, device, y_mean=y_mean, y_std=y_std)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{variant_name}_sine_grid{grid_range[0]}_{grid_range[1]}_seed{seed}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "variant": asdict(variant),
            "seed": seed,
            "task": "sine",
            "grid_range": grid_range,
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
    result = {
        "run_type": "adjusted_grid_supervised",
        "variant": variant_name,
        "seed": seed,
        "epochs": epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "grid_range": list(grid_range),
        "global_steps": global_steps,
        "train_rows": int(arrays["train_x"].shape[0]),
        "val_rows": int(arrays["val_x"].shape[0]),
        "test_rows": int(arrays["test_x"].shape[0]),
        "complexity": model_complexity(model),
        "checkpoint_path": str(checkpoint_path),
        "epoch_summaries": epoch_summaries,
        "final_val_metrics": epoch_summaries[-1]["val_metrics"],
        "test_metrics": test_metrics,
        "timing": {
            "wall_clock_seconds": total_seconds,
            "seconds_per_epoch": total_seconds / max(1, epochs),
            "seconds_per_step": total_seconds / max(1, global_steps),
            "peak_cuda_memory_mb": (
                float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0
            ),
        },
    }
    return model, result


def train_distilled_student(
    args: argparse.Namespace,
    arrays: dict[str, np.ndarray],
    teacher: nn.Module,
    teacher_name: str,
    seed: int,
    hidden_dim: int,
    distill_weight: float,
    *,
    epochs: int,
    device: torch.device,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    set_seed(seed)
    teacher_train = predict_array(teacher, arrays["train_x"], args.batch_size, device)
    teacher_val = predict_array(teacher, arrays["val_x"], args.batch_size, device)
    teacher_test = predict_array(teacher, arrays["test_x"], args.batch_size, device)

    student = GRURegressor(input_dim=2, hidden_dim=hidden_dim, output_dim=1).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.student_lr, weight_decay=args.weight_decay)
    train_dataset = TensorDataset(
        torch.from_numpy(arrays["train_x"]),
        torch.from_numpy(arrays["train_y"]),
        torch.from_numpy(teacher_train),
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
    val_loader = make_loader(arrays["val_x"], arrays["val_y"], args.batch_size, shuffle=False, seed=seed)
    test_loader = make_loader(arrays["test_x"], arrays["test_y"], args.batch_size, shuffle=False, seed=seed)
    y_mean = float(arrays["y_mean"][0, 0])
    y_std = float(arrays["y_std"][0, 0])

    epoch_summaries = []
    global_steps = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start_all = time.perf_counter()
    for epoch in range(1, epochs + 1):
        student.train()
        losses = []
        epoch_start = time.perf_counter()
        for x, y, t in train_loader:
            global_steps += 1
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            t = t.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = predict_last(student, x)
            target_loss = torch.mean((pred - y) ** 2)
            distill_loss = torch.mean((pred - t) ** 2)
            loss = (1.0 - distill_weight) * target_loss + distill_weight * distill_loss
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        val_metrics = evaluate(student, val_loader, device, y_mean=y_mean, y_std=y_std)
        epoch_summaries.append(
            {
                "epoch": epoch,
                "train_loss_mean": float(np.mean(losses)),
                "train_loss_last": float(losses[-1]),
                "epoch_seconds": time.perf_counter() - epoch_start,
                "val_metrics": val_metrics,
            }
        )
    total_seconds = time.perf_counter() - start_all
    test_metrics = evaluate(student, test_loader, device, y_mean=y_mean, y_std=y_std)
    teacher_val_metrics = regression_metrics(torch.from_numpy(teacher_val), torch.from_numpy(arrays["val_y"]))
    teacher_test_metrics = regression_metrics(torch.from_numpy(teacher_test), torch.from_numpy(arrays["test_y"]))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / (
        f"distilled_gru_h{hidden_dim}_from_{teacher_name}_w{distill_weight:g}_seed{seed}.pt"
    )
    torch.save(
        {
            "model_state_dict": student.state_dict(),
            "student": {"family": "gru", "hidden_dim": hidden_dim},
            "teacher": teacher_name,
            "distill_weight": distill_weight,
            "seed": seed,
            "task": "sine",
            "complexity": model_complexity(student),
        },
        checkpoint_path,
    )
    return {
        "run_type": "offline_distillation",
        "variant": f"distilled_gru_h{hidden_dim}",
        "teacher_variant": teacher_name,
        "seed": seed,
        "epochs": epochs,
        "batch_size": args.batch_size,
        "student_lr": args.student_lr,
        "distill_weight": distill_weight,
        "global_steps": global_steps,
        "complexity": model_complexity(student),
        "checkpoint_path": str(checkpoint_path),
        "epoch_summaries": epoch_summaries,
        "final_val_metrics": epoch_summaries[-1]["val_metrics"],
        "test_metrics": test_metrics,
        "teacher_val_metrics": teacher_val_metrics,
        "teacher_test_metrics": teacher_test_metrics,
        "timing": {
            "wall_clock_seconds": total_seconds,
            "seconds_per_epoch": total_seconds / max(1, epochs),
            "seconds_per_step": total_seconds / max(1, global_steps),
            "peak_cuda_memory_mb": (
                float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0
            ),
        },
    }


def summarize(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in results:
        key = (
            row["run_type"],
            row["variant"],
            row.get("teacher_variant", ""),
            row.get("distill_weight", ""),
        )
        grouped.setdefault(key, []).append(row)
    summary = []
    for key, rows in grouped.items():
        def vals(section: str, metric: str) -> list[float]:
            return [float(row[section][metric]) for row in rows]

        summary.append(
            {
                "run_type": key[0],
                "variant": key[1],
                "teacher_variant": key[2],
                "distill_weight": key[3],
                "runs": len(rows),
                "params": rows[0]["complexity"]["total_parameters"],
                "val_r2_mean": mean(vals("final_val_metrics", "r2")),
                "val_r2_std": pstdev(vals("final_val_metrics", "r2")),
                "test_r2_mean": mean(vals("test_metrics", "r2")),
                "test_r2_std": pstdev(vals("test_metrics", "r2")),
                "test_raw_rmse_mean": mean(vals("test_metrics", "raw_rmse")),
                "seconds_per_epoch_mean": mean([float(row["timing"]["seconds_per_epoch"]) for row in rows]),
                "peak_cuda_memory_mb_max": max(float(row["timing"]["peak_cuda_memory_mb"]) for row in rows),
            }
        )
    summary.sort(key=lambda r: float(r["test_r2_mean"]), reverse=True)
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_run(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_type": row["run_type"],
        "variant": row["variant"],
        "teacher_variant": row.get("teacher_variant", ""),
        "distill_weight": row.get("distill_weight", ""),
        "seed": row["seed"],
        "epochs": row["epochs"],
        "params": row["complexity"]["total_parameters"],
        "test_r2": row["test_metrics"]["r2"],
        "test_raw_rmse": row["test_metrics"]["raw_rmse"],
        "val_r2": row["final_val_metrics"]["r2"],
        "seconds_per_epoch": row["timing"]["seconds_per_epoch"],
        "wall_clock_seconds": row["timing"]["wall_clock_seconds"],
        "peak_cuda_memory_mb": row["timing"]["peak_cuda_memory_mb"],
        "checkpoint_path": row["checkpoint_path"],
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# RRAM Sine Adjusted-Grid And Distillation Batch",
        "",
        "This run tests whether the RRAM-sine KAN weakness is improved by widening the fixed KAN spline grid and whether a dense GRU student can retain teacher behavior at lower cost.",
        "",
        f"Grid range: `{payload['grid_range']}`",
        "",
        "| Run type | Variant | Teacher | Distill weight | Runs | Params | Test R2 | Test raw RMSE | Sec/epoch |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]:
        lines.append(
            f"| `{row['run_type']}` | `{row['variant']}` | `{row['teacher_variant']}` | "
            f"{row['distill_weight']} | {row['runs']} | {int(row['params']):,} | "
            f"{row['test_r2_mean']:.4f} +/- {row['test_r2_std']:.4f} | "
            f"{row['test_raw_rmse_mean']:.6g} | {row['seconds_per_epoch_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation guide:",
            "",
            "- If adjusted-grid KAN improves over the locked RRAM-sine KAN rows, the no-GPU grid-range diagnostic is supported.",
            "- If adjusted-grid KAN remains below dense GRU, report RRAM sine as a robust dense-GRU boundary.",
            "- Distillation is useful only if a student row approaches teacher accuracy with materially lower seconds/epoch.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_payload(out: Path, payload: dict[str, Any]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(out.with_suffix(".summary.csv"), payload["summary"])
    write_csv(out.with_suffix(".per_run.csv"), [flatten_run(row) for row in payload["results"]])
    write_markdown(out.with_suffix(".md"), payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/rram_lstm"))
    parser.add_argument("--out", type=Path, default=Path("results/rram_sine_grid_distill_batch.json"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/rram_sine_grid_distill"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--kan-epochs", type=int, default=100)
    parser.add_argument("--distill-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--student-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--grid-range", nargs=2, type=float, default=[-2.0, 2.0])
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[
            "gru_h116",
            "clipped_kan_gru_h32_wide",
            "kan_gru_h32_wide",
            "fused_clipped_kan_gru_h32_wide",
            "kan_lstm_h32_wide",
        ],
        choices=list(VARIANTS),
    )
    parser.add_argument("--teacher-variant", default="clipped_kan_gru_h32_wide", choices=list(VARIANTS))
    parser.add_argument("--student-hidden", nargs="+", type=int, default=[32, 64, 116])
    parser.add_argument("--distill-weights", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    parser.add_argument("--skip-distillation", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true", help="Override to one seed, one epoch, one student setting.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.seeds = [0]
        args.kan_epochs = 1
        args.distill_epochs = 1
        args.variants = ["gru_h116", args.teacher_variant, "fused_clipped_kan_gru_h32_wide"]
        args.student_hidden = [32]
        args.distill_weights = [0.5]
    device = device_from_arg(args.device)
    arrays = load_rram_sine(args.root)
    results: list[dict[str, Any]] = []
    teacher_models: dict[int, nn.Module] = {}
    started = time.perf_counter()

    for seed in args.seeds:
        for variant_name in args.variants:
            print(f"Running adjusted-grid {variant_name} seed={seed}", flush=True)
            model, result = train_supervised(
                args,
                arrays,
                variant_name,
                seed,
                epochs=args.kan_epochs,
                device=device,
                checkpoint_dir=args.checkpoint_dir,
            )
            results.append(result)
            if variant_name == args.teacher_variant:
                teacher_models[seed] = model
            payload = {
                "note": "RRAM sine adjusted-grid and distillation GPU batch.",
                "root": str(args.root),
                "grid_range": list(args.grid_range),
                "results": results,
                "summary": summarize(results),
                "elapsed_seconds_so_far": time.perf_counter() - started,
            }
            write_payload(args.out, payload)

        if not args.skip_distillation:
            if seed not in teacher_models:
                print(f"Training missing teacher {args.teacher_variant} seed={seed}", flush=True)
                teacher, result = train_supervised(
                    args,
                    arrays,
                    args.teacher_variant,
                    seed,
                    epochs=args.kan_epochs,
                    device=device,
                    checkpoint_dir=args.checkpoint_dir,
                )
                teacher_models[seed] = teacher
                results.append(result)
            teacher = teacher_models[seed]
            for hidden_dim in args.student_hidden:
                for weight in args.distill_weights:
                    print(
                        f"Running distillation teacher={args.teacher_variant} student=gru_h{hidden_dim} "
                        f"seed={seed} weight={weight}",
                        flush=True,
                    )
                    results.append(
                        train_distilled_student(
                            args,
                            arrays,
                            teacher,
                            args.teacher_variant,
                            seed,
                            hidden_dim,
                            weight,
                            epochs=args.distill_epochs,
                            device=device,
                            checkpoint_dir=args.checkpoint_dir,
                        )
                    )
                    payload = {
                        "note": "RRAM sine adjusted-grid and distillation GPU batch.",
                        "root": str(args.root),
                        "grid_range": list(args.grid_range),
                        "results": results,
                        "summary": summarize(results),
                        "elapsed_seconds_so_far": time.perf_counter() - started,
                    }
                    write_payload(args.out, payload)

    payload = {
        "note": "RRAM sine adjusted-grid and distillation GPU batch.",
        "root": str(args.root),
        "grid_range": list(args.grid_range),
        "results": results,
        "summary": summarize(results),
        "elapsed_seconds_total": time.perf_counter() - started,
    }
    write_payload(args.out, payload)
    print(json.dumps({"out": str(args.out), "runs": len(results), "elapsed_seconds": payload["elapsed_seconds_total"]}, indent=2))


if __name__ == "__main__":
    main()
