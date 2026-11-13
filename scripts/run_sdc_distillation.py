"""Distill original SDC clipped KAN-GRU teachers into dense GRU students."""

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
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_sequence_checkpoints import load_checkpoint, variant_from_checkpoint  # noqa: E402
from memristorkan.complexity import model_complexity  # noqa: E402
from memristorkan.metrics import sequence_metrics  # noqa: E402
from memristorkan.models import GRURegressor  # noqa: E402
from run_sdc_sequence import WindowedCurveDataset, build_sequence_model  # noqa: E402


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


def teacher_path_for_seed(teacher_dir: Path, seed: int) -> Path:
    matches = sorted(teacher_dir.rglob(f"clipped_kan_gru_h32_no_current_seed{seed}.pt"))
    if not matches:
        raise FileNotFoundError(f"teacher checkpoint for seed {seed} not found under {teacher_dir}")
    return matches[0]


def make_dataset_arrays(cache: Path, split: str, window: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    dataset = WindowedCurveDataset(cache, split, window=window, stride=stride, feature_mode="no_current")
    xs = []
    ys = []
    for idx in range(len(dataset)):
        x, y = dataset[idx]
        xs.append(x.numpy())
        ys.append(y.numpy())
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)


@torch.no_grad()
def predict_sequences(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size, shuffle=False)
    model.eval()
    preds = []
    for (batch_x,) in loader:
        preds.append(model(batch_x.to(device, non_blocking=True)).cpu().numpy())
    return np.concatenate(preds, axis=0).astype(np.float32)


@torch.no_grad()
def evaluate_model(model: nn.Module, x: np.ndarray, y: np.ndarray, batch_size: int, device: torch.device) -> dict[str, float]:
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=batch_size, shuffle=False)
    model.eval()
    preds = []
    targets = []
    start = time.perf_counter()
    for batch_x, batch_y in loader:
        preds.append(model(batch_x.to(device, non_blocking=True)).cpu())
        targets.append(batch_y.cpu())
    metrics = sequence_metrics(torch.cat(preds, dim=0), torch.cat(targets, dim=0))
    metrics["eval_seconds"] = time.perf_counter() - start
    return metrics


def load_teacher(path: Path, input_dim: int, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    payload = load_checkpoint(path, device)
    variant = variant_from_checkpoint(payload)
    model = build_sequence_model(variant, input_dim=input_dim).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, {"variant": asdict(variant), "seed": payload.get("seed"), "path": str(path)}


def train_student(
    *,
    args: argparse.Namespace,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    teacher_train: np.ndarray | None,
    teacher_variant: str,
    seed: int,
    hidden_dim: int,
    mode: str,
    distill_weight: float,
    device: torch.device,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    set_seed(seed)
    model = GRURegressor(input_dim=train_x.shape[-1], hidden_dim=hidden_dim, output_dim=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if teacher_train is None:
        dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    else:
        dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y), torch.from_numpy(teacher_train))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=args.num_workers)
    epoch_summaries = []
    global_steps = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start_all = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        epoch_start = time.perf_counter()
        for batch in loader:
            global_steps += 1
            x = batch[0].to(device, non_blocking=True)
            y = batch[1].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            target_loss = torch.mean((pred - y) ** 2)
            if teacher_train is None:
                loss = target_loss
                distill_loss_value = 0.0
            else:
                teacher = batch[2].to(device, non_blocking=True)
                distill_loss = torch.mean((pred - teacher) ** 2)
                loss = (1.0 - distill_weight) * target_loss + distill_weight * distill_loss
                distill_loss_value = float(distill_loss.detach().cpu().item())
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(
                {
                    "loss": float(loss.detach().cpu().item()),
                    "target_loss": float(target_loss.detach().cpu().item()),
                    "distill_loss": distill_loss_value,
                }
            )
        val_metrics = evaluate_model(model, val_x, val_y, args.batch_size, device)
        epoch_summaries.append(
            {
                "epoch": epoch,
                "epoch_seconds": time.perf_counter() - epoch_start,
                "train_loss_mean": float(mean(row["loss"] for row in losses)),
                "target_loss_mean": float(mean(row["target_loss"] for row in losses)),
                "distill_loss_mean": float(mean(row["distill_loss"] for row in losses)),
                "val_metrics": val_metrics,
            }
        )
    total_seconds = time.perf_counter() - start_all
    test_metrics = evaluate_model(model, test_x, test_y, args.batch_size, device)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{mode}_gru_h{hidden_dim}"
    if mode == "distilled":
        suffix += f"_from_{teacher_variant}_w{distill_weight:g}"
    checkpoint_path = checkpoint_dir / f"{suffix}_seed{seed}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "mode": mode,
            "seed": seed,
            "hidden_dim": hidden_dim,
            "teacher_variant": teacher_variant,
            "distill_weight": distill_weight,
            "complexity": model_complexity(model),
        },
        checkpoint_path,
    )
    return {
        "mode": mode,
        "variant": f"{mode}_gru_h{hidden_dim}",
        "teacher_variant": teacher_variant if mode == "distilled" else "",
        "distill_weight": distill_weight if mode == "distilled" else "",
        "seed": seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "global_steps": global_steps,
        "complexity": model_complexity(model),
        "checkpoint_path": str(checkpoint_path),
        "epoch_summaries": epoch_summaries,
        "final_val_metrics": epoch_summaries[-1]["val_metrics"],
        "test_metrics": test_metrics,
        "timing": {
            "wall_clock_seconds": total_seconds,
            "seconds_per_epoch": total_seconds / max(1, args.epochs),
            "seconds_per_step": total_seconds / max(1, global_steps),
            "peak_cuda_memory_mb": (
                float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0
            ),
        },
    }


def summarize(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in results:
        grouped.setdefault((row["variant"], str(row["teacher_variant"]), str(row["distill_weight"])), []).append(row)
    summary = []
    for (variant, teacher, weight), rows in grouped.items():
        def vals(section: str, key: str) -> list[float]:
            return [float(row[section][key]) for row in rows]

        summary.append(
            {
                "variant": variant,
                "teacher_variant": teacher,
                "distill_weight": weight,
                "runs": len(rows),
                "params": rows[0]["complexity"]["total_parameters"],
                "val_r2_mean": mean(vals("final_val_metrics", "r2")),
                "val_r2_std": pstdev(vals("final_val_metrics", "r2")),
                "test_r2_mean": mean(vals("test_metrics", "r2")),
                "test_r2_std": pstdev(vals("test_metrics", "r2")),
                "test_rmse_mean": mean(vals("test_metrics", "rmse")),
                "test_final_state_mae_mean": mean(vals("test_metrics", "final_state_mae")),
                "seconds_per_epoch_mean": mean(float(row["timing"]["seconds_per_epoch"]) for row in rows),
                "peak_cuda_memory_mb_max": max(float(row["timing"]["peak_cuda_memory_mb"]) for row in rows),
            }
        )
    summary.sort(key=lambda row: float(row["test_r2_mean"]), reverse=True)
    return summary


def flatten(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "variant": row["variant"],
        "teacher_variant": row["teacher_variant"],
        "distill_weight": row["distill_weight"],
        "seed": row["seed"],
        "epochs": row["epochs"],
        "params": row["complexity"]["total_parameters"],
        "test_r2": row["test_metrics"]["r2"],
        "test_rmse": row["test_metrics"]["rmse"],
        "test_final_state_mae": row["test_metrics"]["final_state_mae"],
        "val_r2": row["final_val_metrics"]["r2"],
        "seconds_per_epoch": row["timing"]["seconds_per_epoch"],
        "wall_clock_seconds": row["timing"]["wall_clock_seconds"],
        "peak_cuda_memory_mb": row["timing"]["peak_cuda_memory_mb"],
        "checkpoint_path": row["checkpoint_path"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# SDC No-Current Teacher Distillation",
        "",
        "Dense GRU students are trained on the original SDC no-current benchmark using either supervised target loss only or a mixed target/teacher loss from the locked clipped KAN-GRU teacher.",
        "",
        "| Variant | Teacher | Distill weight | Runs | Params | Test R2 | Test RMSE | Final-state MAE | Sec/epoch |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]:
        lines.append(
            f"| `{row['variant']}` | `{row['teacher_variant']}` | {row['distill_weight']} | "
            f"{row['runs']} | {int(row['params']):,} | "
            f"{row['test_r2_mean']:.4f} +/- {row['test_r2_std']:.4f} | "
            f"{row['test_rmse_mean']:.4f} | {row['test_final_state_mae_mean']:.4f} | "
            f"{row['seconds_per_epoch_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Decision rule: distillation is useful if a distilled student beats the same-hidden-size supervised student and approaches the locked clipped KAN-GRU teacher at much lower runtime.",
            "",
            f"Teacher reference: {payload['teacher_reference']}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(path.with_suffix(".summary.csv"), payload["summary"])
    write_csv(path.with_suffix(".per_run.csv"), [flatten(row) for row in payload["results"]])
    write_markdown(path.with_suffix(".md"), payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("data/processed/sdc_state_splits.npz"))
    parser.add_argument("--teacher-dir", type=Path, default=Path("checkpoints/sdc_teachers"))
    parser.add_argument("--out", type=Path, default=Path("results/sdc_distillation_full.json"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/sdc_distillation_full"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[32, 64, 116])
    parser.add_argument("--distill-weights", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--train-stride", type=int, default=256)
    parser.add_argument("--eval-stride", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.seeds = [0]
        args.hidden_dims = [32]
        args.distill_weights = [0.5]
        args.epochs = 1
    device = device_from_arg(args.device)
    train_x, train_y = make_dataset_arrays(args.cache, "train", args.window, args.train_stride)
    val_x, val_y = make_dataset_arrays(args.cache, "val", args.window, args.eval_stride)
    test_x, test_y = make_dataset_arrays(args.cache, "test", args.window, args.eval_stride)
    results: list[dict[str, Any]] = []
    teacher_reference: dict[str, Any] = {}
    started = time.perf_counter()

    for seed in args.seeds:
        teacher_path = teacher_path_for_seed(args.teacher_dir, seed)
        teacher, teacher_info = load_teacher(teacher_path, input_dim=train_x.shape[-1], device=device)
        teacher_reference[str(seed)] = teacher_info
        teacher_train = predict_sequences(teacher, train_x, args.batch_size, device)
        teacher_test_metrics = evaluate_model(teacher, test_x, test_y, args.batch_size, device)
        print(f"Teacher seed={seed} test_r2={teacher_test_metrics['r2']:.4f}", flush=True)

        for hidden_dim in args.hidden_dims:
            print(f"Running supervised GRU h{hidden_dim} seed={seed}", flush=True)
            results.append(
                train_student(
                    args=args,
                    train_x=train_x,
                    train_y=train_y,
                    val_x=val_x,
                    val_y=val_y,
                    test_x=test_x,
                    test_y=test_y,
                    teacher_train=None,
                    teacher_variant="",
                    seed=seed,
                    hidden_dim=hidden_dim,
                    mode="supervised",
                    distill_weight=0.0,
                    device=device,
                    checkpoint_dir=args.checkpoint_dir,
                )
            )
            payload = {
                "note": "Original SDC no-current distillation from clipped KAN-GRU teachers.",
                "teacher_reference": teacher_reference,
                "results": results,
                "summary": summarize(results),
                "elapsed_seconds_so_far": time.perf_counter() - started,
            }
            write_payload(args.out, payload)

            for weight in args.distill_weights:
                print(f"Running distilled GRU h{hidden_dim} seed={seed} weight={weight}", flush=True)
                results.append(
                    train_student(
                        args=args,
                        train_x=train_x,
                        train_y=train_y,
                        val_x=val_x,
                        val_y=val_y,
                        test_x=test_x,
                        test_y=test_y,
                        teacher_train=teacher_train,
                        teacher_variant="clipped_kan_gru_h32",
                        seed=seed,
                        hidden_dim=hidden_dim,
                        mode="distilled",
                        distill_weight=weight,
                        device=device,
                        checkpoint_dir=args.checkpoint_dir,
                    )
                )
                payload = {
                    "note": "Original SDC no-current distillation from clipped KAN-GRU teachers.",
                    "teacher_reference": teacher_reference,
                    "results": results,
                    "summary": summarize(results),
                    "elapsed_seconds_so_far": time.perf_counter() - started,
                }
                write_payload(args.out, payload)

    payload = {
        "note": "Original SDC no-current distillation from clipped KAN-GRU teachers.",
        "teacher_reference": teacher_reference,
        "results": results,
        "summary": summarize(results),
        "elapsed_seconds_total": time.perf_counter() - started,
    }
    write_payload(args.out, payload)
    print(json.dumps({"out": str(args.out), "runs": len(results), "elapsed_seconds": payload["elapsed_seconds_total"]}, indent=2))


if __name__ == "__main__":
    main()
