"""Fine-tune original-SDC sequence checkpoints on another SDC-style cache.

This is a transfer-learning quick-pass runner for caches with the same schema
as ``scripts/run_sdc_sequence.py``. It evaluates source checkpoints zero-shot,
then compares head-only/full fine-tuning against same-budget scratch training
on the target cache's true test split.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from memristorkan.complexity import model_complexity

from run_sdc_sequence import (
    VARIANTS,
    SequenceVariant,
    WindowedCurveDataset,
    build_sequence_model,
    current_lags_for_mode,
    evaluate_autoregressive_rollout,
    evaluate_teacher_forced,
    feature_names_for_mode,
)


DEFAULT_SOURCE_CHECKPOINTS = {
    "static_mlp_h200": "checkpoints/sequence/static_mlp_h200_no_current_seed{seed}.pt",
    "lstm_h100": (
        "checkpoints/kaggle_imported/no_current_baselines_100epoch_3seed/"
        "sequence/lstm_h100_no_current_seed{seed}.pt"
    ),
    "gru_h116": (
        "checkpoints/kaggle_imported/no_current_baselines_100epoch_3seed/"
        "sequence/gru_h116_no_current_seed{seed}.pt"
    ),
    "clipped_kan_gru_h32": (
        "results/kaggle_imported/{source_dir}/checkpoints/sequence/"
        "clipped_kan_gru_h32_no_current_seed{seed}.pt"
    ),
}

KAN_SOURCE_DIR_BY_SEED = {
    0: "kangru_sdc_100epoch_seed0",
    1: "clipped_kangru_sdc_100epoch_seed1_2",
    2: "clipped_kangru_sdc_100epoch_seed1_2",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def variant_from_checkpoint(payload: dict[str, Any]) -> SequenceVariant:
    variant = payload.get("variant")
    if not isinstance(variant, dict):
        raise ValueError(f"checkpoint does not contain a variant dictionary: {payload.keys()}")
    allowed = {"name", "model_name", "hidden_dim", "grid_size", "gauge_every_steps", "moment_mode"}
    return SequenceVariant(**{key: value for key, value in variant.items() if key in allowed})


def default_source_checkpoint(variant_name: str, seed: int) -> Path:
    if variant_name == "clipped_kan_gru_h32":
        source_dir = KAN_SOURCE_DIR_BY_SEED.get(seed)
        if source_dir is None:
            raise ValueError(f"no default clipped KAN-GRU source checkpoint for seed {seed}")
        return Path(DEFAULT_SOURCE_CHECKPOINTS[variant_name].format(seed=seed, source_dir=source_dir))
    template = DEFAULT_SOURCE_CHECKPOINTS.get(variant_name)
    if template is None:
        raise ValueError(f"no default source checkpoint for variant {variant_name}")
    return Path(template.format(seed=seed))


def parse_checkpoint_overrides(values: list[str]) -> dict[tuple[str, int], Path]:
    overrides: dict[tuple[str, int], Path] = {}
    for value in values:
        try:
            variant, seed_text, path_text = value.split("=", 2)
        except ValueError as exc:
            raise ValueError("--source-checkpoint entries must be variant=seed=path") from exc
        overrides[(variant, int(seed_text))] = Path(path_text)
    return overrides


def resolve_source_checkpoint(
    variant_name: str,
    seed: int,
    overrides: dict[tuple[str, int], Path],
) -> Path:
    path = overrides.get((variant_name, seed), default_source_checkpoint(variant_name, seed))
    if not path.exists():
        raise FileNotFoundError(f"source checkpoint not found for {variant_name} seed {seed}: {path}")
    return path


def build_loaders(
    *,
    cache: Path,
    feature_mode: str,
    window: int,
    train_stride: int,
    eval_stride: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    seed: int,
    max_train_windows: int | None,
    max_eval_windows: int | None,
) -> tuple[WindowedCurveDataset, WindowedCurveDataset, DataLoader, DataLoader]:
    train_dataset = WindowedCurveDataset(
        cache,
        "train",
        window=window,
        stride=train_stride,
        feature_mode=feature_mode,
        max_windows=max_train_windows,
    )
    test_dataset = WindowedCurveDataset(
        cache,
        "test",
        window=window,
        stride=eval_stride,
        feature_mode=feature_mode,
        max_windows=max_eval_windows,
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    return train_dataset, test_dataset, train_loader, test_loader


def validate_checkpoint(
    *,
    payload: dict[str, Any],
    source_path: Path,
    expected_variant: str,
    feature_mode: str,
    target_input_dim: int,
) -> SequenceVariant:
    variant = variant_from_checkpoint(payload)
    source_feature_mode = str(payload.get("feature_mode", ""))
    source_input_dim = int(payload.get("model_input_dim", -1))
    if variant.name != expected_variant:
        raise ValueError(f"{source_path} variant is {variant.name}, expected {expected_variant}")
    if source_feature_mode != feature_mode:
        raise ValueError(f"{source_path} feature_mode is {source_feature_mode}, expected {feature_mode}")
    if source_input_dim != target_input_dim:
        raise ValueError(f"{source_path} input dim is {source_input_dim}, target dim is {target_input_dim}")
    return variant


def set_trainable_for_mode(model: nn.Module, mode: str) -> None:
    for param in model.parameters():
        param.requires_grad = True
    if mode == "full":
        return
    if mode != "head_only":
        raise ValueError(f"unexpected trainable mode: {mode}")

    for param in model.parameters():
        param.requires_grad = False
    if hasattr(model, "head"):
        for param in model.head.parameters():  # type: ignore[attr-defined]
            param.requires_grad = True
        return
    if hasattr(model, "net") and isinstance(model.net, nn.Sequential):  # type: ignore[attr-defined]
        for module in reversed(model.net):  # type: ignore[attr-defined]
            if isinstance(module, nn.Linear):
                for param in module.parameters():
                    param.requires_grad = True
                return
    raise ValueError(f"could not identify trainable head for {type(model).__name__}")


def train_epochs(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    grad_clip: float,
    weight_decay: float,
) -> dict[str, Any]:
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise ValueError("no trainable parameters")
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    epoch_summaries: list[dict[str, float]] = []
    start_all = time.perf_counter()
    global_steps = 0
    for epoch in range(1, epochs + 1):
        model.train()
        start_epoch = time.perf_counter()
        losses: list[float] = []
        for x, y in loader:
            global_steps += 1
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = torch.mean((pred - y) ** 2)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        epoch_summaries.append(
            {
                "epoch": float(epoch),
                "train_loss_mean": float(np.mean(losses)),
                "train_loss_last": float(losses[-1]),
                "epoch_seconds": time.perf_counter() - start_epoch,
            }
        )
    wall = time.perf_counter() - start_all
    return {
        "global_steps": global_steps,
        "wall_clock_seconds": wall,
        "seconds_per_epoch": wall / max(1, epochs),
        "epoch_summaries": epoch_summaries,
    }


def evaluate_model(
    *,
    model: nn.Module,
    loader: DataLoader,
    cache: Path,
    feature_mode: str,
    device: torch.device,
    rollout_eval: bool,
) -> tuple[dict[str, float], dict[str, float], float]:
    start = time.perf_counter()
    teacher = evaluate_teacher_forced(model, loader, device)
    rollout: dict[str, float] = {}
    if rollout_eval:
        cache_data = np.load(cache)
        rollout = evaluate_autoregressive_rollout(
            model,
            loader,
            device,
            previous_current_lags=current_lags_for_mode(feature_mode),
            x_mean_prev_current=float(cache_data["x_mean"][0, 2]),
            x_std_prev_current=float(cache_data["x_std"][0, 2]),
            y_mean=float(cache_data["y_mean"][0, 0]),
            y_std=float(cache_data["y_std"][0, 0]),
        )
    return teacher, rollout, time.perf_counter() - start


def run_transfer_case(
    *,
    args: argparse.Namespace,
    target_name: str,
    target_cache: Path,
    variant_name: str,
    seed: int,
    mode: str,
    source_path: Path | None,
    overrides: dict[tuple[str, int], Path],
) -> dict[str, Any]:
    set_seed(seed)
    device = torch.device(args.device)
    train_dataset, test_dataset, train_loader, test_loader = build_loaders(
        cache=target_cache,
        feature_mode=args.feature_mode,
        window=args.window,
        train_stride=args.train_stride,
        eval_stride=args.eval_stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        seed=seed,
        max_train_windows=args.max_train_windows,
        max_eval_windows=args.max_eval_windows,
    )

    if mode == "scratch_same_budget":
        variant = VARIANTS[variant_name]
        model = build_sequence_model(variant, input_dim=train_dataset.feature_dim).to(device)
        source_checkpoint = None
    else:
        checkpoint = source_path or resolve_source_checkpoint(variant_name, seed, overrides)
        payload = load_checkpoint(checkpoint, device)
        variant = validate_checkpoint(
            payload=payload,
            source_path=checkpoint,
            expected_variant=variant_name,
            feature_mode=args.feature_mode,
            target_input_dim=train_dataset.feature_dim,
        )
        model = build_sequence_model(variant, input_dim=train_dataset.feature_dim).to(device)
        model.load_state_dict(payload["model_state_dict"])
        source_checkpoint = str(checkpoint)

    train_info = {
        "global_steps": 0,
        "wall_clock_seconds": 0.0,
        "seconds_per_epoch": 0.0,
        "epoch_summaries": [],
    }
    if mode in {"head_only", "full", "scratch_same_budget"}:
        set_trainable_for_mode(model, "full" if mode in {"full", "scratch_same_budget"} else "head_only")
        train_info = train_epochs(
            model=model,
            loader=train_loader,
            device=device,
            epochs=args.epochs,
            lr=args.scratch_lr if mode == "scratch_same_budget" else args.lr,
            grad_clip=args.grad_clip,
            weight_decay=args.weight_decay,
        )
    elif mode != "zero_shot":
        raise ValueError(f"unknown adaptation mode: {mode}")

    teacher, rollout, eval_seconds = evaluate_model(
        model=model,
        loader=test_loader,
        cache=target_cache,
        feature_mode=args.feature_mode,
        device=device,
        rollout_eval=args.rollout_eval,
    )

    return {
        "source_dataset": args.source_name,
        "target_fold": target_name,
        "target_cache": str(target_cache),
        "variant": variant.name,
        "variant_config": asdict(variant),
        "adaptation_mode": mode,
        "seed": seed,
        "source_checkpoint": source_checkpoint,
        "feature_mode": args.feature_mode,
        "feature_names": feature_names_for_mode(args.feature_mode),
        "model_input_dim": train_dataset.feature_dim,
        "epochs": 0 if mode == "zero_shot" else args.epochs,
        "lr": 0.0 if mode == "zero_shot" else (args.scratch_lr if mode == "scratch_same_budget" else args.lr),
        "batch_size": args.batch_size,
        "window": args.window,
        "train_stride": args.train_stride,
        "eval_stride": args.eval_stride,
        "train_windows": len(train_dataset),
        "test_windows": len(test_dataset),
        "complexity": model_complexity(model),
        "teacher_forced_test_metrics": teacher,
        "rollout_test_metrics": rollout,
        "timing": {
            **{key: value for key, value in train_info.items() if key != "epoch_summaries"},
            "eval_wall_clock_seconds": eval_seconds,
        },
        "epoch_summaries": train_info["epoch_summaries"],
    }


def summarize(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in results:
        groups.setdefault((row["target_fold"], row["variant"], row["adaptation_mode"]), []).append(row)
    summary: list[dict[str, Any]] = []
    for (target, variant, mode), rows in groups.items():
        teacher_r2 = [float(row["teacher_forced_test_metrics"]["r2"]) for row in rows]
        teacher_rmse = [float(row["teacher_forced_test_metrics"]["rmse"]) for row in rows]
        final_mae = [float(row["teacher_forced_test_metrics"]["final_state_mae"]) for row in rows]
        rollout = rows[0].get("rollout_test_metrics", {})
        row_summary = {
            "target_fold": target,
            "variant": variant,
            "adaptation_mode": mode,
            "runs": len(rows),
            "epochs": int(rows[0]["epochs"]),
            "params": int(rows[0]["complexity"]["total_parameters"]),
            "test_r2_mean": mean(teacher_r2),
            "test_r2_std": pstdev(teacher_r2),
            "test_rmse_mean": mean(teacher_rmse),
            "final_state_mae_mean": mean(final_mae),
            "seconds_per_epoch_mean": mean([float(row["timing"]["seconds_per_epoch"]) for row in rows]),
            "eval_seconds_mean": mean([float(row["timing"]["eval_wall_clock_seconds"]) for row in rows]),
        }
        if rollout:
            row_summary.update(
                {
                    "rollout_r2_mean": mean([float(row["rollout_test_metrics"]["rollout_r2"]) for row in rows]),
                    "rollout_r2_std": pstdev([float(row["rollout_test_metrics"]["rollout_r2"]) for row in rows]),
                    "rollout_rmse_mean": mean([float(row["rollout_test_metrics"]["rollout_rmse"]) for row in rows]),
                }
            )
        summary.append(row_summary)
    summary.sort(key=lambda row: (row["target_fold"], row["variant"], row["adaptation_mode"]))
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def flatten_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in results:
        tf = row["teacher_forced_test_metrics"]
        ro = row.get("rollout_test_metrics", {})
        rows.append(
            {
                "source_dataset": row["source_dataset"],
                "target_fold": row["target_fold"],
                "variant": row["variant"],
                "adaptation_mode": row["adaptation_mode"],
                "seed": row["seed"],
                "epochs": row["epochs"],
                "params": row["complexity"]["total_parameters"],
                "test_r2": tf["r2"],
                "test_rmse": tf["rmse"],
                "test_mae": tf["mae"],
                "final_state_mae": tf["final_state_mae"],
                "rollout_r2": ro.get("rollout_r2", float("nan")),
                "rollout_rmse": ro.get("rollout_rmse", float("nan")),
                "seconds_per_epoch": row["timing"]["seconds_per_epoch"],
                "eval_seconds": row["timing"]["eval_wall_clock_seconds"],
                "source_checkpoint": row["source_checkpoint"],
            }
        )
    return rows


def write_markdown(path: Path, summary: list[dict[str, Any]]) -> None:
    lines = [
        "# SDC-to-Four-Dopant Fine-Tuning Quick Pass",
        "",
        "Source dataset: original SDC state-characterisation no-current checkpoints.",
        "",
        "Decision rule: fine-tuning is useful only if `head_only` or `full` beats `scratch_same_budget` on the same target fold and model.",
        "",
        "| Target | Model | Mode | Runs | Epochs | Test R2 | Test RMSE | Final-state MAE | sec/epoch | Params |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(summary, key=lambda r: (r["target_fold"], r["variant"], -r["test_r2_mean"])):
        lines.append(
            "| `{target_fold}` | `{variant}` | `{adaptation_mode}` | {runs} | {epochs} | "
            "{r2:.4f} +/- {std:.4f} | {rmse:.4f} | {fmae:.4f} | {sec:.3f} | {params:,} |".format(
                target_fold=row["target_fold"],
                variant=row["variant"],
                adaptation_mode=row["adaptation_mode"],
                runs=row["runs"],
                epochs=row["epochs"],
                r2=row["test_r2_mean"],
                std=row["test_r2_std"],
                rmse=row["test_rmse_mean"],
                fmae=row["final_state_mae_mean"],
                sec=row["seconds_per_epoch_mean"],
                params=row["params"],
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(out: Path, results: list[dict[str, Any]]) -> None:
    summary = summarize(results)
    payload = {
        "note": "SDC-to-SDC-cache transfer quick pass. Evaluation uses target true test split.",
        "results": results,
        "summary": summary,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(out.with_name(out.stem + "_summary.csv"), summary)
    write_csv(out.with_name(out.stem + "_per_run.csv"), flatten_results(results))
    write_markdown(out.with_suffix(".md"), summary)


def parse_target(values: list[str]) -> dict[str, Path]:
    targets: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            path = Path(value)
            targets[path.stem] = path
            continue
        name, path = value.split("=", 1)
        targets[name] = Path(path)
    return targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["holdout_w=data/processed/sdc4dopant_holdout_w.npz"],
        help="Target cache entries as name=path.",
    )
    parser.add_argument("--out", type=Path, default=Path("results/transfer/sdc_to_sdc4dopant_quickpass.json"))
    parser.add_argument("--variants", nargs="+", default=["static_mlp_h200", "gru_h116", "lstm_h100"], choices=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["zero_shot", "head_only", "full", "scratch_same_budget"],
        choices=["zero_shot", "head_only", "full", "scratch_same_budget"],
    )
    parser.add_argument("--source-checkpoint", action="append", default=[], help="Override as variant=seed=path.")
    parser.add_argument("--source-name", default="original_sdc_state_characterisation")
    parser.add_argument("--feature-mode", default="no_current", choices=["no_current"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--train-stride", type=int, default=256)
    parser.add_argument("--eval-stride", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--scratch-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--max-eval-windows", type=int, default=0)
    parser.add_argument("--rollout-eval", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_train_windows <= 0:
        args.max_train_windows = None
    if args.max_eval_windows <= 0:
        args.max_eval_windows = None
    overrides = parse_checkpoint_overrides(args.source_checkpoint)
    targets = parse_target(args.targets)
    results: list[dict[str, Any]] = []
    for target_name, target_cache in targets.items():
        if not target_cache.exists():
            raise FileNotFoundError(f"target cache not found: {target_cache}")
        for seed in args.seeds:
            for variant_name in args.variants:
                source_path = None
                if "scratch_same_budget" not in args.modes or len(args.modes) > 1:
                    source_path = resolve_source_checkpoint(variant_name, seed, overrides)
                for mode in args.modes:
                    print(f"Running target={target_name} variant={variant_name} seed={seed} mode={mode}", flush=True)
                    results.append(
                        run_transfer_case(
                            args=args,
                            target_name=target_name,
                            target_cache=target_cache,
                            variant_name=variant_name,
                            seed=seed,
                            mode=mode,
                            source_path=source_path,
                            overrides=overrides,
                        )
                    )
                    write_outputs(args.out, results)
    print(args.out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
