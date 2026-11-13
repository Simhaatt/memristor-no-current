"""Run Track B/C sequence models on SDC curve windows.

This is the Kaggle/T4-oriented runner. It trains on truncated windows sampled
from curve-wise splits, records timing for every model, and checkpoints after
each variant/seed so long quota runs do not lose completed results.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from memristorkan.complexity import model_complexity
from memristorkan.gauge import GaugeProjectionOptions, gauge_project_model
from memristorkan.metrics import sequence_metrics
from memristorkan.models import (
    BoundedKANLSTMRegressor,
    ClippedKANGRURegressor,
    ClippedKANLSTMRegressor,
    FusedClippedKANGRURegressor,
    GRURegressor,
    HybridClippedKANGRURegressor,
    KANGRURegressor,
    KANLSTMRegressor,
    LSTMRegressor,
    StaticMLPSequenceRegressor,
)
from memristorkan.train import ModelConfig, build_model


@dataclass(frozen=True)
class SequenceVariant:
    name: str
    model_name: str
    hidden_dim: int
    grid_size: int = 4
    gauge_every_steps: int = 0
    moment_mode: str = "unchanged"


VARIANTS = {
    "lstm": SequenceVariant("lstm", "lstm", hidden_dim=32),
    "gru": SequenceVariant("gru", "gru", hidden_dim=32),
    "lstm_h100": SequenceVariant("lstm_h100", "lstm", hidden_dim=100),
    "gru_h116": SequenceVariant("gru_h116", "gru", hidden_dim=116),
    "static_mlp_h116": SequenceVariant("static_mlp_h116", "static_mlp_sequence", hidden_dim=116),
    "static_mlp_h200": SequenceVariant("static_mlp_h200", "static_mlp_sequence", hidden_dim=200),
    "kan_gru_h32": SequenceVariant("kan_gru_h32", "kan_gru", hidden_dim=32, grid_size=4),
    "clipped_kan_gru_h32_g3": SequenceVariant(
        "clipped_kan_gru_h32_g3",
        "clipped_kan_gru",
        hidden_dim=32,
        grid_size=3,
    ),
    "clipped_kan_gru_h32": SequenceVariant(
        "clipped_kan_gru_h32",
        "clipped_kan_gru",
        hidden_dim=32,
        grid_size=4,
    ),
    "clipped_kan_gru_h32_g6": SequenceVariant(
        "clipped_kan_gru_h32_g6",
        "clipped_kan_gru",
        hidden_dim=32,
        grid_size=6,
    ),
    "fused_clipped_kan_gru_h32": SequenceVariant(
        "fused_clipped_kan_gru_h32",
        "fused_clipped_kan_gru",
        hidden_dim=32,
        grid_size=4,
    ),
    "hybrid_clipped_kan_gru_h32": SequenceVariant(
        "hybrid_clipped_kan_gru_h32",
        "hybrid_clipped_kan_gru",
        hidden_dim=32,
        grid_size=4,
    ),
    "hybrid_clipped_kan_gru_h64": SequenceVariant(
        "hybrid_clipped_kan_gru_h64",
        "hybrid_clipped_kan_gru",
        hidden_dim=64,
        grid_size=4,
    ),
    "kan_lstm": SequenceVariant("kan_lstm", "kan_lstm", hidden_dim=8, grid_size=4),
    "gauge_kan_lstm": SequenceVariant(
        "gauge_kan_lstm",
        "gauge_kan_lstm",
        hidden_dim=8,
        grid_size=4,
        gauge_every_steps=1,
        moment_mode="reset",
    ),
    "kan_lstm_h32": SequenceVariant("kan_lstm_h32", "kan_lstm", hidden_dim=32, grid_size=4),
    "gauge_kan_lstm_h32": SequenceVariant(
        "gauge_kan_lstm_h32",
        "gauge_kan_lstm",
        hidden_dim=32,
        grid_size=4,
        gauge_every_steps=1,
        moment_mode="reset",
    ),
    "bounded_kan_lstm": SequenceVariant("bounded_kan_lstm", "bounded_kan_lstm", hidden_dim=8, grid_size=4),
    "gauge_bounded_kan_lstm": SequenceVariant(
        "gauge_bounded_kan_lstm",
        "gauge_bounded_kan_lstm",
        hidden_dim=8,
        grid_size=4,
        gauge_every_steps=1,
        moment_mode="reset",
    ),
    "bounded_kan_lstm_h32": SequenceVariant(
        "bounded_kan_lstm_h32",
        "bounded_kan_lstm",
        hidden_dim=32,
        grid_size=4,
    ),
    "gauge_bounded_kan_lstm_h32": SequenceVariant(
        "gauge_bounded_kan_lstm_h32",
        "gauge_bounded_kan_lstm",
        hidden_dim=32,
        grid_size=4,
        gauge_every_steps=1,
        moment_mode="reset",
    ),
    "clipped_kan_lstm": SequenceVariant("clipped_kan_lstm", "clipped_kan_lstm", hidden_dim=8, grid_size=4),
    "gauge_clipped_kan_lstm": SequenceVariant(
        "gauge_clipped_kan_lstm",
        "gauge_clipped_kan_lstm",
        hidden_dim=8,
        grid_size=4,
        gauge_every_steps=1,
        moment_mode="reset",
    ),
    "clipped_kan_lstm_h32": SequenceVariant(
        "clipped_kan_lstm_h32",
        "clipped_kan_lstm",
        hidden_dim=32,
        grid_size=4,
    ),
    "gauge_clipped_kan_lstm_h32": SequenceVariant(
        "gauge_clipped_kan_lstm_h32",
        "gauge_clipped_kan_lstm",
        hidden_dim=32,
        grid_size=4,
        gauge_every_steps=1,
        moment_mode="reset",
    ),
}


BASE_FEATURE_NAMES = ["V_t", "V_t_minus_1", "I_t_minus_1", "dV_t", "sweep_direction"]

FEATURE_MODES = {
    "full": {
        "names": BASE_FEATURE_NAMES,
        "columns": [0, 1, 2, 3, 4],
        "current_lags": [(2, 1)],
    },
    "minimal_history": {
        "names": ["V_t", "V_t_minus_1", "I_t_minus_1"],
        "columns": [0, 1, 2],
        "current_lags": [(2, 1)],
    },
    "no_current": {
        "names": ["V_t", "V_t_minus_1", "dV_t", "sweep_direction"],
        "columns": [0, 1, 3, 4],
        "current_lags": [],
    },
    "strict_source_voltage": {
        "names": [
            "source_voltage_t",
            "source_voltage_t_minus_1",
            "delta_source_voltage_t",
            "source_sweep_direction",
        ],
        "columns": [0, 1, 3, 4],
        "current_lags": [],
    },
    "voltage_only": {
        "names": ["V_t", "V_t_minus_1"],
        "columns": [0, 1],
        "current_lags": [],
    },
    "state_change_direction": {
        "names": ["I_t_minus_1", "dV_t", "sweep_direction"],
        "columns": [2, 3, 4],
        "current_lags": [(0, 1)],
    },
    "voltage_direction": {
        "names": ["V_t", "sweep_direction"],
        "columns": [0, 4],
        "current_lags": [],
    },
    "current_memory_only": {
        "names": ["V_t", "I_t_minus_1"],
        "columns": [0, 2],
        "current_lags": [(1, 1)],
    },
    "previous_state_only": {
        "names": ["V_t_minus_1", "I_t_minus_1"],
        "columns": [1, 2],
        "current_lags": [(1, 1)],
    },
    "vt_only": {
        "names": ["V_t"],
        "columns": [0],
        "current_lags": [],
    },
}


def feature_names_for_mode(feature_mode: str) -> list[str]:
    if feature_mode in FEATURE_MODES:
        return list(FEATURE_MODES[feature_mode]["names"])
    if feature_mode == "history2":
        return ["V_t", "V_t_minus_1", "V_t_minus_2", "I_t_minus_1", "I_t_minus_2"]
    if feature_mode == "history3":
        return [
            "V_t",
            "V_t_minus_1",
            "V_t_minus_2",
            "V_t_minus_3",
            "I_t_minus_1",
            "I_t_minus_2",
            "I_t_minus_3",
        ]
    raise ValueError(f"unknown feature_mode: {feature_mode}")


def current_lags_for_mode(feature_mode: str) -> list[tuple[int, int]]:
    if feature_mode in FEATURE_MODES:
        return list(FEATURE_MODES[feature_mode]["current_lags"])
    if feature_mode == "history2":
        return [(3, 1), (4, 2)]
    if feature_mode == "history3":
        return [(4, 1), (5, 2), (6, 3)]
    raise ValueError(f"unknown feature_mode: {feature_mode}")


def _shift_curve_feature(
    curve_x: np.ndarray,
    length: int,
    *,
    source_column: int,
    shift: int,
) -> np.ndarray:
    shifted = np.zeros(curve_x.shape[0], dtype=np.float32)
    if length <= 0:
        return shifted
    valid = curve_x[:length, source_column]
    if shift <= 0:
        shifted[:length] = valid
        return shifted
    shifted[: min(shift, length)] = valid[0]
    if length > shift:
        shifted[shift:length] = valid[: length - shift]
    return shifted


def build_feature_array(x: np.ndarray, lengths: np.ndarray, feature_mode: str) -> np.ndarray:
    if feature_mode in FEATURE_MODES:
        columns = FEATURE_MODES[feature_mode]["columns"]
        return x[:, :, columns].astype(np.float32)
    if feature_mode not in {"history2", "history3"}:
        raise ValueError(f"unknown feature_mode: {feature_mode}")

    features = np.zeros((x.shape[0], x.shape[1], len(feature_names_for_mode(feature_mode))), dtype=np.float32)
    for curve_idx, length_value in enumerate(lengths):
        length = int(length_value)
        curve_x = x[curve_idx]
        if feature_mode == "history2":
            parts = [
                curve_x[:, 0],
                curve_x[:, 1],
                _shift_curve_feature(curve_x, length, source_column=1, shift=1),
                curve_x[:, 2],
                _shift_curve_feature(curve_x, length, source_column=2, shift=1),
            ]
        else:
            parts = [
                curve_x[:, 0],
                curve_x[:, 1],
                _shift_curve_feature(curve_x, length, source_column=1, shift=1),
                _shift_curve_feature(curve_x, length, source_column=1, shift=2),
                curve_x[:, 2],
                _shift_curve_feature(curve_x, length, source_column=2, shift=1),
                _shift_curve_feature(curve_x, length, source_column=2, shift=2),
            ]
        features[curve_idx] = np.stack(parts, axis=-1)
    return features


class WindowedCurveDataset(Dataset):
    def __init__(
        self,
        cache: Path,
        split: str,
        *,
        window: int,
        stride: int,
        feature_mode: str,
        max_windows: int | None = None,
    ) -> None:
        data = np.load(cache)
        x = data[f"{split}_curve_x"].astype(np.float32)
        y = data[f"{split}_curve_y"].astype(np.float32)
        lengths = data[f"{split}_curve_lengths"].astype(np.int32)
        self.x = build_feature_array(x, lengths, feature_mode)
        self.y = y
        self.feature_names = feature_names_for_mode(feature_mode)
        self.windows: list[tuple[int, int]] = []
        for curve_idx, length in enumerate(lengths):
            if length <= 0:
                continue
            if length <= window:
                self.windows.append((curve_idx, 0))
                continue
            for start in range(0, int(length) - window + 1, stride):
                self.windows.append((curve_idx, start))
            last_start = int(length) - window
            if self.windows[-1] != (curve_idx, last_start):
                self.windows.append((curve_idx, last_start))
        if max_windows is not None and max_windows > 0 and len(self.windows) > max_windows:
            rng = np.random.default_rng(0)
            keep = np.sort(rng.choice(len(self.windows), size=max_windows, replace=False))
            self.windows = [self.windows[int(i)] for i in keep]
        self.window = window
        self.feature_dim = self.x.shape[-1]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        curve_idx, start = self.windows[idx]
        end = start + self.window
        return torch.from_numpy(self.x[curve_idx, start:end]), torch.from_numpy(self.y[curve_idx, start:end])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_from_arg(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def build_sequence_model(variant: SequenceVariant, input_dim: int) -> nn.Module:
    return build_model(
        ModelConfig(
            name=variant.model_name,
            input_dim=input_dim,
            output_dim=1,
            hidden_dim=variant.hidden_dim,
            hidden_dims=(variant.hidden_dim,),
            grid_size=variant.grid_size,
            spline_order=3,
        )
    )


def current_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024**2))


def evaluate_teacher_forced(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            preds.append(model(x).cpu())
            targets.append(y.cpu())
    return sequence_metrics(torch.cat(preds, dim=0), torch.cat(targets, dim=0))


@torch.no_grad()
def _sequence_step(
    model: nn.Module,
    x_t: torch.Tensor,
    state: object | None,
) -> tuple[torch.Tensor, object | None]:
    if isinstance(model, LSTMRegressor):
        out, next_state = model.rnn(x_t.unsqueeze(1), state)
        return model.head(out[:, -1, :]), next_state
    if isinstance(model, GRURegressor):
        out, next_state = model.rnn(x_t.unsqueeze(1), state)
        return model.head(out[:, -1, :]), next_state
    if isinstance(model, StaticMLPSequenceRegressor):
        return model(x_t.unsqueeze(1))[:, -1, :], None
    if isinstance(
        model,
        (KANGRURegressor, ClippedKANGRURegressor, FusedClippedKANGRURegressor, HybridClippedKANGRURegressor),
    ):
        batch = x_t.shape[0]
        h_t = x_t.new_zeros(batch, model.hidden_dim) if state is None else state
        h_t = model.cell(x_t, h_t)
        return model.head(h_t), h_t
    if isinstance(model, (KANLSTMRegressor, BoundedKANLSTMRegressor, ClippedKANLSTMRegressor)):
        batch = x_t.shape[0]
        if state is None:
            h_t = x_t.new_zeros(batch, model.hidden_dim)
            c_t = x_t.new_zeros(batch, model.hidden_dim)
        else:
            h_t, c_t = state
        h_t, c_t = model.cell(x_t, (h_t, c_t))
        return model.head(h_t), (h_t, c_t)
    y = model(x_t.unsqueeze(1))
    return y[:, -1, :], None


@torch.no_grad()
def evaluate_autoregressive_rollout(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    previous_current_lags: list[tuple[int, int]],
    x_mean_prev_current: float,
    x_std_prev_current: float,
    y_mean: float,
    y_std: float,
) -> dict[str, float]:
    """Evaluate sequence windows by feeding predicted current back as input."""

    model.eval()
    preds = []
    targets = []
    for x, y in loader:
        x = x.to(device, non_blocking=True).clone()
        y = y.to(device, non_blocking=True)
        batch, seq_len, _ = x.shape
        h_preds = []
        state = None
        for t in range(seq_len):
            for feature_index, lag in previous_current_lags:
                if t < lag:
                    continue
                raw_prev = h_preds[t - lag][:, 0] * y_std + y_mean
                normalized_prev = (raw_prev - x_mean_prev_current) / x_std_prev_current
                x[:, t, feature_index] = normalized_prev
            y_t, state = _sequence_step(model, x[:, t, :], state)
            h_preds.append(y_t)
        pred = torch.stack(h_preds, dim=1).reshape(batch, seq_len, 1)
        preds.append(pred.cpu())
        targets.append(y.cpu())
    metrics = sequence_metrics(torch.cat(preds, dim=0), torch.cat(targets, dim=0))
    return {f"rollout_{key}": value for key, value in metrics.items()}


def run_one(args: argparse.Namespace, variant: SequenceVariant, seed: int) -> dict[str, object]:
    set_seed(seed)
    device = device_from_arg(args.device)
    train_dataset = WindowedCurveDataset(
        Path(args.cache),
        "train",
        window=args.window,
        stride=args.stride,
        feature_mode=args.feature_mode,
        max_windows=args.max_train_windows,
    )
    val_dataset = WindowedCurveDataset(
        Path(args.cache),
        args.eval_split,
        window=args.window,
        stride=args.eval_stride,
        feature_mode=args.feature_mode,
        max_windows=args.max_eval_windows,
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = build_sequence_model(variant, input_dim=train_dataset.feature_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    gauge_options = GaugeProjectionOptions(moment_mode=variant.moment_mode)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    epoch_summaries = []
    global_step = 0
    projection_seconds = 0.0
    start_all = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        losses = []
        for x, y in train_loader:
            global_step += 1
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = torch.mean((pred - y) ** 2)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            if variant.gauge_every_steps > 0 and global_step % variant.gauge_every_steps == 0:
                proj_start = time.perf_counter()
                gauge_project_model(model, optimizer, gauge_options)
                projection_seconds += time.perf_counter() - proj_start
            losses.append(float(loss.detach().cpu().item()))
        val_metrics = evaluate_teacher_forced(model, val_loader, device)
        rollout_metrics = {}
        if args.rollout_eval:
            cache_data = np.load(args.cache)
            rollout_metrics = evaluate_autoregressive_rollout(
                model,
                val_loader,
                device,
                previous_current_lags=current_lags_for_mode(args.feature_mode),
                x_mean_prev_current=float(cache_data["x_mean"][0, 2]),
                x_std_prev_current=float(cache_data["x_std"][0, 2]),
                y_mean=float(cache_data["y_mean"][0, 0]),
                y_std=float(cache_data["y_std"][0, 0]),
            )
        epoch_seconds = time.perf_counter() - epoch_start
        epoch_summaries.append(
            {
                "epoch": epoch,
                "train_loss_mean": float(np.mean(losses)),
                "train_loss_last": float(losses[-1]),
                "epoch_seconds": epoch_seconds,
                "teacher_forced_val_metrics": val_metrics,
                "rollout_val_metrics": rollout_metrics,
                "val_metrics": val_metrics,
            }
        )

    total_seconds = time.perf_counter() - start_all
    checkpoint_path = None
    if args.save_models:
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"{variant.name}_{args.feature_mode}_seed{seed}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "variant": asdict(variant),
                "seed": seed,
                "feature_mode": args.feature_mode,
                "feature_names": feature_names_for_mode(args.feature_mode),
                "window": args.window,
                "model_input_dim": train_dataset.feature_dim,
                "complexity": model_complexity(model),
            },
            checkpoint_path,
        )
    return {
        "variant": variant.name,
        "seed": seed,
        "variant_config": asdict(variant),
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "feature_mode": args.feature_mode,
        "feature_names": feature_names_for_mode(args.feature_mode),
        "model_input_dim": train_dataset.feature_dim,
        "rollout_eval": args.rollout_eval,
        "window": args.window,
        "stride": args.stride,
        "eval_stride": args.eval_stride,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "train_windows": len(train_dataset),
        "eval_windows": len(val_dataset),
        "global_steps": global_step,
        "complexity": model_complexity(model),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "epoch_summaries": epoch_summaries,
        "final_val_metrics": epoch_summaries[-1]["val_metrics"],
        "final_teacher_forced_val_metrics": epoch_summaries[-1]["teacher_forced_val_metrics"],
        "final_rollout_val_metrics": epoch_summaries[-1]["rollout_val_metrics"],
        "timing": {
            "wall_clock_seconds": total_seconds,
            "seconds_per_epoch": total_seconds / max(1, args.epochs),
            "seconds_per_step": total_seconds / max(1, global_step),
            "projection_wall_clock_seconds": projection_seconds,
            "projection_seconds_per_event": projection_seconds / max(1, global_step)
            if variant.gauge_every_steps > 0
            else 0.0,
            "projection_wall_clock_fraction": projection_seconds / total_seconds if total_seconds > 0 else 0.0,
            "peak_cuda_memory_mb": current_memory_mb(device),
        },
    }


def aggregate(results: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in results:
        grouped.setdefault(str(row["variant"]), []).append(row)
    summary = {}
    for variant, rows in grouped.items():
        rmse = np.asarray([row["final_teacher_forced_val_metrics"]["rmse"] for row in rows], dtype=float)
        r2 = np.asarray([row["final_teacher_forced_val_metrics"]["r2"] for row in rows], dtype=float)
        seconds = np.asarray([row["timing"]["wall_clock_seconds"] for row in rows], dtype=float)
        row_summary = {
            "runs": len(rows),
            "teacher_forced_rmse_mean": float(rmse.mean()),
            "teacher_forced_rmse_std": float(rmse.std(ddof=0)),
            "teacher_forced_r2_mean": float(r2.mean()),
            "teacher_forced_r2_std": float(r2.std(ddof=0)),
            "wall_clock_seconds_mean": float(seconds.mean()),
            "seconds_per_step_mean": float(np.mean([row["timing"]["seconds_per_step"] for row in rows])),
            "peak_cuda_memory_mb_max": float(max(row["timing"]["peak_cuda_memory_mb"] for row in rows)),
        }
        rollout_rows = [row for row in rows if row.get("final_rollout_val_metrics")]
        if rollout_rows:
            rollout_rmse = np.asarray([row["final_rollout_val_metrics"]["rollout_rmse"] for row in rollout_rows])
            rollout_r2 = np.asarray([row["final_rollout_val_metrics"]["rollout_r2"] for row in rollout_rows])
            rollout_final = np.asarray(
                [row["final_rollout_val_metrics"]["rollout_final_state_mae"] for row in rollout_rows]
            )
            row_summary.update(
                {
                    "rollout_rmse_mean": float(rollout_rmse.mean()),
                    "rollout_rmse_std": float(rollout_rmse.std(ddof=0)),
                    "rollout_r2_mean": float(rollout_r2.mean()),
                    "rollout_r2_std": float(rollout_r2.std(ddof=0)),
                    "rollout_final_state_mae_mean": float(rollout_final.mean()),
                    "rollout_final_state_mae_std": float(rollout_final.std(ddof=0)),
                }
            )
        summary[variant] = row_summary
    return summary


def write_payload(path: Path, results: list[dict[str, object]]) -> None:
    payload = {
        "note": "Track B/C sequence run. Timing is recorded for every variant and seed.",
        "results": results,
        "summary": aggregate(results),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="data/processed/sdc_state_splits.npz")
    parser.add_argument("--out", default="results/sdc_sequence_kaggle.json")
    parser.add_argument("--variants", nargs="+", default=["lstm", "gru", "kan_lstm", "gauge_kan_lstm"], choices=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--eval-stride", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--feature-mode",
        default="no_current",
        choices=[*FEATURE_MODES.keys(), "history2", "history3"],
    )
    parser.add_argument("--rollout-eval", action="store_true")
    parser.add_argument("--eval-split", default="val", choices=["val", "test"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--max-eval-windows", type=int, default=0)
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--checkpoint-dir", default="checkpoints/sequence")
    args = parser.parse_args()
    if args.max_train_windows <= 0:
        args.max_train_windows = None
    if args.max_eval_windows <= 0:
        args.max_eval_windows = None

    out = Path(args.out)
    results = []
    for seed in args.seeds:
        for variant_name in args.variants:
            print(f"Running {variant_name} seed={seed}", flush=True)
            results.append(run_one(args, VARIANTS[variant_name], seed))
            write_payload(out, results)
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
