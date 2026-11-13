"""Reviewer-revision audit and inference-only analyses for the SDC study.

This script intentionally does not train any model. It recovers the archived
split and raw-file identities, audits the no-current features, compares reset
chunk and continuous-state inference, exports hidden-state diagnostics, and
creates the reviewer-facing figures/CSVs used by REVISION_REQUIRED_VALUES.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from memristorkan.data.sdc import load_sdc_state_curves  # noqa: E402
from memristorkan.models.recurrent import ClippedKANGRURegressor, GRURegressor  # noqa: E402
from run_sdc_sequence import build_feature_array, build_sequence_model  # noqa: E402
from evaluate_sequence_checkpoints import load_checkpoint, variant_from_checkpoint  # noqa: E402


FEATURE_AUDIT = [
    {
        "feature_name": "V_t",
        "formula_source": (
            "Device voltage at the present sample: Volt.1_t - Volt_t. "
            "The source measurement circuit defines Volt_t as the voltage across the 100 kOhm series resistor, "
            "so I_t = Volt_t / 100 kOhm."
        ),
        "uses_present_voltage": True,
        "uses_previous_voltage": False,
        "uses_voltage_difference": False,
        "uses_time": False,
        "uses_measured_current": True,
        "uses_previous_measured_current": False,
        "notes": (
            "Selected from base feature column 0. This is not an independent applied-voltage channel: "
            "the same-time current-sense voltage is subtracted from total circuit voltage."
        ),
    },
    {
        "feature_name": "V_t_minus_1",
        "formula_source": "V_(t-1) = Volt.1_(t-1) - Volt_(t-1)",
        "uses_present_voltage": False,
        "uses_previous_voltage": True,
        "uses_voltage_difference": False,
        "uses_time": False,
        "uses_measured_current": False,
        "uses_previous_measured_current": True,
        "notes": (
            "Selected from base feature column 1; reset independently for every raw curve. "
            "It contains the preceding series-resistor voltage and therefore previous measured current indirectly."
        ),
    },
    {
        "feature_name": "dV_t",
        "formula_source": (
            "V_t - V_(t-1) = [Volt.1_t - Volt_t] - [Volt.1_(t-1) - Volt_(t-1)]"
        ),
        "uses_present_voltage": True,
        "uses_previous_voltage": True,
        "uses_voltage_difference": True,
        "uses_time": False,
        "uses_measured_current": True,
        "uses_previous_measured_current": True,
        "notes": "Contains the change in series-resistor voltage; not divided by elapsed time",
    },
    {
        "feature_name": "sweep_direction",
        "formula_source": "sign(dV_t)",
        "uses_present_voltage": True,
        "uses_previous_voltage": True,
        "uses_voltage_difference": True,
        "uses_time": False,
        "uses_measured_current": True,
        "uses_previous_measured_current": True,
        "notes": (
            "sign(dV_t), where dV_t contains present and previous series-resistor voltage. "
            "Raw encoding is -1, 0, or +1 before train-only z-score scaling."
        ),
    },
]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def metric_values(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    residual = pred - target
    mse = float(np.mean(residual**2))
    denom = float(np.sum((target - target.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / denom if denom > 0 else float("nan")
    return {
        "r2": r2,
        "rmse": math.sqrt(mse),
        "mae": float(np.mean(np.abs(residual))),
        "number_of_samples": int(target.size),
    }


def load_model(checkpoint: Path, input_dim: int, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = load_checkpoint(checkpoint, device)
    variant = variant_from_checkpoint(payload)
    model = build_sequence_model(variant, input_dim=input_dim).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


@torch.no_grad()
def recurrent_trace(
    model: torch.nn.Module,
    x: np.ndarray,
    *,
    device: torch.device,
    reset_window: int | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return predictions and hidden states for one complete curve.

    ``reset_window=None`` means one initial reset for the complete trajectory.
    Otherwise state is reset at every contiguous chunk boundary, including the
    final short chunk, so every sample is evaluated exactly once.
    """

    xt = torch.from_numpy(x.astype(np.float32)).to(device)
    preds: list[np.ndarray] = []
    states: list[np.ndarray] = []
    state: torch.Tensor | None = None
    reset_count = 0
    for t in range(xt.shape[0]):
        if t == 0 or (reset_window is not None and t % reset_window == 0):
            state = None
            reset_count += 1
        x_t = xt[t : t + 1]
        if isinstance(model, ClippedKANGRURegressor):
            h = x_t.new_zeros(1, model.hidden_dim) if state is None else state
            h = model.cell(x_t, h)
            y_t = model.head(h)
            state = h
        elif isinstance(model, GRURegressor):
            out, h = model.rnn(x_t.unsqueeze(1), None if state is None else state.unsqueeze(0))
            y_t = model.head(out[:, -1, :])
            state = h.squeeze(0)
        else:
            raise TypeError(f"unsupported recurrent model: {type(model).__name__}")
        preds.append(y_t.squeeze(0).cpu().numpy())
        states.append(state.squeeze(0).cpu().numpy())
    return np.asarray(preds, dtype=np.float32).reshape(-1), np.stack(states).astype(np.float32), reset_count


@torch.no_grad()
def reset_chunk_predictions(
    model: torch.nn.Module,
    x: np.ndarray,
    *,
    device: torch.device,
    window: int,
) -> tuple[np.ndarray, int]:
    """Evaluate contiguous independent chunks while batching equal-length chunks."""

    length = int(x.shape[0])
    full_chunks = length // window
    predictions: list[np.ndarray] = []
    if full_chunks:
        full = x[: full_chunks * window].reshape(full_chunks, window, x.shape[-1])
        # Limit batch size to avoid excessive temporary KAN tensors on GPU.
        for start in range(0, full_chunks, 32):
            batch = torch.from_numpy(full[start : start + 32].astype(np.float32)).to(device)
            predictions.append(model(batch).squeeze(-1).cpu().numpy().reshape(-1))
    remainder = length - full_chunks * window
    if remainder:
        tail = torch.from_numpy(x[full_chunks * window :][None].astype(np.float32)).to(device)
        predictions.append(model(tail).squeeze(0).squeeze(-1).cpu().numpy())
    return np.concatenate(predictions).astype(np.float32), full_chunks + int(remainder > 0)


def infer_source_fields(path: Path, raw_root: Path) -> dict[str, Any]:
    rel = path.relative_to(raw_root).as_posix()
    parts = rel.split("/")
    day_group = next((part for part in parts if re.fullmatch(r"data_\d+", part)), "")
    batch = next((part for part in parts if part.startswith("batch")), "")
    condition_dir = next((part for part in parts if re.fullmatch(r"\d+m\d+m", part)), "")
    stem = path.stem
    repeat_match = re.search(r"(\d+)$", stem)
    repeat_token = repeat_match.group(1) if repeat_match else ""
    amp_match = re.search(r"triangle(\d+)m(\d+)m", stem)
    if amp_match:
        condition = f"triangle-wave filename condition {amp_match.group(1)}m/{amp_match.group(2)}m; {batch}"
    elif "400m" in stem:
        condition = "triangle-wave 400m, filename states 1 kHz"
    else:
        condition = "triangle-wave; exact condition not encoded"
    return {
        "source_filename": rel,
        "day_group": day_group,
        "batch": batch,
        "condition_dir": condition_dir,
        "cycle_or_repeat_token": repeat_token,
        "measurement_condition": condition,
        "device_id": "not encoded in archived metadata",
    }


def read_time_metadata(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path, header=1)
    if "second" not in df.columns:
        return {"timestamp_available": False}
    t = pd.to_numeric(df["second"], errors="coerce").dropna().to_numpy(dtype=np.float64)
    dt = np.diff(t)
    mean_dt = float(dt.mean())
    return {
        "timestamp_available": True,
        "start_second": float(t[0]),
        "end_second": float(t[-1]),
        "duration_second": float(t[-1] - t[0]),
        "mean_dt_second": mean_dt,
        "std_dt_second": float(dt.std()),
        "min_dt_second": float(dt.min()),
        "max_dt_second": float(dt.max()),
        "sampling_frequency_hz": 1.0 / mean_dt,
        "relative_dt_std": float(dt.std() / abs(mean_dt)),
        "sample_count_raw": int(t.size),
    }


def estimate_cycles(voltage: np.ndarray) -> dict[str, Any]:
    """Estimate period from positive-going crossings at the midrange voltage."""

    v = np.asarray(voltage, dtype=np.float64)
    midpoint = 0.5 * (float(v.min()) + float(v.max()))
    crossing = np.flatnonzero((v[:-1] < midpoint) & (v[1:] >= midpoint)) + 1
    periods = np.diff(crossing)
    if periods.size == 0:
        return {"crossings": int(crossing.size), "complete_cycles": 0, "samples_per_cycle": float("nan")}
    period = float(np.median(periods))
    return {
        "crossings": int(crossing.size),
        "complete_cycles": int(periods.size),
        "samples_per_cycle": period,
        "period_std_samples": float(periods.std()),
        "window_256_cycles": 256.0 / period,
    }


def save_figure(fig: plt.Figure, stem: Path) -> dict[str, str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png")
    svg = stem.with_suffix(".svg")
    fig.savefig(png, dpi=350, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png), "svg": str(svg)}


def make_context_figure(voltage: np.ndarray, current: np.ndarray, start: int, out_dir: Path) -> dict[str, str]:
    end = min(start + 256, len(voltage))
    idx = np.arange(len(voltage))
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.6), constrained_layout=True)
    axes[0].plot(idx, voltage, color="#1f4e79", lw=0.65)
    axes[0].axvspan(start, end - 1, color="#e69f00", alpha=0.28)
    axes[0].set_ylabel("Reconstructed device voltage\n(source units)")
    axes[0].set_title("(a) Full reconstructed device-voltage trajectory")
    axes[1].plot(idx, current, color="#262626", lw=0.65)
    axes[1].axvspan(start, end - 1, color="#e69f00", alpha=0.28)
    axes[1].set_ylabel("Measured current\n(source units)")
    axes[1].set_title("(b) Full measured-current trajectory")
    axes[2].plot(voltage, current, color="#808080", lw=0.65, alpha=0.75, label="Complete trajectory")
    axes[2].plot(voltage[start:end], current[start:end], color="#d55e00", lw=1.3, label="Highlighted 256 samples")
    axes[2].set_xlabel("Reconstructed device voltage (source units)")
    axes[2].set_ylabel("Measured current (source units)")
    axes[2].set_title("(c) Measured I-V trajectory")
    axes[2].legend(frameon=False, loc="best")
    for ax in axes[:2]:
        ax.set_xlabel("Sample index")
    return save_figure(fig, out_dir / "figure_01_full_trajectory_window_context")


def make_prediction_figure(
    voltage: np.ndarray,
    target: np.ndarray,
    pred_kan: np.ndarray,
    pred_gru: np.ndarray,
    out_dir: Path,
) -> dict[str, str]:
    idx = np.arange(len(target))
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.6), constrained_layout=True)
    axes[0].plot(voltage, target, color="black", lw=0.8, label="Measured")
    axes[0].plot(voltage, pred_kan, color="#0072b2", lw=0.7, ls="--", label="Bounded KAN-GRU")
    axes[0].plot(voltage, pred_gru, color="#d55e00", lw=0.65, ls=":", label="GRU")
    axes[0].set(xlabel="Voltage (source units)", ylabel="Current (source units)", title="(a) Continuous-state full-trajectory I-V prediction")
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    span = min(2400, len(target))
    axes[1].plot(idx[:span], target[:span], color="black", lw=0.8, label="Measured")
    axes[1].plot(idx[:span], pred_kan[:span], color="#0072b2", lw=0.7, ls="--", label="Bounded KAN-GRU")
    axes[1].plot(idx[:span], pred_gru[:span], color="#d55e00", lw=0.65, ls=":", label="GRU")
    axes[1].set(xlabel="Sample index", ylabel="Current (source units)", title=f"(b) First {span} samples")
    axes[2].plot(idx, pred_kan - target, color="#0072b2", lw=0.55, label="KAN-GRU residual")
    axes[2].plot(idx, pred_gru - target, color="#d55e00", lw=0.5, alpha=0.85, label="GRU residual")
    axes[2].axhline(0, color="black", lw=0.5)
    axes[2].set(xlabel="Sample index", ylabel="Prediction - measured", title="(c) Residual over the complete trajectory")
    axes[2].legend(frameon=False, ncol=2, fontsize=8)
    return save_figure(fig, out_dir / "figure_03_continuous_full_trajectory_prediction")


def make_hidden_norm_figure(hidden_rows: list[dict[str, Any]], out_dir: Path) -> dict[str, str]:
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.2), constrained_layout=True)
    for row in hidden_rows:
        arr = np.load(row["trace_path"])
        norm = np.linalg.norm(arr, axis=1)
        label = f"test {row['local_test_index']}" if row["seed"] == 0 else None
        axes[0].plot(norm, color="#0072b2", alpha=0.25 if row["seed"] else 0.45, lw=0.5, label=label)
    rep = next(row for row in hidden_rows if row["model"] == "clipped_kan_gru_h32" and row["seed"] == 0 and row["local_test_index"] == 0)
    rep_arr = np.load(rep["trace_path"])
    axes[1].plot(np.linalg.norm(rep_arr, axis=1), color="#0072b2", lw=0.75)
    axes[0].set(title="(a) Bounded KAN-GRU hidden-state norms across held-out trajectories/seeds", ylabel=r"$||h_t||_2$")
    axes[1].set(title="(b) Representative test trajectory 0, seed 0", xlabel="Sample index", ylabel=r"$||h_t||_2$")
    return save_figure(fig, out_dir / "figure_04_hidden_state_norm_stability")


def make_combined_figure(
    voltage: np.ndarray,
    target: np.ndarray,
    pred: np.ndarray,
    hidden: np.ndarray,
    marks: list[int],
    out_dir: Path,
) -> tuple[dict[str, str], dict[str, float]]:
    centered = hidden - hidden.mean(axis=0, keepdims=True)
    _u, s, vt = np.linalg.svd(centered, full_matrices=False)
    scores = centered @ vt[:2].T
    variance = s**2
    ratios = variance / variance.sum()
    labels = ["A", "B", "C", "D"]
    colors = ["#d55e00", "#009e73", "#cc79a7", "#e69f00"]
    idx = np.arange(len(target))
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.2), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(voltage, target, color="black", lw=0.75, label="Measured")
    ax.plot(voltage, pred, color="#0072b2", lw=0.65, ls="--", label="Bounded KAN-GRU")
    for label, color, mark in zip(labels, colors, marks, strict=True):
        ax.scatter(voltage[mark], target[mark], color=color, s=28, zorder=5)
        ax.annotate(label, (voltage[mark], target[mark]), xytext=(4, 4), textcoords="offset points")
    ax.set(xlabel="Voltage", ylabel="Current", title="(a) Measured and predicted I-V loop")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[0, 1]
    ax.plot(idx, pred - target, color="#0072b2", lw=0.55)
    for label, color, mark in zip(labels, colors, marks, strict=True):
        ax.scatter(mark, pred[mark] - target[mark], color=color, s=28, zorder=5)
        ax.annotate(label, (mark, pred[mark] - target[mark]), xytext=(4, 4), textcoords="offset points")
    ax.axhline(0, color="black", lw=0.5)
    ax.set(xlabel="Sample index", ylabel="Residual", title="(b) Prediction residual")
    ax = axes[1, 0]
    ax.plot(idx, scores[:, 0], color="#0072b2", lw=0.55, label="PC1")
    ax.plot(idx, scores[:, 1], color="#d55e00", lw=0.55, label="PC2")
    for label, color, mark in zip(labels, colors, marks, strict=True):
        ax.scatter([mark, mark], scores[mark, :2], color=color, s=22, zorder=5)
    ax.set(xlabel="Sample index", ylabel="PCA score", title="(c) Learned recurrent-state PCs")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1, 1]
    ax.plot(scores[:, 0], scores[:, 1], color="#6f6f6f", lw=0.55)
    for label, color, mark in zip(labels, colors, marks, strict=True):
        ax.scatter(scores[mark, 0], scores[mark, 1], color=color, s=30, zorder=5)
        ax.annotate(label, (scores[mark, 0], scores[mark, 1]), xytext=(4, 4), textcoords="offset points")
    ax.set(xlabel="PC1", ylabel="PC2", title="(d) Internal recurrent representation")
    paths = save_figure(fig, out_dir / "figure_05_combined_prediction_hidden_pca")
    pca = {
        "pc1_explained_variance_ratio": float(ratios[0]),
        "pc2_explained_variance_ratio": float(ratios[1]),
        "pc1_pc2_cumulative_ratio": float(ratios[:2].sum()),
    }
    return paths, pca


def aggregate_rows(rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
    subset = [r for r in rows if r["model"] == model]
    reset_r2 = np.asarray([r["reset_r2"] for r in subset], dtype=float)
    cont_r2 = np.asarray([r["continuous_r2"] for r in subset], dtype=float)
    reset_rmse = np.asarray([r["reset_rmse"] for r in subset], dtype=float)
    cont_rmse = np.asarray([r["continuous_rmse"] for r in subset], dtype=float)
    delta_r2 = cont_r2 - reset_r2
    delta_rmse = cont_rmse - reset_rmse
    return {
        "model": model,
        "runs_times_curves": len(subset),
        "reset_r2_mean": float(reset_r2.mean()),
        "continuous_r2_mean": float(cont_r2.mean()),
        "delta_r2_mean": float(delta_r2.mean()),
        "delta_r2_median": float(np.median(delta_r2)),
        "reset_rmse_mean": float(reset_rmse.mean()),
        "continuous_rmse_mean": float(cont_rmse.mean()),
        "delta_rmse_mean": float(delta_rmse.mean()),
        "trajectories_runs_improved": int((delta_r2 > 1e-12).sum()),
        "trajectories_runs_degraded": int((delta_r2 < -1e-12).sum()),
        "trajectories_runs_unchanged": int((np.abs(delta_r2) <= 1e-12).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("data/processed/sdc_state_splits.npz"))
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/sdc_state_metadata.json"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/sdc/memristor-state-characterisation"))
    parser.add_argument("--out-dir", type=Path, default=Path("revision_experiments"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reset-window", type=int, default=256)
    parser.add_argument("--kan-checkpoints", nargs="+", type=Path, default=[
        Path("results/kaggle_imported/kangru_sdc_100epoch_seed0/checkpoints/sequence/clipped_kan_gru_h32_no_current_seed0.pt"),
        Path("results/kaggle_imported/clipped_kangru_sdc_100epoch_seed1_2/checkpoints/sequence/clipped_kan_gru_h32_no_current_seed1.pt"),
        Path("results/kaggle_imported/clipped_kangru_sdc_100epoch_seed1_2/checkpoints/sequence/clipped_kan_gru_h32_no_current_seed2.pt"),
    ])
    parser.add_argument("--gru-checkpoints", nargs="+", type=Path, default=[
        Path("checkpoints/kaggle_imported/no_current_baselines_100epoch_3seed/sequence/gru_h116_no_current_seed0.pt"),
        Path("checkpoints/kaggle_imported/no_current_baselines_100epoch_3seed/sequence/gru_h116_no_current_seed1.pt"),
        Path("checkpoints/kaggle_imported/no_current_baselines_100epoch_3seed/sequence/gru_h116_no_current_seed2.pt"),
    ])
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    figure_dir = out_dir / "figures"
    hidden_dir = out_dir / "hidden_states"
    out_dir.mkdir(parents=True, exist_ok=True)
    hidden_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    cache = np.load(args.cache)
    curves, _summary = load_sdc_state_curves(args.raw_root)
    raw_state_root = (args.raw_root / "data" / "state_data").resolve()

    # Feature audit.
    write_csv(out_dir / "no_current_feature_audit.csv", FEATURE_AUDIT)
    no_current_verified = not any(
        row["uses_measured_current"] or row["uses_previous_measured_current"] for row in FEATURE_AUDIT
    )

    # Exact split manifest in archived ordering.
    manifest_rows: list[dict[str, Any]] = []
    test_identity: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        global_indices = metadata["split_curve_indices"][split]
        lengths = cache[f"{split}_curve_lengths"].astype(int)
        for local_idx, (global_idx, length) in enumerate(zip(global_indices, lengths, strict=True)):
            path = Path(str(curves[int(global_idx)]["path"])).resolve()
            source = infer_source_fields(path, raw_state_root)
            row = {
                "dataset": "SDC state-characterisation",
                "trajectory_id": int(global_idx),
                "source_filename": source["source_filename"],
                "device_id": source["device_id"],
                "cycle_id_if_available": source["cycle_or_repeat_token"],
                "split": split,
                "local_split_index": local_idx,
                "number_of_samples": int(length),
                "measurement_condition": source["measurement_condition"],
                "day_group": source["day_group"],
                "batch": source["batch"],
                "split_seed": int(metadata["seed"]),
            }
            manifest_rows.append(row)
            if split == "test":
                test_identity.append(row)
    write_csv(out_dir / "sdc_split_manifest.csv", manifest_rows)

    # Time and physical-window context from test trajectory 0.
    test0 = test_identity[0]
    test0_path = raw_state_root / test0["source_filename"]
    time_metadata = read_time_metadata(test0_path)
    n0 = int(cache["test_curve_lengths"][0])
    voltage0 = cache["test_curve_x_raw"][0, :n0, 0].astype(np.float64)
    current0 = cache["test_curve_y_raw"][0, :n0, 0].astype(np.float64)
    cycle_metadata = estimate_cycles(voltage0)
    window_context = {
        "local_test_index": 0,
        "trajectory_id": int(test0["trajectory_id"]),
        "source_filename": test0["source_filename"],
        "number_of_samples": n0,
        "window_length": 256,
        "window_fraction": 256.0 / n0,
        "window_percentage": 100.0 * 256.0 / n0,
        "nonoverlapping_full_windows": n0 // 256,
        "remainder_samples": n0 % 256,
        **cycle_metadata,
        **time_metadata,
    }
    context_paths = make_context_figure(voltage0, current0, 0, figure_dir)
    window_context["figure_paths"] = context_paths

    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "device_used": str(device),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
    }

    if not no_current_verified:
        audit = {
            "status": "BLOCKED_BY_STRICT_NO_CURRENT_AUDIT",
            "environment": environment,
            "no_current_verified": False,
            "stop_rule_applied": True,
            "stop_reason": (
                "The target/current-sense channel Volt is algebraically embedded in every nominal "
                "no-current input through V = Volt.1 - Volt. The revision brief requires stopping "
                "when any directly or indirectly current-derived input is found."
            ),
            "measurement_equations": {
                "series_resistor_ohm": 100000.0,
                "current": "I_t = Volt_t / 100000 ohm",
                "device_voltage": "V_device,t = Volt.1_t - Volt_t",
                "target_used_by_project": "y_t = Volt_t (current-sense voltage proportional to I_t)",
            },
            "source_evidence": [
                {
                    "path": "data/raw/sdc/memristor-state-characterisation/memristor_state_characterisation/functions.py",
                    "lines": "23-24",
                    "evidence": "voltage = Volt.1 - Volt; current = Volt",
                },
                {
                    "path": "data/raw/sdc/memristor-state-characterisation/plot_resistances.py",
                    "lines": "30, 69-70",
                    "evidence": "R = voltage/current * series_resistance; axes identify voltage and current",
                },
                {
                    "path": "src/memristorkan/data/sdc.py",
                    "lines": "55-56, 80-84, 146-150",
                    "evidence": "project loader reproduces the same subtraction and uses Volt as the target",
                },
            ],
            "feature_audit": FEATURE_AUDIT,
            "split_counts": metadata["split_curve_counts"],
            "split_seed": metadata["seed"],
            "split_curve_indices": metadata["split_curve_indices"],
            "test_identity": test_identity,
            "identity_0_4_8": [test_identity[i] for i in (0, 4, 8)],
            "identity_conclusion": (
                "All three files are in the data_04 acquisition-day group, but the archived repository metadata "
                "does not encode a verified physical device identifier; therefore same-device versus different-device "
                "origin cannot be established from the archive."
            ),
            "window_context": window_context,
            "time_metadata": time_metadata,
            "experiments_not_run": [
                "window-length sensitivity retraining",
                "complete-trajectory no-reset inference",
                "sequence-position stability",
                "hidden-state stability and PCA",
            ],
            "figure_paths": {"window_context": context_paths},
        }
        (out_dir / "audit_and_inference_summary.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
        print(json.dumps({
            "out_dir": str(out_dir),
            "status": audit["status"],
            "no_current_verified": False,
            "message": audit["stop_reason"],
        }, indent=2))
        return

    x_all = build_feature_array(cache["test_curve_x"].astype(np.float32), cache["test_curve_lengths"], "no_current")
    y_all = cache["test_curve_y"].astype(np.float32)
    y_raw_all = cache["test_curve_y_raw"].astype(np.float32)
    x_raw_all = cache["test_curve_x_raw"].astype(np.float32)
    lengths = cache["test_curve_lengths"].astype(int)
    y_mean = float(cache["y_mean"].reshape(-1)[0])
    y_std = float(cache["y_std"].reshape(-1)[0])

    checkpoint_sets = {
        "clipped_kan_gru_h32": args.kan_checkpoints,
        "gru_h116": args.gru_checkpoints,
    }
    full_rows: list[dict[str, Any]] = []
    quarter_rows: list[dict[str, Any]] = []
    hidden_summary: list[dict[str, Any]] = []
    hidden_trace_rows: list[dict[str, Any]] = []
    representative: dict[str, Any] = {}

    for model_name, checkpoints in checkpoint_sets.items():
        for checkpoint in checkpoints:
            model, payload = load_model(checkpoint, input_dim=4, device=device)
            seed = int(payload.get("seed", re.search(r"seed(\d+)", checkpoint.name).group(1)))
            for local_idx, length in enumerate(lengths):
                length = int(length)
                x = x_all[local_idx, :length]
                target = y_all[local_idx, :length, 0]
                pred_reset, reset_count = reset_chunk_predictions(
                    model, x, device=device, window=args.reset_window
                )
                pred_cont, hidden_cont, continuous_reset_count = recurrent_trace(
                    model, x, device=device, reset_window=None
                )
                reset_metrics = metric_values(pred_reset, target)
                cont_metrics = metric_values(pred_cont, target)
                identity = test_identity[local_idx]
                full_rows.append({
                    "model": model_name,
                    "seed": seed,
                    "local_test_index": local_idx,
                    "trajectory_id": identity["trajectory_id"],
                    "source_filename": identity["source_filename"],
                    "number_of_samples": length,
                    "reset_window": args.reset_window,
                    "reset_count": reset_count,
                    "continuous_reset_count": continuous_reset_count,
                    "reset_r2": reset_metrics["r2"],
                    "continuous_r2": cont_metrics["r2"],
                    "delta_r2": cont_metrics["r2"] - reset_metrics["r2"],
                    "reset_rmse": reset_metrics["rmse"],
                    "continuous_rmse": cont_metrics["rmse"],
                    "delta_rmse": cont_metrics["rmse"] - reset_metrics["rmse"],
                    "reset_mae": reset_metrics["mae"],
                    "continuous_mae": cont_metrics["mae"],
                })
                boundaries = np.linspace(0, length, 5, dtype=int)
                for quarter in range(4):
                    lo, hi = int(boundaries[quarter]), int(boundaries[quarter + 1])
                    q_metrics = metric_values(pred_cont[lo:hi], target[lo:hi])
                    quarter_rows.append({
                        "model": model_name,
                        "seed": seed,
                        "local_test_index": local_idx,
                        "trajectory_id": identity["trajectory_id"],
                        "quarter": quarter + 1,
                        "start_sample": lo,
                        "end_sample_exclusive": hi,
                        **q_metrics,
                    })
                norm = np.linalg.norm(hidden_cont.astype(np.float64), axis=1)
                near = min(100, length - 1)
                finite = np.isfinite(hidden_cont)
                hidden_path = hidden_dir / f"{model_name}_seed{seed}_test{local_idx:02d}.npy"
                np.save(hidden_path, hidden_cont)
                hidden_summary.append({
                    "model": model_name,
                    "seed": seed,
                    "local_test_index": local_idx,
                    "trajectory_id": identity["trajectory_id"],
                    "number_of_samples": length,
                    "hidden_dim": int(hidden_cont.shape[1]),
                    "norm_min": float(norm.min()),
                    "norm_max": float(norm.max()),
                    "norm_mean": float(norm.mean()),
                    "norm_std": float(norm.std()),
                    "norm_beginning": float(norm[near]),
                    "norm_middle": float(norm[length // 2]),
                    "norm_end": float(norm[-1]),
                    "activation_min": float(np.nanmin(hidden_cont)),
                    "activation_max": float(np.nanmax(hidden_cont)),
                    "activation_mean": float(np.nanmean(hidden_cont)),
                    "activation_std": float(np.nanstd(hidden_cont)),
                    "nan_count": int(np.isnan(hidden_cont).sum()),
                    "inf_count": int(np.isinf(hidden_cont).sum()),
                    "trace_path": str(hidden_path),
                })
                hidden_trace_rows.append({
                    "model": model_name,
                    "seed": seed,
                    "local_test_index": local_idx,
                    "trace_path": str(hidden_path),
                })
                if seed == 0 and local_idx == 0:
                    representative[model_name] = {
                        "prediction_norm": pred_cont,
                        "hidden": hidden_cont,
                    }

    write_csv(out_dir / "full_trajectory_per_trajectory.csv", full_rows)
    full_summary = [aggregate_rows(full_rows, name) for name in checkpoint_sets]
    write_csv(out_dir / "full_trajectory_summary.csv", full_summary)
    write_csv(out_dir / "sequence_position_error.csv", quarter_rows)
    write_csv(out_dir / "hidden_state_summary.csv", hidden_summary)

    quarter_summary: list[dict[str, Any]] = []
    for model_name in checkpoint_sets:
        model_rows = [r for r in quarter_rows if r["model"] == model_name]
        quarter_means = {
            q: float(np.mean([r["rmse"] for r in model_rows if r["quarter"] == q])) for q in range(1, 5)
        }
        by_run_curve: dict[tuple[int, int], dict[int, float]] = {}
        for row in model_rows:
            by_run_curve.setdefault((row["seed"], row["local_test_index"]), {})[row["quarter"]] = row["rmse"]
        differences = np.asarray([v[4] - v[1] for v in by_run_curve.values()], dtype=float)
        ratios = np.asarray([v[4] / v[1] for v in by_run_curve.values()], dtype=float)
        quarter_summary.append({
            "model": model_name,
            "q1_rmse_mean": quarter_means[1],
            "q2_rmse_mean": quarter_means[2],
            "q3_rmse_mean": quarter_means[3],
            "q4_rmse_mean": quarter_means[4],
            "q4_minus_q1_mean": float(differences.mean()),
            "q4_div_q1_mean": float(ratios.mean()),
            "run_curve_count": len(by_run_curve),
        })
    write_csv(out_dir / "sequence_position_summary.csv", quarter_summary)

    pred_kan = representative["clipped_kan_gru_h32"]["prediction_norm"] * y_std + y_mean
    pred_gru = representative["gru_h116"]["prediction_norm"] * y_std + y_mean
    hidden_kan = representative["clipped_kan_gru_h32"]["hidden"]
    prediction_paths = make_prediction_figure(voltage0, current0, pred_kan, pred_gru, figure_dir)
    hidden_paths = make_hidden_norm_figure(
        [row for row in hidden_trace_rows if row["model"] == "clipped_kan_gru_h32"], figure_dir
    )
    marks = [0, n0 // 3, (2 * n0) // 3, n0 - 1]
    combined_paths, pca_metadata = make_combined_figure(
        voltage0, current0, pred_kan, hidden_kan, marks, figure_dir
    )
    pca_metadata.update({
        "trajectory_id": int(test0["trajectory_id"]),
        "local_test_index": 0,
        "seed": 0,
        "fit_data": "all 15,999 continuous-inference hidden-state rows for test trajectory 0, seed 0",
        "centering": "per-hidden-dimension mean centering",
        "scaling": "none",
        "algorithm": "SVD on centered 32-dimensional hidden states",
        "marked_indices": marks,
        "figure_paths": combined_paths,
    })

    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "device_used": str(device),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
    }
    audit = {
        "environment": environment,
        "no_current_verified": no_current_verified,
        "feature_audit": FEATURE_AUDIT,
        "split_counts": metadata["split_curve_counts"],
        "split_seed": metadata["seed"],
        "split_curve_indices": metadata["split_curve_indices"],
        "test_identity": test_identity,
        "identity_0_4_8": [test_identity[i] for i in (0, 4, 8)],
        "identity_conclusion": (
            "All three files are in the data_04 acquisition-day group, but the archived repository metadata "
            "does not encode a verified physical device identifier; therefore same-device versus different-device "
            "origin cannot be established from the archive."
        ),
        "window_context": window_context,
        "time_metadata": time_metadata,
        "full_trajectory_summary": full_summary,
        "sequence_position_summary": quarter_summary,
        "pca": pca_metadata,
        "figure_paths": {
            "window_context": context_paths,
            "continuous_prediction": prediction_paths,
            "hidden_norm": hidden_paths,
            "combined_pca": combined_paths,
        },
    }
    (out_dir / "audit_and_inference_summary.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({
        "out_dir": str(out_dir),
        "no_current_verified": no_current_verified,
        "full_trajectory_rows": len(full_rows),
        "hidden_summary_rows": len(hidden_summary),
        "figures": audit["figure_paths"],
    }, indent=2))


if __name__ == "__main__":
    main()
