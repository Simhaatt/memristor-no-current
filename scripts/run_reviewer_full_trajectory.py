"""Reviewer 2.8 full-trajectory inference for the original SDC formulation.

This script deliberately uses the submitted manuscript cache and archived
256-window checkpoints. It performs inference only: contiguous reset windows
versus one continuous recurrent state over each complete held-out trajectory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT / "src", ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from evaluate_sequence_checkpoints import load_checkpoint, variant_from_checkpoint  # noqa: E402
from memristorkan.data.sdc import load_sdc_state_curves  # noqa: E402
from memristorkan.models.recurrent import ClippedKANGRURegressor, GRURegressor  # noqa: E402
from run_sdc_sequence import build_feature_array, build_sequence_model  # noqa: E402


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    residual = prediction - target
    denominator = float(np.sum((target - target.mean()) ** 2))
    return {
        "r2": 1.0 - float(np.sum(residual**2)) / denominator if denominator > 0 else float("nan"),
        "rmse": math.sqrt(float(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
    }


def load_model(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = load_checkpoint(path, device)
    model = build_sequence_model(variant_from_checkpoint(payload), input_dim=4).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


@torch.no_grad()
def continuous_trace(
    model: torch.nn.Module, x: np.ndarray, device: torch.device
) -> tuple[np.ndarray, np.ndarray, int]:
    xt = torch.from_numpy(x.astype(np.float32)).to(device)
    if isinstance(model, GRURegressor):
        hidden_sequence, _final = model.rnn(xt.unsqueeze(0))
        prediction = model.head(hidden_sequence).squeeze(0).squeeze(-1)
        return prediction.cpu().numpy(), hidden_sequence.squeeze(0).cpu().numpy(), 1
    if not isinstance(model, ClippedKANGRURegressor):
        raise TypeError(type(model).__name__)
    state = xt.new_zeros(1, model.hidden_dim)
    predictions: list[np.ndarray] = []
    hidden: list[np.ndarray] = []
    for step in range(xt.shape[0]):
        state = model.cell(xt[step : step + 1], state)
        predictions.append(model.head(state).squeeze(0).cpu().numpy())
        hidden.append(state.squeeze(0).cpu().numpy())
    return np.asarray(predictions).reshape(-1), np.stack(hidden), 1


@torch.no_grad()
def reset_trace(
    model: torch.nn.Module, x: np.ndarray, window: int, device: torch.device
) -> tuple[np.ndarray, int]:
    length = len(x)
    chunks: list[np.ndarray] = []
    full_count = length // window
    if full_count:
        full = x[: full_count * window].reshape(full_count, window, x.shape[-1])
        for start in range(0, full_count, 32):
            batch = torch.from_numpy(full[start : start + 32].astype(np.float32)).to(device)
            chunks.append(model(batch).squeeze(-1).cpu().numpy().reshape(-1))
    remainder = length - full_count * window
    if remainder:
        batch = torch.from_numpy(x[full_count * window :][None].astype(np.float32)).to(device)
        chunks.append(model(batch).squeeze(0).squeeze(-1).cpu().numpy())
    reset_count = full_count + int(remainder > 0)
    prediction = np.concatenate(chunks)
    if prediction.shape[0] != length:
        raise AssertionError((prediction.shape, length))
    return prediction, reset_count


def save_figure(fig: plt.Figure, stem: Path) -> dict[str, str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png")
    svg = stem.with_suffix(".svg")
    fig.savefig(png, dpi=350, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png), "svg": str(svg)}


def representative_figure(
    voltage: np.ndarray,
    measured: np.ndarray,
    predictions: dict[str, np.ndarray],
    hidden: dict[str, np.ndarray],
    out_dir: Path,
) -> dict[str, str]:
    index = np.arange(len(measured))
    colors = {"clipped_kan_gru_h32": "#d55e00", "gru_h116": "#0072b2"}
    labels = {"clipped_kan_gru_h32": "Bounded KAN-GRU", "gru_h116": "GRU"}
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 9.0), constrained_layout=True, sharex=True)
    axes[0].plot(index, voltage, color="#2f4b7c", lw=0.7)
    axes[0].set_ylabel("Input voltage\n(source units)")
    axes[0].set_title("(a) Original manuscript input trajectory")
    axes[1].plot(index, measured, color="black", lw=0.85, label="Measured")
    for model, prediction in predictions.items():
        axes[1].plot(index, prediction, color=colors[model], lw=0.65, alpha=0.9, label=labels[model])
    axes[1].set_ylabel("Current-like target\n(source units)")
    axes[1].set_title("(b) Continuous full-trajectory prediction")
    axes[1].legend(frameon=False, ncol=3, loc="upper right")
    for model, prediction in predictions.items():
        axes[2].plot(index, prediction - measured, color=colors[model], lw=0.65, label=labels[model])
    axes[2].axhline(0, color="#777777", lw=0.6)
    axes[2].set_ylabel("Residual")
    axes[2].set_title("(c) Prediction residual")
    for model, trace in hidden.items():
        axes[3].plot(index, np.linalg.norm(trace.astype(np.float64), axis=1), color=colors[model], lw=0.7, label=labels[model])
    axes[3].set_ylabel("Hidden-state L2 norm")
    axes[3].set_xlabel("Sample index")
    axes[3].set_title("(d) Learned hidden-state magnitude")
    axes[3].legend(frameon=False, ncol=2, loc="upper right")
    return save_figure(fig, out_dir / "figure_full_trajectory_continuous_inference")


def pca_figure(
    voltage: np.ndarray,
    measured: np.ndarray,
    prediction: np.ndarray,
    hidden: np.ndarray,
    out_dir: Path,
) -> tuple[dict[str, str], dict[str, Any]]:
    centered = hidden.astype(np.float64) - hidden.mean(axis=0, keepdims=True)
    _u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    scores = centered @ vt[:2].T
    variance = singular**2 / max(1, hidden.shape[0] - 1)
    ratio = variance / variance.sum()
    marks = [0, len(measured) // 3, 2 * len(measured) // 3, len(measured) - 1]
    labels = list("ABCD")
    index = np.arange(len(measured))
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.8), constrained_layout=True)
    axes[0, 0].plot(voltage, measured, color="black", lw=0.8, label="Measured")
    axes[0, 0].plot(voltage, prediction, color="#d55e00", lw=0.7, label="Bounded KAN-GRU")
    axes[0, 0].set(xlabel="Input voltage", ylabel="Current-like target", title="(a) Complete I-V trajectory")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].plot(index, prediction - measured, color="#d55e00", lw=0.65)
    axes[0, 1].axhline(0, color="#777777", lw=0.6)
    axes[0, 1].set(xlabel="Sample index", ylabel="Residual", title="(b) Residual")
    axes[1, 0].plot(index, scores[:, 0], color="#0072b2", lw=0.65, label="PC1")
    axes[1, 0].plot(index, scores[:, 1], color="#009e73", lw=0.65, label="PC2")
    axes[1, 0].set(xlabel="Sample index", ylabel="PCA score", title="(c) Hidden-state PCA versus sequence position")
    axes[1, 0].legend(frameon=False)
    axes[1, 1].plot(scores[:, 0], scores[:, 1], color="#555555", lw=0.55)
    axes[1, 1].set(xlabel="PC1", ylabel="PC2", title="(d) Internal-state trajectory")
    for label, position in zip(labels, marks, strict=True):
        axes[0, 1].scatter(position, prediction[position] - measured[position], s=24, zorder=4)
        axes[1, 0].scatter(position, scores[position, 0], s=24, zorder=4)
        axes[1, 1].scatter(scores[position, 0], scores[position, 1], s=24, zorder=4)
        axes[1, 1].annotate(label, (scores[position, 0], scores[position, 1]), xytext=(4, 4), textcoords="offset points")
    paths = save_figure(fig, out_dir / "figure_same_trajectory_prediction_hidden_pca")
    return paths, {
        "fit_rows": int(hidden.shape[0]),
        "hidden_dim": int(hidden.shape[1]),
        "centering": "per-dimension mean centering",
        "scaling": "none",
        "algorithm": "SVD",
        "pc1_explained_variance_ratio": float(ratio[0]),
        "pc2_explained_variance_ratio": float(ratio[1]),
        "pc1_pc2_cumulative_ratio": float(ratio[:2].sum()),
        "marked_indices": marks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("data/processed/sdc_state_splits.npz"))
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/sdc_state_metadata.json"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/sdc/memristor-state-characterisation"))
    parser.add_argument("--out-dir", type=Path, default=Path("reviewer_experiments"))
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--device", default="cpu")
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
    out = args.out_dir.resolve()
    figures = out / "figures"
    traces = out / "hidden_states"
    figures.mkdir(parents=True, exist_ok=True)
    traces.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    cache = np.load(args.cache)
    curves, _ = load_sdc_state_curves(args.raw_root)
    test_ids = [int(value) for value in metadata["split_curve_indices"]["test"]]
    x_all = build_feature_array(cache["test_curve_x"].astype(np.float32), cache["test_curve_lengths"], "no_current")
    y_all = cache["test_curve_y"].astype(np.float32)
    y_mean = float(cache["y_mean"].reshape(-1)[0])
    y_std = float(cache["y_std"].reshape(-1)[0])
    raw_x = cache["test_curve_x_raw"].astype(np.float32)
    raw_y = cache["test_curve_y_raw"].astype(np.float32)
    lengths = cache["test_curve_lengths"].astype(int)

    checkpoints = {
        "clipped_kan_gru_h32": args.kan_checkpoints,
        "gru_h116": args.gru_checkpoints,
    }
    rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    hidden_rows: list[dict[str, Any]] = []
    representative: dict[str, Any] = {}
    for model_name, paths in checkpoints.items():
        for checkpoint in paths:
            model, payload = load_model(checkpoint, device)
            seed = int(payload.get("seed", re.search(r"seed(\d+)", checkpoint.name).group(1)))
            for local_index, length in enumerate(lengths):
                length = int(length)
                x = x_all[local_index, :length]
                target = y_all[local_index, :length, 0]
                reset_prediction, reset_count = reset_trace(model, x, args.window, device)
                continuous_prediction, hidden, continuous_reset_count = continuous_trace(model, x, device)
                expected_reset_count = math.ceil(length / args.window)
                if reset_count != expected_reset_count or continuous_reset_count != 1:
                    raise AssertionError((reset_count, expected_reset_count, continuous_reset_count))
                for mode, prediction, count in (
                    ("reset_window", reset_prediction, reset_count),
                    ("continuous", continuous_prediction, continuous_reset_count),
                ):
                    value = metrics(prediction, target)
                    rows.append({
                        "model": model_name,
                        "seed": seed,
                        "trajectory_id": test_ids[local_index],
                        "local_test_index": local_index,
                        "inference_mode": mode,
                        "num_samples": length,
                        "reset_count": count,
                        **value,
                        "checkpoint": str(checkpoint),
                    })
                boundaries = np.linspace(0, length, 5, dtype=int)
                for quarter in range(4):
                    start, stop = boundaries[quarter], boundaries[quarter + 1]
                    value = metrics(continuous_prediction[start:stop], target[start:stop])
                    position_rows.append({
                        "model": model_name,
                        "seed": seed,
                        "trajectory_id": test_ids[local_index],
                        "local_test_index": local_index,
                        "quarter": quarter + 1,
                        "start": int(start),
                        "stop_exclusive": int(stop),
                        "num_samples": int(stop - start),
                        **value,
                    })
                norm = np.linalg.norm(hidden.astype(np.float64), axis=1)
                trace_path = traces / f"{model_name}_seed{seed}_test{local_index}.npy"
                np.save(trace_path, hidden.astype(np.float32))
                hidden_rows.append({
                    "model": model_name,
                    "seed": seed,
                    "trajectory_id": test_ids[local_index],
                    "local_test_index": local_index,
                    "num_samples": length,
                    "hidden_dim": int(hidden.shape[1]),
                    "norm_min": float(norm.min()),
                    "norm_max": float(norm.max()),
                    "norm_mean": float(norm.mean()),
                    "norm_std": float(norm.std()),
                    "norm_start": float(norm[0]),
                    "norm_middle": float(norm[length // 2]),
                    "norm_end": float(norm[-1]),
                    "activation_min": float(hidden.min()),
                    "activation_max": float(hidden.max()),
                    "activation_mean": float(hidden.mean()),
                    "activation_std": float(hidden.std()),
                    "nan_count": int(np.isnan(hidden).sum()),
                    "inf_count": int(np.isinf(hidden).sum()),
                    "trace_path": str(trace_path),
                })
                if seed == 0 and local_index == 0:
                    representative[model_name] = {
                        "prediction_normalized": continuous_prediction,
                        "hidden": hidden,
                    }
                write_csv(out / "full_trajectory_per_trajectory.partial.csv", rows)
                write_csv(out / "sequence_position_error.partial.csv", position_rows)
                write_csv(out / "hidden_state_summary.partial.csv", hidden_rows)
                print(
                    f"completed model={model_name} seed={seed} "
                    f"trajectory_id={test_ids[local_index]} ({local_index + 1}/{len(lengths)})",
                    flush=True,
                )

    write_csv(out / "full_trajectory_per_trajectory.csv", rows)
    write_csv(out / "sequence_position_error.csv", position_rows)
    write_csv(out / "hidden_state_summary.csv", hidden_rows)

    summary_rows = []
    for model_name in checkpoints:
        model_rows = [row for row in rows if row["model"] == model_name]
        reset = [row for row in model_rows if row["inference_mode"] == "reset_window"]
        continuous = [row for row in model_rows if row["inference_mode"] == "continuous"]
        reset_map = {(row["seed"], row["trajectory_id"]): row for row in reset}
        continuous_map = {(row["seed"], row["trajectory_id"]): row for row in continuous}
        keys = sorted(reset_map)
        delta_r2 = [continuous_map[key]["r2"] - reset_map[key]["r2"] for key in keys]
        delta_rmse = [continuous_map[key]["rmse"] - reset_map[key]["rmse"] for key in keys]
        summary_rows.append({
            "model": model_name,
            "observations": len(keys),
            "reset_r2_mean": mean(row["r2"] for row in reset),
            "reset_r2_std": pstdev(row["r2"] for row in reset),
            "continuous_r2_mean": mean(row["r2"] for row in continuous),
            "continuous_r2_std": pstdev(row["r2"] for row in continuous),
            "delta_r2_mean": mean(delta_r2),
            "delta_r2_median": float(np.median(delta_r2)),
            "reset_rmse_mean": mean(row["rmse"] for row in reset),
            "reset_rmse_std": pstdev(row["rmse"] for row in reset),
            "continuous_rmse_mean": mean(row["rmse"] for row in continuous),
            "continuous_rmse_std": pstdev(row["rmse"] for row in continuous),
            "delta_rmse_mean": mean(delta_rmse),
            "improved": sum(value > 0 for value in delta_r2),
            "degraded": sum(value < 0 for value in delta_r2),
            "ties": sum(value == 0 for value in delta_r2),
        })
    write_csv(out / "full_trajectory_summary.csv", summary_rows)

    position_summary = []
    for model_name in checkpoints:
        subset = [row for row in position_rows if row["model"] == model_name]
        q = {quarter: [row["rmse"] for row in subset if row["quarter"] == quarter] for quarter in range(1, 5)}
        paired = {}
        for row in subset:
            paired.setdefault((row["seed"], row["trajectory_id"]), {})[row["quarter"]] = row["rmse"]
        differences = [values[4] - values[1] for values in paired.values()]
        ratios = [values[4] / values[1] for values in paired.values() if values[1] > 0]
        by_trajectory: dict[int, dict[int, list[float]]] = {}
        for row in subset:
            by_trajectory.setdefault(int(row["trajectory_id"]), {}).setdefault(int(row["quarter"]), []).append(float(row["rmse"]))
        seed_averaged_differences = [mean(values[4]) - mean(values[1]) for values in by_trajectory.values()]
        position_summary.append({
            "model": model_name,
            "observations": len(paired),
            "q1_rmse_mean": mean(q[1]),
            "q2_rmse_mean": mean(q[2]),
            "q3_rmse_mean": mean(q[3]),
            "q4_rmse_mean": mean(q[4]),
            "q4_minus_q1_mean": mean(differences),
            "q4_div_q1_mean": mean(ratios),
            "seed_trajectory_observations_with_q4_gt_q1": sum(value > 0 for value in differences),
            "seed_trajectory_observations": len(differences),
            "seed_averaged_trajectories_with_q4_gt_q1": sum(value > 0 for value in seed_averaged_differences),
            "test_trajectories": len(seed_averaged_differences),
        })
    write_csv(out / "sequence_position_summary.csv", position_summary)

    representative_predictions = {
        name: value["prediction_normalized"] * y_std + y_mean
        for name, value in representative.items()
    }
    representative_hidden = {name: value["hidden"] for name, value in representative.items()}
    n0 = int(lengths[0])
    figure_paths = representative_figure(
        raw_x[0, :n0, 0], raw_y[0, :n0, 0], representative_predictions, representative_hidden, figures
    )
    pca_paths, pca_metadata = pca_figure(
        raw_x[0, :n0, 0],
        raw_y[0, :n0, 0],
        representative_predictions["clipped_kan_gru_h32"],
        representative_hidden["clipped_kan_gru_h32"],
        figures,
    )
    pca_metadata.update({"trajectory_id": test_ids[0], "local_test_index": 0, "seed": 0})
    payload = {
        "dataset": "original submitted SDC formulation",
        "split_counts": metadata["split_curve_counts"],
        "split_seed": metadata["seed"],
        "test_trajectory_ids": test_ids,
        "window": args.window,
        "expected_reset_count_for_15999_samples": math.ceil(15999 / args.window),
        "continuous_reset_count": 1,
        "summary": summary_rows,
        "position_summary": position_summary,
        "pca": pca_metadata,
        "figures": {"full_trajectory": figure_paths, "pca": pca_paths},
    }
    (out / "full_trajectory_analysis.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
