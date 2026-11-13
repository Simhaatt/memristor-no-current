"""Deep interpretability diagnostics for the SDC clipped KAN-GRU.

This is evaluation-only. It loads saved original-SDC no-current checkpoints and
produces:

* top learned KAN edge-function plots,
* edge-importance rank consistency across seeds,
* top-k edge ablation versus random-edge ablation,
* dense-GRU input ablations for comparison,
* compact parametric fits to dominant KAN edges.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
SCRIPT_DIR = ROOT / "scripts"
for path in (SRC_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from memristorkan.metrics import sequence_metrics
from memristorkan.models.kan import KANLayer
from memristorkan.models.recurrent import ClippedKANGRURegressor, GRURegressor
from memristorkan.models.spline import bspline_basis
from run_sdc_sequence import (  # type: ignore
    SequenceVariant,
    WindowedCurveDataset,
    build_sequence_model,
    feature_names_for_mode,
)


@dataclass(frozen=True)
class EdgeSpec:
    layer_name: str
    gate: str
    input_idx: int
    input_name: str
    output_idx: int
    layer_output_idx: int

    @property
    def edge_id(self) -> str:
        return (
            f"{self.layer_name}|{self.gate}|{self.input_idx}:{self.input_name}|"
            f"{self.output_idx}|{self.layer_output_idx}"
        )


class FeatureZeroDataset(Dataset):
    def __init__(self, base: Dataset, feature_index: int | None) -> None:
        self.base = base
        self.feature_index = feature_index

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.base[idx]
        if self.feature_index is None:
            return x, y
        x = x.clone()
        x[:, self.feature_index] = 0.0
        return x, y


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def variant_from_checkpoint(payload: dict[str, Any]) -> SequenceVariant:
    variant = payload.get("variant")
    if not isinstance(variant, dict):
        raise ValueError("checkpoint does not contain a variant dictionary")
    allowed = {"name", "model_name", "hidden_dim", "grid_size", "gauge_every_steps", "moment_mode"}
    return SequenceVariant(**{key: value for key, value in variant.items() if key in allowed})


def build_dataset(
    *,
    cache: Path,
    split: str,
    window: int,
    stride: int,
    feature_mode: str,
    max_windows: int,
) -> WindowedCurveDataset:
    return WindowedCurveDataset(
        cache,
        split,
        window=window,
        stride=stride,
        feature_mode=feature_mode,
        max_windows=max_windows if max_windows > 0 else None,
    )


def load_model(checkpoint_path: Path, dataset: WindowedCurveDataset, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = load_checkpoint(checkpoint_path, device)
    variant = variant_from_checkpoint(payload)
    model = build_sequence_model(variant, input_dim=dataset.feature_dim).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def evaluate_model(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x.to(device)).cpu())
            targets.append(y.cpu())
    return sequence_metrics(torch.cat(preds, dim=0), torch.cat(targets, dim=0))


def edge_function_values(layer: KANLayer, input_idx: int, output_idx: int, grid: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        grid_cpu = grid.detach().cpu()
        basis = bspline_basis(grid_cpu, layer.knots.detach().cpu(), layer.spline_order)
        coef = layer.coef.detach().cpu()[input_idx, output_idx, :]
        spline = basis @ coef
        base = torch.nn.functional.silu(grid_cpu) * layer.scale_base.detach().cpu()[input_idx, output_idx]
        values = base + spline * layer.scale_sp.detach().cpu()[input_idx, output_idx]
    return values.numpy()


def physical_edge_specs(model: ClippedKANGRURegressor, feature_names: list[str]) -> list[EdgeSpec]:
    specs: list[EdgeSpec] = []
    hidden = model.hidden_dim
    for input_idx, input_name in enumerate(feature_names):
        for out in range(hidden):
            specs.append(EdgeSpec("gates", "update_z_raw", input_idx, input_name, out, out))
        for out in range(hidden):
            specs.append(EdgeSpec("gates", "reset_r_raw", input_idx, input_name, out, hidden + out))
        for out in range(hidden):
            specs.append(EdgeSpec("candidate", "candidate_n_raw", input_idx, input_name, out, out))
    return specs


def layer_for_spec(model: ClippedKANGRURegressor, spec: EdgeSpec) -> KANLayer:
    if spec.layer_name == "gates":
        return model.cell.gates
    if spec.layer_name == "candidate":
        return model.cell.candidate
    raise ValueError(f"unknown layer: {spec.layer_name}")


def compute_edge_rows(
    model: ClippedKANGRURegressor,
    *,
    seed: int,
    feature_names: list[str],
    grid: torch.Tensor,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in physical_edge_specs(model, feature_names):
        values = edge_function_values(layer_for_spec(model, spec), spec.input_idx, spec.layer_output_idx, grid)
        rows.append(
            {
                "seed": seed,
                "edge_id": spec.edge_id,
                "layer": spec.layer_name,
                "gate": spec.gate,
                "input_idx": spec.input_idx,
                "input": spec.input_name,
                "output_idx": spec.output_idx,
                "layer_output_idx": spec.layer_output_idx,
                "edge_std": float(np.std(values)),
                "edge_abs_mean": float(np.mean(np.abs(values))),
                "edge_range": float(np.ptp(values)),
                "edge_min": float(np.min(values)),
                "edge_max": float(np.max(values)),
            }
        )
    return rows


def zero_edges(model: ClippedKANGRURegressor, specs: list[EdgeSpec]) -> None:
    with torch.no_grad():
        for spec in specs:
            layer = layer_for_spec(model, spec)
            layer.scale_base[spec.input_idx, spec.layer_output_idx] = 0.0
            layer.scale_sp[spec.input_idx, spec.layer_output_idx] = 0.0


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return ranks


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return pearson(rankdata(a), rankdata(b))


def edge_rank_consistency(edge_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_seed: dict[int, dict[str, float]] = {}
    for row in edge_rows:
        by_seed.setdefault(int(row["seed"]), {})[str(row["edge_id"])] = float(row["edge_std"])
    seeds = sorted(by_seed)
    rows: list[dict[str, Any]] = []
    for i, seed_a in enumerate(seeds):
        for seed_b in seeds[i + 1 :]:
            common = sorted(set(by_seed[seed_a]) & set(by_seed[seed_b]))
            a = np.array([by_seed[seed_a][edge] for edge in common], dtype=float)
            b = np.array([by_seed[seed_b][edge] for edge in common], dtype=float)
            rows.append(
                {
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "common_edges": len(common),
                    "spearman_rho": spearman(a, b),
                    "pearson_r": pearson(a, b),
                }
            )
    return rows


def aggregate_edge_importance(edge_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in edge_rows:
        grouped.setdefault(str(row["edge_id"]), []).append(row)
    out: list[dict[str, Any]] = []
    for edge_id, rows in grouped.items():
        first = rows[0]
        values = [float(row["edge_std"]) for row in rows]
        out.append(
            {
                "edge_id": edge_id,
                "runs": len(rows),
                "layer": first["layer"],
                "gate": first["gate"],
                "input_idx": first["input_idx"],
                "input": first["input"],
                "output_idx": first["output_idx"],
                "layer_output_idx": first["layer_output_idx"],
                "edge_std_mean": mean(values),
                "edge_std_std": pstdev(values) if len(values) > 1 else 0.0,
                "edge_abs_mean_mean": mean(float(row["edge_abs_mean"]) for row in rows),
                "edge_range_mean": mean(float(row["edge_range"]) for row in rows),
            }
        )
    out.sort(key=lambda row: float(row["edge_std_mean"]), reverse=True)
    return out


def edge_spec_from_row(row: dict[str, Any]) -> EdgeSpec:
    return EdgeSpec(
        layer_name=str(row["layer"]),
        gate=str(row["gate"]),
        input_idx=int(row["input_idx"]),
        input_name=str(row["input"]),
        output_idx=int(row["output_idx"]),
        layer_output_idx=int(row["layer_output_idx"]),
    )


def summarize_metric_rows(
    rows: list[dict[str, Any]],
    keys: list[str],
    metrics: list[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    out: list[dict[str, Any]] = []
    for key_values, group in sorted(grouped.items()):
        out_row = {key: value for key, value in zip(keys, key_values)}
        out_row["runs"] = len(group)
        for metric in metrics:
            vals = [float(item[metric]) for item in group]
            out_row[f"{metric}_mean"] = mean(vals)
            out_row[f"{metric}_std"] = pstdev(vals) if len(vals) > 1 else 0.0
        out.append(out_row)
    return out


def topk_edge_ablation(
    *,
    checkpoints: list[Path],
    cache: Path,
    split: str,
    window: int,
    stride: int,
    batch_size: int,
    max_windows: int,
    device: torch.device,
    edge_rows: list[dict[str, Any]],
    k_values: list[int],
    random_repeats: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seed_edges: dict[int, list[dict[str, Any]]] = {}
    for row in edge_rows:
        seed_edges.setdefault(int(row["seed"]), []).append(row)
    for checkpoint in checkpoints:
        payload = load_checkpoint(checkpoint, device)
        seed = int(payload.get("seed", -1))
        feature_mode = str(payload.get("feature_mode", "no_current"))
        dataset = build_dataset(
            cache=cache,
            split=split,
            window=window,
            stride=stride,
            feature_mode=feature_mode,
            max_windows=max_windows,
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        base_model, _payload = load_model(checkpoint, dataset, device)
        if not isinstance(base_model, ClippedKANGRURegressor):
            raise TypeError("top-k edge ablation expects clipped KAN-GRU")
        baseline = evaluate_model(base_model, loader, device)
        ranked = sorted(seed_edges[seed], key=lambda row: float(row["edge_std"]), reverse=True)
        universe = [edge_spec_from_row(row) for row in ranked]
        rng = random.Random(1000 + seed)
        for k in k_values:
            k_eff = min(k, len(universe))
            top_specs = [edge_spec_from_row(row) for row in ranked[:k_eff]]
            top_model = copy.deepcopy(base_model).to(device)
            zero_edges(top_model, top_specs)
            metrics = evaluate_model(top_model, loader, device)
            rows.append(
                {
                    "variant": "clipped_kan_gru_h32",
                    "seed": seed,
                    "ablation_kind": "top_by_edge_std",
                    "k": k_eff,
                    "repeat": 0,
                    "r2": metrics["r2"],
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "final_state_mae": metrics["final_state_mae"],
                    "baseline_r2": baseline["r2"],
                    "r2_drop": baseline["r2"] - metrics["r2"],
                    "baseline_rmse": baseline["rmse"],
                    "rmse_increase": metrics["rmse"] - baseline["rmse"],
                }
            )
            for repeat in range(random_repeats):
                random_specs = rng.sample(universe, k_eff)
                random_model = copy.deepcopy(base_model).to(device)
                zero_edges(random_model, random_specs)
                metrics = evaluate_model(random_model, loader, device)
                rows.append(
                    {
                        "variant": "clipped_kan_gru_h32",
                        "seed": seed,
                        "ablation_kind": "random_physical_edges",
                        "k": k_eff,
                        "repeat": repeat,
                        "r2": metrics["r2"],
                        "rmse": metrics["rmse"],
                        "mae": metrics["mae"],
                        "final_state_mae": metrics["final_state_mae"],
                        "baseline_r2": baseline["r2"],
                        "r2_drop": baseline["r2"] - metrics["r2"],
                        "baseline_rmse": baseline["rmse"],
                        "rmse_increase": metrics["rmse"] - baseline["rmse"],
                    }
                )
    return rows


def input_feature_ablation(
    *,
    label: str,
    checkpoints: list[Path],
    cache: Path,
    split: str,
    window: int,
    stride: int,
    batch_size: int,
    max_windows: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        payload = load_checkpoint(checkpoint, device)
        seed = int(payload.get("seed", -1))
        feature_mode = str(payload.get("feature_mode", "no_current"))
        feature_names = list(payload.get("feature_names", feature_names_for_mode(feature_mode)))
        dataset = build_dataset(
            cache=cache,
            split=split,
            window=window,
            stride=stride,
            feature_mode=feature_mode,
            max_windows=max_windows,
        )
        model, payload = load_model(checkpoint, dataset, device)
        baseline_loader = DataLoader(FeatureZeroDataset(dataset, None), batch_size=batch_size, shuffle=False)
        baseline = evaluate_model(model, baseline_loader, device)
        for feature_idx, feature_name in [(None, "none"), *list(enumerate(feature_names))]:
            loader = DataLoader(FeatureZeroDataset(dataset, feature_idx), batch_size=batch_size, shuffle=False)
            metrics = baseline if feature_idx is None else evaluate_model(model, loader, device)
            rows.append(
                {
                    "variant": label,
                    "seed": seed,
                    "intervention": "none" if feature_idx is None else f"zero_{feature_name}",
                    "feature": feature_name,
                    "feature_idx": -1 if feature_idx is None else feature_idx,
                    "r2": metrics["r2"],
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "final_state_mae": metrics["final_state_mae"],
                    "baseline_r2": baseline["r2"],
                    "r2_drop": baseline["r2"] - metrics["r2"],
                    "baseline_rmse": baseline["rmse"],
                    "rmse_increase": metrics["rmse"] - baseline["rmse"],
                    "checkpoint_path": str(checkpoint),
                }
            )
    return rows


def r2_score_np(y: np.ndarray, pred: np.ndarray) -> float:
    denom = float(np.sum((y - np.mean(y)) ** 2))
    if denom <= 1e-12:
        return float("nan")
    return float(1.0 - np.sum((y - pred) ** 2) / denom)


def fit_linear(x: np.ndarray, y: np.ndarray) -> tuple[float, dict[str, float]]:
    coef = np.polyfit(x, y, deg=1)
    pred = coef[0] * x + coef[1]
    return r2_score_np(y, pred), {"a": float(coef[0]), "b": float(coef[1])}


def fit_threshold(x: np.ndarray, y: np.ndarray) -> tuple[float, dict[str, float], np.ndarray]:
    best_r2 = -float("inf")
    best_params = {"threshold": float(x[0]), "left": float(np.mean(y)), "right": float(np.mean(y))}
    best_pred = np.full_like(y, np.mean(y))
    for threshold in x[8:-8]:
        mask = x <= threshold
        if mask.sum() < 4 or (~mask).sum() < 4:
            continue
        left = float(np.mean(y[mask]))
        right = float(np.mean(y[~mask]))
        pred = np.where(mask, left, right)
        score = r2_score_np(y, pred)
        if score > best_r2:
            best_r2 = score
            best_params = {"threshold": float(threshold), "left": left, "right": right}
            best_pred = pred
    return best_r2, best_params, best_pred


def fit_tanh(x: np.ndarray, y: np.ndarray) -> tuple[float, dict[str, float], np.ndarray]:
    torch.manual_seed(0)
    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    params = torch.nn.Parameter(
        torch.tensor(
            [
                float((np.max(y) - np.min(y)) / 2.0),
                1.0,
                0.0,
                float(np.mean(y)),
            ],
            dtype=torch.float32,
        )
    )
    opt = torch.optim.Adam([params], lr=0.03)
    for _ in range(800):
        opt.zero_grad()
        a, b, c, d = params
        pred = a * torch.tanh(b * x_t + c) + d
        loss = torch.mean((pred - y_t) ** 2)
        loss.backward()
        opt.step()
        with torch.no_grad():
            params[1].clamp_(-20.0, 20.0)
    with torch.no_grad():
        a, b, c, d = [float(v) for v in params]
        pred_np = (params[0] * torch.tanh(params[1] * x_t + params[2]) + params[3]).numpy()
    return r2_score_np(y, pred_np), {"a": a, "b": b, "c": c, "d": d}, pred_np


def parametric_edge_fits(
    *,
    top_edges: list[dict[str, Any]],
    checkpoints: list[Path],
    cache: Path,
    grid: torch.Tensor,
    out_dir: Path,
    top_n: int,
) -> list[dict[str, Any]]:
    models: dict[int, ClippedKANGRURegressor] = {}
    feature_names: list[str] = []
    for checkpoint in checkpoints:
        payload = load_checkpoint(checkpoint, torch.device("cpu"))
        feature_mode = str(payload.get("feature_mode", "no_current"))
        dataset = build_dataset(
            cache=cache,
            split="test",
            window=256,
            stride=512,
            feature_mode=feature_mode,
            max_windows=1,
        )
        model, payload = load_model(checkpoint, dataset, torch.device("cpu"))
        if not isinstance(model, ClippedKANGRURegressor):
            raise TypeError("parametric fits expect clipped KAN-GRU")
        seed = int(payload.get("seed", -1))
        models[seed] = model
        feature_names = list(payload.get("feature_names", feature_names_for_mode(feature_mode)))

    x = grid.detach().cpu().numpy()
    rows: list[dict[str, Any]] = []
    fit_examples: list[tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for edge in top_edges[:top_n]:
        spec = edge_spec_from_row(edge)
        curves = []
        for model in models.values():
            curves.append(edge_function_values(layer_for_spec(model, spec), spec.input_idx, spec.layer_output_idx, grid))
        y = np.mean(np.stack(curves, axis=0), axis=0)
        linear_r2, linear_params = fit_linear(x, y)
        tanh_r2, tanh_params, tanh_pred = fit_tanh(x, y)
        threshold_r2, threshold_params, threshold_pred = fit_threshold(x, y)
        lin_pred = linear_params["a"] * x + linear_params["b"]
        candidates = [
            ("linear", linear_r2, linear_params),
            ("tanh", tanh_r2, tanh_params),
            ("threshold", threshold_r2, threshold_params),
        ]
        best = max(candidates, key=lambda item: item[1])
        rows.append(
            {
                "edge_id": spec.edge_id,
                "gate": spec.gate,
                "input": spec.input_name,
                "output_idx": spec.output_idx,
                "edge_std_mean": edge["edge_std_mean"],
                "linear_r2": linear_r2,
                "linear_params": json.dumps(linear_params, sort_keys=True),
                "tanh_r2": tanh_r2,
                "tanh_params": json.dumps(tanh_params, sort_keys=True),
                "threshold_r2": threshold_r2,
                "threshold_params": json.dumps(threshold_params, sort_keys=True),
                "best_family": best[0],
                "best_r2": best[1],
            }
        )
        if len(fit_examples) < 4:
            fit_examples.append((edge, y, lin_pred, tanh_pred, threshold_pred))

    make_fit_example_plot(out_dir, x, fit_examples)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 4) -> str:
    try:
        num = float(value)
    except Exception:
        return str(value)
    if math.isnan(num):
        return "nan"
    return f"{num:.{digits}f}"


def make_plots(
    *,
    out_dir: Path,
    grid: torch.Tensor,
    aggregate_edges: list[dict[str, Any]],
    edge_rows: list[dict[str, Any]],
    checkpoints: list[Path],
    cache: Path,
    topk_summary: list[dict[str, Any]],
    feature_summary: list[dict[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    x = grid.detach().cpu().numpy()
    models: dict[int, ClippedKANGRURegressor] = {}
    for checkpoint in checkpoints:
        payload = load_checkpoint(checkpoint, torch.device("cpu"))
        feature_mode = str(payload.get("feature_mode", "no_current"))
        dataset = build_dataset(
            cache=cache,
            split="test",
            window=256,
            stride=512,
            feature_mode=feature_mode,
            max_windows=1,
        )
        model, payload = load_model(checkpoint, dataset, torch.device("cpu"))
        if isinstance(model, ClippedKANGRURegressor):
            models[int(payload.get("seed", -1))] = model

    top_edges = aggregate_edges[:8]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.3), sharex=True)
    for ax, edge in zip(axes.ravel(), top_edges):
        spec = edge_spec_from_row(edge)
        for seed, model in sorted(models.items()):
            y = edge_function_values(layer_for_spec(model, spec), spec.input_idx, spec.layer_output_idx, grid)
            ax.plot(x, y, linewidth=1.4, label=f"seed {seed}")
        ax.set_title(f"{spec.gate}\n{spec.input_name} -> unit {spec.output_idx}", fontsize=9)
        ax.grid(alpha=0.25)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.supxlabel("Normalized input value")
    fig.supylabel("Learned edge contribution")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / "top_learned_edge_functions.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for kind, color in [("top_by_edge_std", "#A84E3B"), ("random_physical_edges", "#4C78A8")]:
        rows = [row for row in topk_summary if row["ablation_kind"] == kind]
        rows = sorted(rows, key=lambda row: int(row["k"]))
        ax.errorbar(
            [int(row["k"]) for row in rows],
            [float(row["r2_drop_mean"]) for row in rows],
            yerr=[float(row["r2_drop_std"]) for row in rows],
            marker="o",
            linewidth=2,
            label=kind,
            color=color,
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Ablated physical KAN edges")
    ax.set_ylabel("Test R2 drop")
    ax.set_title("KAN Edge-Importance Validation")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "topk_vs_random_edge_ablation.png", dpi=220)
    plt.close(fig)

    rows = [row for row in feature_summary if row["intervention"] != "none"]
    variants = sorted(set(str(row["variant"]) for row in rows))
    features = ["zero_V_t", "zero_V_t_minus_1", "zero_dV_t", "zero_sweep_direction"]
    x_pos = np.arange(len(features))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for idx, variant in enumerate(variants):
        vals = []
        errs = []
        for feature in features:
            match = next(row for row in rows if row["variant"] == variant and row["intervention"] == feature)
            vals.append(float(match["r2_drop_mean"]))
            errs.append(float(match["r2_drop_std"]))
        ax.bar(x_pos + (idx - 0.5) * width, vals, width, yerr=errs, label=variant, capsize=3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(features, rotation=20, ha="right")
    ax.set_ylabel("Test R2 drop")
    ax.set_title("Input Feature Ablation: KAN-GRU vs Dense GRU")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "dense_gru_vs_kan_input_ablation.png", dpi=220)
    plt.close(fig)


def make_fit_example_plot(
    out_dir: Path,
    x: np.ndarray,
    examples: list[tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    if not examples:
        return
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, (edge, y, lin_pred, tanh_pred, threshold_pred) in zip(axes.ravel(), examples):
        ax.plot(x, y, linewidth=2.2, label="KAN edge", color="#222222")
        ax.plot(x, lin_pred, linestyle="--", label="linear")
        ax.plot(x, tanh_pred, linestyle="--", label="tanh")
        ax.plot(x, threshold_pred, linestyle="--", label="threshold")
        ax.set_title(f"{edge['gate']} / {edge['input']} / unit {edge['output_idx']}", fontsize=9)
        ax.grid(alpha=0.25)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.supxlabel("Normalized input value")
    fig.supylabel("Edge contribution")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / "parametric_fit_examples.png", dpi=220)
    plt.close(fig)


def write_markdown(
    path: Path,
    *,
    aggregate_edges: list[dict[str, Any]],
    rank_rows: list[dict[str, Any]],
    topk_summary: list[dict[str, Any]],
    feature_summary: list[dict[str, Any]],
    fit_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    rank_mean = mean(float(row["spearman_rho"]) for row in rank_rows) if rank_rows else float("nan")
    lines = [
        "# KAN Gate Deep-Dive Interpretability Results",
        "",
        "This analysis is evaluation-only and uses the locked original SDC no-current checkpoints.",
        "",
        "## Analysis Settings",
        "",
        f"- Cache: `{manifest['cache']}`",
        f"- Split: `{manifest['split']}`",
        f"- Window/stride: `{manifest['window']}` / `{manifest['stride']}`",
        f"- KAN checkpoints: {len(manifest['kan_checkpoints'])}",
        f"- Dense GRU checkpoints: {len(manifest['gru_checkpoints'])}",
        f"- Top-k ablation k values: `{manifest['k_values']}`",
        "",
        "## 1. Top Learned Edge Functions",
        "",
        "The top physical-input KAN edges were ranked by the standard deviation of the represented edge function over the normalized grid.",
        "",
        "| Rank | Gate | Input | Unit | Edge std | Edge range |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for idx, row in enumerate(aggregate_edges[:12], start=1):
        lines.append(
            f"| {idx} | `{row['gate']}` | `{row['input']}` | {row['output_idx']} | "
            f"{fmt(row['edge_std_mean'])} +/- {fmt(row['edge_std_std'])} | {fmt(row['edge_range_mean'])} |"
        )

    lines.extend(
        [
            "",
            "Figure files:",
            "",
            "- `top_learned_edge_functions.png`",
            "- `parametric_fit_examples.png`",
            "",
            "## 2. Seed Consistency",
            "",
            f"Mean pairwise Spearman rank correlation across edge-importance rankings: **{fmt(rank_mean)}**.",
            "",
            "| Seed A | Seed B | Common edges | Spearman rho | Pearson r |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rank_rows:
        lines.append(
            f"| {row['seed_a']} | {row['seed_b']} | {row['common_edges']} | "
            f"{fmt(row['spearman_rho'])} | {fmt(row['pearson_r'])} |"
        )

    lines.extend(
        [
            "",
            "## 3. Top-k Edge Ablation Versus Random Edges",
            "",
            "This validates whether the spline standard-deviation score is load-bearing. If top-ranked edges degrade faster than random physical edges, the importance score has intervention support.",
            "",
            "| Ablation | k | Runs | R2 drop | RMSE increase |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    topk_sorted = sorted(topk_summary, key=lambda row: (int(row["k"]), str(row["ablation_kind"])))
    for row in topk_sorted:
        lines.append(
            f"| `{row['ablation_kind']}` | {row['k']} | {row['runs']} | "
            f"{fmt(row['r2_drop_mean'])} +/- {fmt(row['r2_drop_std'])} | "
            f"{fmt(row['rmse_increase_mean'])} |"
        )

    lines.extend(
        [
            "",
            "Figure file: `topk_vs_random_edge_ablation.png`",
            "",
            "## 4. Dense GRU Input Ablation",
            "",
            "Input-level feature ablations were run on both clipped KAN-GRU and dense GRU h116. This makes the input-attribution claim comparative: both models can be probed at the feature level, while the KAN model additionally exposes edge-level functions.",
            "",
            "| Model | Intervention | Runs | R2 | R2 drop | RMSE increase |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    feature_sorted = sorted(feature_summary, key=lambda row: (str(row["variant"]), -float(row["r2_drop_mean"])))
    for row in feature_sorted:
        lines.append(
            f"| `{row['variant']}` | `{row['intervention']}` | {row['runs']} | "
            f"{fmt(row['r2_mean'])} +/- {fmt(row['r2_std'])} | "
            f"{fmt(row['r2_drop_mean'])} +/- {fmt(row['r2_drop_std'])} | "
            f"{fmt(row['rmse_increase_mean'])} |"
        )

    lines.extend(
        [
            "",
            "Figure file: `dense_gru_vs_kan_input_ablation.png`",
            "",
            "## 5. Compact Parametric Fits To Dominant Edges",
            "",
            "| Edge | Gate | Input | Unit | Best family | Best R2 | Linear R2 | Tanh R2 | Threshold R2 |",
            "|---|---|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in fit_rows:
        lines.append(
            f"| `{row['edge_id']}` | `{row['gate']}` | `{row['input']}` | {row['output_idx']} | "
            f"`{row['best_family']}` | {fmt(row['best_r2'])} | {fmt(row['linear_r2'])} | "
            f"{fmt(row['tanh_r2'])} | {fmt(row['threshold_r2'])} |"
        )

    lines.extend(
        [
            "",
            "## Manuscript-Safe Claim",
            "",
            "The clipped KAN-GRU exposes learned one-dimensional edge functions inside the recurrent gates. The edge-function plots and compact fits make these gate dependencies inspectable, the cross-seed rank correlations test stability, and top-k ablations test whether the ranked edges are load-bearing. This supports an interpretability contribution at the level of learned function structure, not a claim that the network recovered a unique physical switching law.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("data/processed/sdc_state_splits.npz"))
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--grid-points", type=int, default=401)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--kan-checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--gru-checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    parser.add_argument("--random-repeats", type=int, default=3)
    parser.add_argument("--top-fit-edges", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=Path("results/interpretability/kan_gate_deep_dive_sdc"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    grid = torch.linspace(-1.0, 1.0, args.grid_points)

    edge_rows: list[dict[str, Any]] = []
    for checkpoint in args.kan_checkpoints:
        payload = load_checkpoint(checkpoint, device)
        feature_mode = str(payload.get("feature_mode", "no_current"))
        feature_names = list(payload.get("feature_names", feature_names_for_mode(feature_mode)))
        dataset = build_dataset(
            cache=args.cache,
            split=args.split,
            window=args.window,
            stride=args.stride,
            feature_mode=feature_mode,
            max_windows=args.max_windows,
        )
        model, payload = load_model(checkpoint, dataset, device)
        if not isinstance(model, ClippedKANGRURegressor):
            raise TypeError(f"expected clipped KAN-GRU checkpoint, got {type(model)}")
        edge_rows.extend(
            compute_edge_rows(
                model,
                seed=int(payload.get("seed", -1)),
                feature_names=feature_names,
                grid=grid,
            )
        )

    aggregate_edges = aggregate_edge_importance(edge_rows)
    rank_rows = edge_rank_consistency(edge_rows)
    topk_rows = topk_edge_ablation(
        checkpoints=args.kan_checkpoints,
        cache=args.cache,
        split=args.split,
        window=args.window,
        stride=args.stride,
        batch_size=args.batch_size,
        max_windows=args.max_windows,
        device=device,
        edge_rows=edge_rows,
        k_values=args.k_values,
        random_repeats=args.random_repeats,
    )
    topk_summary = summarize_metric_rows(
        topk_rows,
        keys=["ablation_kind", "k"],
        metrics=["r2", "rmse", "final_state_mae", "r2_drop", "rmse_increase"],
    )
    feature_rows = []
    feature_rows.extend(
        input_feature_ablation(
            label="clipped_kan_gru_h32",
            checkpoints=args.kan_checkpoints,
            cache=args.cache,
            split=args.split,
            window=args.window,
            stride=args.stride,
            batch_size=args.batch_size,
            max_windows=args.max_windows,
            device=device,
        )
    )
    feature_rows.extend(
        input_feature_ablation(
            label="gru_h116",
            checkpoints=args.gru_checkpoints,
            cache=args.cache,
            split=args.split,
            window=args.window,
            stride=args.stride,
            batch_size=args.batch_size,
            max_windows=args.max_windows,
            device=device,
        )
    )
    feature_summary = summarize_metric_rows(
        feature_rows,
        keys=["variant", "intervention", "feature"],
        metrics=["r2", "rmse", "final_state_mae", "r2_drop", "rmse_increase"],
    )
    fit_rows = parametric_edge_fits(
        top_edges=aggregate_edges,
        checkpoints=args.kan_checkpoints,
        cache=args.cache,
        grid=grid,
        out_dir=args.out_dir,
        top_n=args.top_fit_edges,
    )

    write_csv(args.out_dir / "edge_importance_per_edge_seed.csv", edge_rows)
    write_csv(args.out_dir / "edge_importance_ranked_mean.csv", aggregate_edges)
    write_csv(args.out_dir / "edge_rank_consistency.csv", rank_rows)
    write_csv(args.out_dir / "edge_ablation_topk_vs_random_per_run.csv", topk_rows)
    write_csv(args.out_dir / "edge_ablation_topk_vs_random_summary.csv", topk_summary)
    write_csv(args.out_dir / "input_feature_ablation_per_seed.csv", feature_rows)
    write_csv(args.out_dir / "input_feature_ablation_summary.csv", feature_summary)
    write_csv(args.out_dir / "parametric_edge_fits.csv", fit_rows)

    make_plots(
        out_dir=args.out_dir,
        grid=grid,
        aggregate_edges=aggregate_edges,
        edge_rows=edge_rows,
        checkpoints=args.kan_checkpoints,
        cache=args.cache,
        topk_summary=topk_summary,
        feature_summary=feature_summary,
    )
    manifest = {
        "cache": str(args.cache),
        "split": args.split,
        "window": args.window,
        "stride": args.stride,
        "batch_size": args.batch_size,
        "max_windows": args.max_windows,
        "grid_points": args.grid_points,
        "device": args.device,
        "kan_checkpoints": [str(path) for path in args.kan_checkpoints],
        "gru_checkpoints": [str(path) for path in args.gru_checkpoints],
        "k_values": args.k_values,
        "random_repeats": args.random_repeats,
        "top_fit_edges": args.top_fit_edges,
    }
    (args.out_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_markdown(
        args.out_dir / "kan_gate_deep_dive_results.md",
        aggregate_edges=aggregate_edges,
        rank_rows=rank_rows,
        topk_summary=topk_summary,
        feature_summary=feature_summary,
        fit_rows=fit_rows,
        manifest=manifest,
    )
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "edge_rows": len(edge_rows),
                "topk_ablation_rows": len(topk_rows),
                "feature_ablation_rows": len(feature_rows),
                "fit_rows": len(fit_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
