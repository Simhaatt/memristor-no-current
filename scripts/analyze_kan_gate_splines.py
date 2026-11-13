"""Interpret learned KAN gate splines from saved recurrent checkpoints.

The script is evaluation-only. It loads existing clipped KAN-GRU checkpoints,
summarizes the learned univariate KAN edge functions for the physical inputs,
and records runtime gate activation statistics on a selected split.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
SCRIPT_DIR = ROOT / "scripts"
for path in (SRC_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from memristorkan.models.kan import KANLayer
from memristorkan.models.spline import bspline_basis
from memristorkan.models.recurrent import ClippedKANGRURegressor
from run_sdc_sequence import (  # type: ignore  # imported from sibling script
    SequenceVariant,
    WindowedCurveDataset,
    build_sequence_model,
    feature_names_for_mode,
)


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


def clipped_kangru_gate_step(
    model: ClippedKANGRURegressor,
    x_t: torch.Tensor,
    h_prev: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    gate_input = torch.cat([x_t, h_prev], dim=-1)
    z_raw, r_raw = model.cell.gates(gate_input).chunk(2, dim=-1)
    z_t = ((z_raw + 1.0) * 0.5).clamp(0.0, 1.0)
    r_t = ((r_raw + 1.0) * 0.5).clamp(0.0, 1.0)
    candidate_input = torch.cat([x_t, r_t * h_prev], dim=-1)
    n_raw = model.cell.candidate(candidate_input)
    n_t = n_raw.clamp(-1.0, 1.0)
    h_t = (1.0 - z_t) * n_t + z_t * h_prev
    return h_t, {
        "z_raw": z_raw,
        "r_raw": r_raw,
        "n_raw": n_raw,
        "z_t": z_t,
        "r_t": r_t,
        "n_t": n_t,
    }


def edge_values(layer: KANLayer, input_idx: int, output_indices: np.ndarray, grid: torch.Tensor) -> torch.Tensor:
    """Return represented edge functions for one input and many outputs."""

    with torch.no_grad():
        basis = bspline_basis(grid, layer.knots.detach().cpu(), layer.spline_order)
        coef = layer.coef.detach().cpu()[input_idx, output_indices, :]
        scale_sp = layer.scale_sp.detach().cpu()[input_idx, output_indices]
        scale_base = layer.scale_base.detach().cpu()[input_idx, output_indices]
        spline = basis @ coef.T
        base = torch.nn.functional.silu(grid).unsqueeze(1) * scale_base.unsqueeze(0)
        return base + spline * scale_sp.unsqueeze(0)


def summarize_edges(
    *,
    seed: int,
    layer: KANLayer,
    layer_name: str,
    gate_name: str,
    output_indices: np.ndarray,
    feature_names: list[str],
    hidden_dim: int,
    grid: torch.Tensor,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    input_groups: list[tuple[str, str, list[int]]] = [
        (name, "physical", [idx]) for idx, name in enumerate(feature_names)
    ]
    hidden_indices = list(range(len(feature_names), len(feature_names) + hidden_dim))
    input_groups.append(("hidden_state_mean", "hidden", hidden_indices))

    for input_name, input_type, input_indices in input_groups:
        edge_chunks = []
        scale_base_chunks = []
        scale_sp_chunks = []
        for input_idx in input_indices:
            values = edge_values(layer, input_idx, output_indices, grid).numpy()
            edge_chunks.append(values)
            scale_base_chunks.append(layer.scale_base.detach().cpu()[input_idx, output_indices].abs().numpy())
            scale_sp_chunks.append(layer.scale_sp.detach().cpu()[input_idx, output_indices].abs().numpy())
        edge_matrix = np.concatenate(edge_chunks, axis=1)
        centered = edge_matrix - edge_matrix.mean(axis=0, keepdims=True)
        rows.append(
            {
                "seed": seed,
                "layer": layer_name,
                "gate": gate_name,
                "input": input_name,
                "input_type": input_type,
                "edge_count": int(edge_matrix.shape[1]),
                "edge_abs_mean": float(np.mean(np.abs(edge_matrix))),
                "edge_centered_abs_mean": float(np.mean(np.abs(centered))),
                "edge_std_mean": float(np.mean(np.std(edge_matrix, axis=0))),
                "edge_range_mean": float(np.mean(np.ptp(edge_matrix, axis=0))),
                "scale_base_abs_mean": float(np.mean(np.concatenate(scale_base_chunks))),
                "scale_sp_abs_mean": float(np.mean(np.concatenate(scale_sp_chunks))),
            }
        )
    return rows


def summarize_gate_tensor(values: np.ndarray, *, seed: int, gate: str) -> dict[str, Any]:
    flat = values.reshape(-1)
    row: dict[str, Any] = {
        "seed": seed,
        "gate": gate,
        "count": int(flat.size),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "p05": float(np.quantile(flat, 0.05)),
        "p50": float(np.quantile(flat, 0.50)),
        "p95": float(np.quantile(flat, 0.95)),
    }
    if gate in {"z_t", "r_t"}:
        row["clip_low_fraction"] = float(np.mean(flat <= 1e-6))
        row["clip_high_fraction"] = float(np.mean(flat >= 1.0 - 1e-6))
    elif gate == "n_t":
        row["clip_low_fraction"] = float(np.mean(flat <= -1.0 + 1e-6))
        row["clip_high_fraction"] = float(np.mean(flat >= 1.0 - 1e-6))
    else:
        row["clip_low_fraction"] = float("nan")
        row["clip_high_fraction"] = float("nan")
    return row


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def runtime_gate_stats(
    *,
    model: ClippedKANGRURegressor,
    loader: DataLoader,
    device: torch.device,
    seed: int,
    feature_names: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray]]:
    gate_values: dict[str, list[np.ndarray]] = {name: [] for name in ["z_t", "r_t", "n_t", "z_raw", "r_raw", "n_raw"]}
    gate_means: dict[str, list[np.ndarray]] = {name: [] for name in ["z_t", "r_t", "n_t"]}
    feature_values: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for x, _y in loader:
            x = x.to(device)
            batch, seq_len, _ = x.shape
            h_t = x.new_zeros(batch, model.hidden_dim)
            for t in range(seq_len):
                x_t = x[:, t, :]
                h_t, gates = clipped_kangru_gate_step(model, x_t, h_t)
                feature_values.append(x_t.detach().cpu().numpy())
                for name, value in gates.items():
                    arr = value.detach().cpu().numpy()
                    gate_values[name].append(arr)
                    if name in gate_means:
                        gate_means[name].append(arr.mean(axis=1))

    gate_arrays = {name: np.concatenate(chunks, axis=0) for name, chunks in gate_values.items()}
    mean_arrays = {name: np.concatenate(chunks, axis=0) for name, chunks in gate_means.items()}
    features = np.concatenate(feature_values, axis=0)

    stats_rows = [summarize_gate_tensor(gate_arrays[name], seed=seed, gate=name) for name in ["z_t", "r_t", "n_t"]]
    corr_rows: list[dict[str, Any]] = []
    for gate_name, gate_mean in mean_arrays.items():
        for feature_idx, feature_name in enumerate(feature_names):
            corr_rows.append(
                {
                    "seed": seed,
                    "gate": gate_name,
                    "feature": feature_name,
                    "corr_gate_mean_with_feature": safe_corr(gate_mean, features[:, feature_idx]),
                }
            )
    return stats_rows, corr_rows, {"gates": gate_arrays, "gate_means": mean_arrays, "features": features}


def aggregate_rows(rows: list[dict[str, Any]], keys: list[str], numeric_fields: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    out: list[dict[str, Any]] = []
    for key_values, group in sorted(groups.items()):
        row = {key: value for key, value in zip(keys, key_values)}
        row["runs"] = len(group)
        for field in numeric_fields:
            values = [float(item[field]) for item in group if not np.isnan(float(item[field]))]
            row[f"{field}_mean"] = mean(values) if values else float("nan")
            row[f"{field}_std"] = pstdev(values) if len(values) > 1 else 0.0
        out.append(row)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(
    *,
    out_dir: Path,
    edge_summary: list[dict[str, Any]],
    gate_summary: list[dict[str, Any]],
    feature_names: list[str],
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        return [f"plotting skipped: {exc}"]

    plot_paths: list[str] = []
    physical = [row for row in edge_summary if row["input_type"] == "physical"]
    gates = ["update_z_raw", "reset_r_raw", "candidate_n_raw"]
    x = np.arange(len(feature_names))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10, 4.8))
    for idx, gate in enumerate(gates):
        values = []
        for feature in feature_names:
            matches = [
                row
                for row in physical
                if row["gate"] == gate and row["input"] == feature
            ]
            values.append(matches[0]["edge_std_mean_mean"] if matches else 0.0)
        ax.bar(x + (idx - 1) * width, values, width=width, label=gate)
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=20, ha="right")
    ax.set_ylabel("Mean edge-function std over spline grid")
    ax.set_title("Clipped KAN-GRU physical-input spline importance")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "physical_input_spline_importance.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    plot_paths.append(str(path))

    gate_order = ["z_t", "r_t", "n_t"]
    means = []
    lows = []
    highs = []
    for gate in gate_order:
        matches = [row for row in gate_summary if row["gate"] == gate]
        if matches:
            means.append(matches[0]["mean_mean"])
            lows.append(matches[0]["clip_low_fraction_mean"])
            highs.append(matches[0]["clip_high_fraction_mean"])
        else:
            means.append(0.0)
            lows.append(0.0)
            highs.append(0.0)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    axes[0].bar(gate_order, means, color=["#4C78A8", "#F58518", "#54A24B"])
    axes[0].set_title("Mean gate activation")
    axes[0].set_ylim(-1.05, 1.05)
    axes[0].set_ylabel("Activation")
    axes[1].bar(np.arange(len(gate_order)) - 0.18, lows, width=0.36, label="low clip")
    axes[1].bar(np.arange(len(gate_order)) + 0.18, highs, width=0.36, label="high clip")
    axes[1].set_xticks(np.arange(len(gate_order)))
    axes[1].set_xticklabels(gate_order)
    axes[1].set_title("Clip saturation fraction")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    fig.tight_layout()
    path = out_dir / "gate_activation_and_saturation.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    plot_paths.append(str(path))
    return plot_paths


def format_float(value: Any, digits: int = 4) -> str:
    try:
        numeric = float(value)
    except Exception:
        return str(value)
    if np.isnan(numeric):
        return "nan"
    return f"{numeric:.{digits}f}"


def write_markdown(
    path: Path,
    *,
    checkpoints: list[Path],
    cache: Path,
    split: str,
    edge_summary: list[dict[str, Any]],
    gate_summary: list[dict[str, Any]],
    corr_summary: list[dict[str, Any]],
    plots: list[str],
) -> None:
    lines: list[str] = [
        "# KAN Gate-Spline Interpretability",
        "",
        f"Cache: `{cache}`",
        f"Split: `{split}`",
        f"Checkpoints: {len(checkpoints)}",
        "",
        "This analysis is evaluation-only. It uses saved clipped KAN-GRU checkpoints and does not retrain models.",
        "",
        "## What Was Measured",
        "",
        "- Learned KAN edge functions for physical inputs in the update, reset, and candidate KAN layers.",
        "- Mean edge-function standard deviation across the spline grid as a simple importance score.",
        "- Runtime gate activation distributions and clipping/saturation fractions on held-out windows.",
        "- Correlations between mean gate activations and normalized physical inputs.",
        "",
        "## Physical-Input Spline Importance",
        "",
        "| Gate | Input | Edge std mean | Centered abs mean | Scale base abs | Scale spline abs |",
        "|---|---|---:|---:|---:|---:|",
    ]
    physical_rows = [row for row in edge_summary if row["input_type"] == "physical"]
    physical_rows.sort(key=lambda row: (row["gate"], -float(row["edge_std_mean_mean"])))
    for row in physical_rows:
        lines.append(
            f"| `{row['gate']}` | `{row['input']}` | "
            f"{format_float(row['edge_std_mean_mean'])} +/- {format_float(row['edge_std_mean_std'])} | "
            f"{format_float(row['edge_centered_abs_mean_mean'])} | "
            f"{format_float(row['scale_base_abs_mean_mean'])} | "
            f"{format_float(row['scale_sp_abs_mean_mean'])} |"
        )

    lines.extend(
        [
            "",
            "## Hidden-State Versus Physical-Input Contribution",
            "",
            "| Gate | Input group | Edge std mean | Edge count |",
            "|---|---|---:|---:|",
        ]
    )
    grouped_rows = [row for row in edge_summary if row["input"] == "hidden_state_mean"]
    grouped_rows.extend([row for row in edge_summary if row["input_type"] == "physical"])
    grouped_rows.sort(key=lambda row: (row["gate"], row["input_type"], row["input"]))
    for row in grouped_rows:
        lines.append(
            f"| `{row['gate']}` | `{row['input']}` | "
            f"{format_float(row['edge_std_mean_mean'])} +/- {format_float(row['edge_std_mean_std'])} | "
            f"{int(row['edge_count_mean'])} |"
        )

    lines.extend(
        [
            "",
            "## Runtime Gate Statistics",
            "",
            "| Gate | Mean | Std | P05 | P50 | P95 | Low clip | High clip |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(gate_summary, key=lambda item: item["gate"]):
        lines.append(
            f"| `{row['gate']}` | {format_float(row['mean_mean'])} +/- {format_float(row['mean_std'])} | "
            f"{format_float(row['std_mean'])} | {format_float(row['p05_mean'])} | "
            f"{format_float(row['p50_mean'])} | {format_float(row['p95_mean'])} | "
            f"{format_float(row['clip_low_fraction_mean'])} | {format_float(row['clip_high_fraction_mean'])} |"
        )

    lines.extend(
        [
            "",
            "## Strongest Gate/Input Correlations",
            "",
            "| Gate | Feature | Correlation |",
            "|---|---|---:|",
        ]
    )
    corr_rows = sorted(
        corr_summary,
        key=lambda row: abs(float(row["corr_gate_mean_with_feature_mean"])),
        reverse=True,
    )
    for row in corr_rows[:12]:
        lines.append(
            f"| `{row['gate']}` | `{row['feature']}` | "
            f"{format_float(row['corr_gate_mean_with_feature_mean'])} +/- "
            f"{format_float(row['corr_gate_mean_with_feature_std'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The KAN gates are interpretable as learned univariate edge functions, but the recurrent hidden-state inputs also carry substantial edge variation. Therefore the result should be framed as a functional diagnostic, not as a direct physics extraction.",
            "",
            "A manuscript-safe statement is:",
            "",
            "```text",
            "Inspection of the learned KAN gate splines shows that the clipped KAN-GRU relies on voltage-history features and recurrent hidden state jointly. The physical-input edge functions provide a compact diagnostic of how voltage, voltage lag, voltage increment, and sweep direction enter the learned update/reset/candidate gates, while saturation statistics show whether clipping creates hard switching behavior.",
            "```",
            "",
            "## Plot Files",
            "",
        ]
    )
    for plot in plots:
        lines.append(f"- `{plot}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("data/processed/sdc_state_splits.npz"))
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--grid-points", type=int, default=401)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", type=Path, default=Path("results/interpretability/kan_gate_splines_sdc"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_edge_rows: list[dict[str, Any]] = []
    all_gate_rows: list[dict[str, Any]] = []
    all_corr_rows: list[dict[str, Any]] = []
    grid = torch.linspace(-1.0, 1.0, args.grid_points)
    feature_names: list[str] | None = None
    checkpoint_summaries: list[dict[str, Any]] = []

    for checkpoint_path in args.checkpoints:
        payload = load_checkpoint(checkpoint_path, device)
        variant = variant_from_checkpoint(payload)
        feature_mode = str(payload.get("feature_mode", "no_current"))
        names = list(payload.get("feature_names", feature_names_for_mode(feature_mode)))
        feature_names = names

        dataset = WindowedCurveDataset(
            args.cache,
            args.split,
            window=args.window,
            stride=args.stride,
            feature_mode=feature_mode,
            max_windows=args.max_windows if args.max_windows > 0 else None,
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        model = build_sequence_model(variant, input_dim=dataset.feature_dim).to(device)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        if not isinstance(model, ClippedKANGRURegressor):
            raise TypeError(f"this script currently expects ClippedKANGRURegressor, got {type(model)}")

        seed = int(payload.get("seed", -1))
        hidden_dim = int(model.hidden_dim)
        z_indices = np.arange(0, hidden_dim)
        r_indices = np.arange(hidden_dim, 2 * hidden_dim)
        n_indices = np.arange(0, hidden_dim)
        all_edge_rows.extend(
            summarize_edges(
                seed=seed,
                layer=model.cell.gates,
                layer_name="gates",
                gate_name="update_z_raw",
                output_indices=z_indices,
                feature_names=names,
                hidden_dim=hidden_dim,
                grid=grid,
            )
        )
        all_edge_rows.extend(
            summarize_edges(
                seed=seed,
                layer=model.cell.gates,
                layer_name="gates",
                gate_name="reset_r_raw",
                output_indices=r_indices,
                feature_names=names,
                hidden_dim=hidden_dim,
                grid=grid,
            )
        )
        all_edge_rows.extend(
            summarize_edges(
                seed=seed,
                layer=model.cell.candidate,
                layer_name="candidate",
                gate_name="candidate_n_raw",
                output_indices=n_indices,
                feature_names=names,
                hidden_dim=hidden_dim,
                grid=grid,
            )
        )
        gate_rows, corr_rows, _arrays = runtime_gate_stats(
            model=model,
            loader=loader,
            device=device,
            seed=seed,
            feature_names=names,
        )
        all_gate_rows.extend(gate_rows)
        all_corr_rows.extend(corr_rows)
        checkpoint_summaries.append(
            {
                "checkpoint_path": str(checkpoint_path),
                "seed": seed,
                "variant": variant.name,
                "feature_mode": feature_mode,
                "feature_names": names,
                "split": args.split,
                "windows": len(dataset),
                "variant_config": asdict(variant),
            }
        )

    edge_summary = aggregate_rows(
        all_edge_rows,
        keys=["layer", "gate", "input", "input_type"],
        numeric_fields=[
            "edge_count",
            "edge_abs_mean",
            "edge_centered_abs_mean",
            "edge_std_mean",
            "edge_range_mean",
            "scale_base_abs_mean",
            "scale_sp_abs_mean",
        ],
    )
    gate_summary = aggregate_rows(
        all_gate_rows,
        keys=["gate"],
        numeric_fields=["mean", "std", "p05", "p50", "p95", "clip_low_fraction", "clip_high_fraction"],
    )
    corr_summary = aggregate_rows(
        all_corr_rows,
        keys=["gate", "feature"],
        numeric_fields=["corr_gate_mean_with_feature"],
    )

    write_csv(args.out_dir / "edge_importance_per_seed.csv", all_edge_rows)
    write_csv(args.out_dir / "edge_importance_summary.csv", edge_summary)
    write_csv(args.out_dir / "gate_activation_per_seed.csv", all_gate_rows)
    write_csv(args.out_dir / "gate_activation_summary.csv", gate_summary)
    write_csv(args.out_dir / "gate_feature_correlations_per_seed.csv", all_corr_rows)
    write_csv(args.out_dir / "gate_feature_correlations_summary.csv", corr_summary)
    plots = make_plots(
        out_dir=args.out_dir,
        edge_summary=edge_summary,
        gate_summary=gate_summary,
        feature_names=feature_names or [],
    )
    write_markdown(
        args.out_dir / "kan_gate_spline_interpretability.md",
        checkpoints=args.checkpoints,
        cache=args.cache,
        split=args.split,
        edge_summary=edge_summary,
        gate_summary=gate_summary,
        corr_summary=corr_summary,
        plots=plots,
    )
    (args.out_dir / "analysis_manifest.json").write_text(
        json.dumps(
            {
                "cache": str(args.cache),
                "split": args.split,
                "window": args.window,
                "stride": args.stride,
                "batch_size": args.batch_size,
                "grid_points": args.grid_points,
                "checkpoints": checkpoint_summaries,
                "plots": plots,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "checkpoints": len(args.checkpoints),
                "edge_rows": len(all_edge_rows),
                "gate_rows": len(all_gate_rows),
                "corr_rows": len(all_corr_rows),
                "plots": plots,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
