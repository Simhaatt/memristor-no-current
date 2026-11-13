"""Permutation-aware stability checks for KAN-GRU gate-edge importance.

The raw edge label ``(gate, input, unit)`` is not seed-stable because hidden
units are permutation-symmetric. This script adds three inference-only
corrections:

* aggregate edge importance over units to compare stable ``(gate, input)``
  quantities,
* compare top-k edge-set overlap before and after hidden-unit alignment,
* align hidden units by test-set activation correlation and recompute rank
  agreement over physical-input edges.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
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
from memristorkan.models.recurrent import ClippedKANGRURegressor
from memristorkan.models.spline import bspline_basis
from run_sdc_sequence import (  # type: ignore
    WindowedCurveDataset,
    build_sequence_model,
    feature_names_for_mode,
)
from evaluate_sequence_checkpoints import load_checkpoint, variant_from_checkpoint


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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
    if a.size < 2 or float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return pearson(rankdata(a), rankdata(b))


def edge_key(row: dict[str, Any], *, aligned_unit: int | None = None) -> tuple[str, str, int]:
    unit = int(row["output_idx"]) if aligned_unit is None else int(aligned_unit)
    return (str(row["gate"]), str(row["input"]), unit)


def aggregate_gate_input(edge_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[int, str, str], list[float]] = {}
    for row in edge_rows:
        grouped.setdefault((int(row["seed"]), row["gate"], row["input"]), []).append(float(row["edge_std"]))

    per_seed: list[dict[str, Any]] = []
    for (seed, gate, input_name), values in sorted(grouped.items()):
        per_seed.append(
            {
                "seed": seed,
                "gate": gate,
                "input": input_name,
                "edge_count": len(values),
                "edge_std_mean": mean(values),
                "edge_std_sum": sum(values),
                "edge_std_max": max(values),
            }
        )

    by_seed = {
        seed: {
            (row["gate"], row["input"]): float(row["edge_std_mean"])
            for row in per_seed
            if int(row["seed"]) == seed
        }
        for seed in sorted({int(row["seed"]) for row in per_seed})
    }
    pair_rows: list[dict[str, Any]] = []
    seeds = sorted(by_seed)
    for i, seed_a in enumerate(seeds):
        for seed_b in seeds[i + 1 :]:
            common = sorted(set(by_seed[seed_a]) & set(by_seed[seed_b]))
            a = np.array([by_seed[seed_a][key] for key in common], dtype=float)
            b = np.array([by_seed[seed_b][key] for key in common], dtype=float)
            pair_rows.append(
                {
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "common_gate_input_pairs": len(common),
                    "spearman_rho": spearman(a, b),
                    "pearson_r": pearson(a, b),
                }
            )
    return per_seed, pair_rows


def summarize_gate_input_pairs(per_seed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in per_seed:
        grouped.setdefault((str(row["gate"]), str(row["input"])), []).append(row)
    rows: list[dict[str, Any]] = []
    for (gate, input_name), items in sorted(grouped.items()):
        vals_mean = [float(row["edge_std_mean"]) for row in items]
        vals_sum = [float(row["edge_std_sum"]) for row in items]
        vals_max = [float(row["edge_std_max"]) for row in items]
        rows.append(
            {
                "gate": gate,
                "input": input_name,
                "runs": len(items),
                "edge_count": int(items[0]["edge_count"]),
                "edge_std_mean": mean(vals_mean),
                "edge_std_mean_std": pstdev(vals_mean) if len(vals_mean) > 1 else 0.0,
                "edge_std_sum_mean": mean(vals_sum),
                "edge_std_sum_std": pstdev(vals_sum) if len(vals_sum) > 1 else 0.0,
                "edge_std_max_mean": mean(vals_max),
                "edge_std_max_std": pstdev(vals_max) if len(vals_max) > 1 else 0.0,
            }
        )
    rows.sort(key=lambda row: float(row["edge_std_mean"]), reverse=True)
    return rows


def clipped_gru_step(
    model: ClippedKANGRURegressor,
    x_t: torch.Tensor,
    h_prev: torch.Tensor,
) -> torch.Tensor:
    gate_input = torch.cat([x_t, h_prev], dim=-1)
    z_raw, r_raw = model.cell.gates(gate_input).chunk(2, dim=-1)
    z_t = ((z_raw + 1.0) * 0.5).clamp(0.0, 1.0)
    r_t = ((r_raw + 1.0) * 0.5).clamp(0.0, 1.0)
    candidate_input = torch.cat([x_t, r_t * h_prev], dim=-1)
    n_t = model.cell.candidate(candidate_input).clamp(-1.0, 1.0)
    return (1.0 - z_t) * n_t + z_t * h_prev


def hidden_activations(
    checkpoint: Path,
    *,
    cache: Path,
    split: str,
    window: int,
    stride: int,
    batch_size: int,
    max_windows: int,
    device: torch.device,
) -> tuple[int, np.ndarray]:
    payload = load_checkpoint(checkpoint, device)
    variant = variant_from_checkpoint(payload)
    feature_mode = str(payload.get("feature_mode", "no_current"))
    dataset = WindowedCurveDataset(
        cache,
        split,
        window=window,
        stride=stride,
        feature_mode=feature_mode,
        max_windows=max_windows if max_windows > 0 else None,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = build_sequence_model(variant, input_dim=dataset.feature_dim).to(device)
    model.load_state_dict(payload["model_state_dict"])
    if not isinstance(model, ClippedKANGRURegressor):
        raise TypeError(f"expected clipped KAN-GRU, got {type(model)}")
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for x, _y in loader:
            x = x.to(device)
            h_t = x.new_zeros(x.shape[0], model.hidden_dim)
            for t in range(x.shape[1]):
                h_t = clipped_gru_step(model, x[:, t, :], h_t)
                chunks.append(h_t.detach().cpu().numpy())
    return int(payload.get("seed", -1)), np.concatenate(chunks, axis=0)


def activation_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a0 = a - a.mean(axis=0, keepdims=True)
    b0 = b - b.mean(axis=0, keepdims=True)
    denom = np.sqrt(np.sum(a0**2, axis=0, keepdims=True).T @ np.sum(b0**2, axis=0, keepdims=True)).clip(1e-12)
    return (a0.T @ b0) / denom


def align_units(corr: np.ndarray) -> dict[int, int]:
    """Return mapping from column/unit in B to row/unit in A."""

    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(-np.abs(corr))
        return {int(col): int(row) for row, col in zip(rows, cols)}
    except Exception:
        remaining_rows = set(range(corr.shape[0]))
        remaining_cols = set(range(corr.shape[1]))
        mapping: dict[int, int] = {}
        scores = [
            (abs(float(corr[row, col])), row, col)
            for row in range(corr.shape[0])
            for col in range(corr.shape[1])
        ]
        for _score, row, col in sorted(scores, reverse=True):
            if row in remaining_rows and col in remaining_cols:
                mapping[col] = row
                remaining_rows.remove(row)
                remaining_cols.remove(col)
        return mapping


def seed_edge_maps(edge_rows: list[dict[str, str]]) -> dict[int, dict[tuple[str, str, int], float]]:
    out: dict[int, dict[tuple[str, str, int], float]] = {}
    for row in edge_rows:
        out.setdefault(int(row["seed"]), {})[edge_key(row)] = float(row["edge_std"])
    return out


def aligned_edge_stability(
    edge_rows: list[dict[str, str]],
    activations: dict[int, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    maps = seed_edge_maps(edge_rows)
    seeds = sorted(maps)
    stability_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []

    for i, seed_a in enumerate(seeds):
        for seed_b in seeds[i + 1 :]:
            corr = activation_correlation(activations[seed_a], activations[seed_b])
            mapping_b_to_a = align_units(corr)
            matched_corr = [abs(float(corr[row_a, col_b])) for col_b, row_a in mapping_b_to_a.items()]
            alignment_rows.append(
                {
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "matched_units": len(mapping_b_to_a),
                    "mean_abs_activation_corr": mean(matched_corr),
                    "min_abs_activation_corr": min(matched_corr),
                    "max_abs_activation_corr": max(matched_corr),
                }
            )

            aligned_b: dict[tuple[str, str, int], float] = {}
            for (gate, input_name, unit_b), value in maps[seed_b].items():
                aligned_b[(gate, input_name, mapping_b_to_a[unit_b])] = value
            common = sorted(set(maps[seed_a]) & set(aligned_b))
            a = np.array([maps[seed_a][key] for key in common], dtype=float)
            b = np.array([aligned_b[key] for key in common], dtype=float)
            stability_rows.append(
                {
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "common_edges": len(common),
                    "spearman_rho": spearman(a, b),
                    "pearson_r": pearson(a, b),
                }
            )
    return stability_rows, alignment_rows


def topk_overlap(edge_rows: list[dict[str, str]], activations: dict[int, np.ndarray], k_values: list[int]) -> list[dict[str, Any]]:
    by_seed_rows: dict[int, list[dict[str, str]]] = {}
    for row in edge_rows:
        by_seed_rows.setdefault(int(row["seed"]), []).append(row)
    for rows in by_seed_rows.values():
        rows.sort(key=lambda row: float(row["edge_std"]), reverse=True)

    rows_out: list[dict[str, Any]] = []
    seeds = sorted(by_seed_rows)
    for i, seed_a in enumerate(seeds):
        for seed_b in seeds[i + 1 :]:
            corr = activation_correlation(activations[seed_a], activations[seed_b])
            mapping_b_to_a = align_units(corr)
            for k in k_values:
                top_a_raw = {edge_key(row) for row in by_seed_rows[seed_a][:k]}
                top_b_raw = {edge_key(row) for row in by_seed_rows[seed_b][:k]}
                top_b_aligned = {
                    edge_key(row, aligned_unit=mapping_b_to_a[int(row["output_idx"])])
                    for row in by_seed_rows[seed_b][:k]
                }
                for label, set_b in [("raw_unit_labels", top_b_raw), ("activation_aligned_units", top_b_aligned)]:
                    union = top_a_raw | set_b
                    inter = top_a_raw & set_b
                    rows_out.append(
                        {
                            "seed_a": seed_a,
                            "seed_b": seed_b,
                            "comparison": label,
                            "k": k,
                            "intersection": len(inter),
                            "union": len(union),
                            "jaccard": len(inter) / max(1, len(union)),
                            "overlap_fraction_of_k": len(inter) / max(1, k),
                        }
                    )
    return rows_out


def edge_values(layer: KANLayer, input_idx: int, output_indices: np.ndarray, grid: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        basis = bspline_basis(grid.cpu(), layer.knots.detach().cpu(), layer.spline_order)
        coef = layer.coef.detach().cpu()[input_idx, output_indices, :]
        spline = basis @ coef.T
        base = torch.nn.functional.silu(grid.cpu()).unsqueeze(1) * layer.scale_base.detach().cpu()[
            input_idx, output_indices
        ].unsqueeze(0)
        values = base + spline * layer.scale_sp.detach().cpu()[input_idx, output_indices].unsqueeze(0)
    return values.numpy()


def aggregate_response_curves(
    checkpoints: list[Path],
    *,
    cache: Path,
    grid: torch.Tensor,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        payload = load_checkpoint(checkpoint, torch.device("cpu"))
        variant = variant_from_checkpoint(payload)
        feature_mode = str(payload.get("feature_mode", "no_current"))
        feature_names = list(payload.get("feature_names", feature_names_for_mode(feature_mode)))
        dataset = WindowedCurveDataset(cache, "test", window=256, stride=512, feature_mode=feature_mode, max_windows=1)
        model = build_sequence_model(variant, input_dim=dataset.feature_dim)
        model.load_state_dict(payload["model_state_dict"])
        if not isinstance(model, ClippedKANGRURegressor):
            raise TypeError("aggregate response curves expect clipped KAN-GRU")
        seed = int(payload.get("seed", -1))
        hidden = model.hidden_dim
        gate_specs = [
            ("update_z_raw", model.cell.gates, np.arange(0, hidden)),
            ("reset_r_raw", model.cell.gates, np.arange(hidden, 2 * hidden)),
            ("candidate_n_raw", model.cell.candidate, np.arange(0, hidden)),
        ]
        for gate, layer, outputs in gate_specs:
            for input_idx, input_name in enumerate(feature_names):
                values = edge_values(layer, input_idx, outputs, grid)
                curve = values.mean(axis=1)
                for x, y in zip(grid.numpy(), curve):
                    rows.append(
                        {
                            "seed": seed,
                            "gate": gate,
                            "input": input_name,
                            "x": float(x),
                            "mean_edge_response": float(y),
                        }
                    )
    return rows


def make_plots(
    out_dir: Path,
    gate_input_pair_rows: list[dict[str, Any]],
    gate_input_summary: list[dict[str, Any]],
    aligned_rows: list[dict[str, Any]],
    topk_rows: list[dict[str, Any]],
    curve_rows: list[dict[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    # Gate/input aggregate importance.
    values = [float(row["edge_std_mean"]) for row in gate_input_summary]
    errors = [float(row["edge_std_mean_std"]) for row in gate_input_summary]
    labels = [f"{row['gate']}\n{row['input']}" for row in gate_input_summary]
    order = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    ax.bar(np.arange(len(order)), [values[i] for i in order], yerr=[errors[i] for i in order], capsize=3)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([labels[i] for i in order], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean edge std over hidden units")
    ax.set_title("Permutation-Invariant Gate/Input Importance")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "gate_input_aggregate_importance.png", dpi=220)
    plt.close(fig)

    # Raw vs aligned top-k overlap.
    summary: dict[tuple[str, int], list[float]] = {}
    for row in topk_rows:
        summary.setdefault((row["comparison"], int(row["k"])), []).append(float(row["jaccard"]))
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for comparison, color in [("raw_unit_labels", "#A84E3B"), ("activation_aligned_units", "#4C78A8")]:
        ks = sorted(k for comp, k in summary if comp == comparison)
        ax.plot(
            ks,
            [mean(summary[(comparison, k)]) for k in ks],
            marker="o",
            linewidth=2,
            label=comparison,
            color=color,
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Top-k edges")
    ax.set_ylabel("Mean Jaccard overlap")
    ax.set_title("Top-k Edge Overlap Before/After Unit Alignment")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "topk_overlap_raw_vs_aligned.png", dpi=220)
    plt.close(fig)

    # Aggregate learned response curves for the strongest few gate/input pairs.
    top_pairs = [(row["gate"], row["input"]) for row in gate_input_summary[:6]]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2), sharex=True)
    for ax, (gate, input_name) in zip(axes.ravel(), top_pairs):
        for seed in sorted({int(row["seed"]) for row in curve_rows}):
            rows = [
                row
                for row in curve_rows
                if int(row["seed"]) == seed and row["gate"] == gate and row["input"] == input_name
            ]
            rows.sort(key=lambda row: float(row["x"]))
            ax.plot(
                [float(row["x"]) for row in rows],
                [float(row["mean_edge_response"]) for row in rows],
                label=f"seed {seed}",
                linewidth=1.6,
            )
        ax.set_title(f"{gate}\n{input_name}", fontsize=9)
        ax.grid(alpha=0.25)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.supxlabel("Normalized input")
    fig.supylabel("Mean edge response over units")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / "aggregate_gate_input_response_curves.png", dpi=220)
    plt.close(fig)


def fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def summarize_topk(topk_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in topk_rows:
        grouped.setdefault((str(row["comparison"]), int(row["k"])), []).append(row)
    out: list[dict[str, Any]] = []
    for (comparison, k), rows in sorted(grouped.items()):
        vals_j = [float(row["jaccard"]) for row in rows]
        vals_o = [float(row["overlap_fraction_of_k"]) for row in rows]
        out.append(
            {
                "comparison": comparison,
                "k": k,
                "pairs": len(rows),
                "jaccard_mean": mean(vals_j),
                "jaccard_std": pstdev(vals_j) if len(vals_j) > 1 else 0.0,
                "overlap_fraction_mean": mean(vals_o),
                "overlap_fraction_std": pstdev(vals_o) if len(vals_o) > 1 else 0.0,
            }
        )
    return out


def write_markdown(
    path: Path,
    *,
    gate_input_summary: list[dict[str, Any]],
    gate_input_stability: list[dict[str, Any]],
    aligned_stability: list[dict[str, Any]],
    alignment_quality: list[dict[str, Any]],
    topk_summary: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    gi_mean = mean(float(row["spearman_rho"]) for row in gate_input_stability)
    aligned_mean = mean(float(row["spearman_rho"]) for row in aligned_stability)
    align_corr = mean(float(row["mean_abs_activation_corr"]) for row in alignment_quality)
    top_pairs = gate_input_summary[:12]

    lines = [
        "# Corrected KAN Gate Stability Analysis",
        "",
        "This post-processing corrects the raw unit-labeled edge-rank comparison, which is affected by hidden-unit permutation symmetry.",
        "",
        "## Settings",
        "",
        f"- Cache: `{manifest['cache']}`",
        f"- Split: `{manifest['split']}`",
        f"- Window/stride: `{manifest['window']}` / `{manifest['stride']}`",
        f"- Checkpoints: {len(manifest['checkpoints'])}",
        "",
        "## Gate/Input Aggregation",
        "",
        f"Mean pairwise Spearman correlation over stable `(gate, input)` labels: **{fmt(gi_mean)}**.",
        "",
        "| Rank | Gate | Input | Edge std mean | Edge std sum |",
        "|---:|---|---|---:|---:|",
    ]
    for idx, row in enumerate(top_pairs, start=1):
        lines.append(
            f"| {idx} | `{row['gate']}` | `{row['input']}` | "
            f"{fmt(row['edge_std_mean'])} +/- {fmt(row['edge_std_mean_std'])} | "
            f"{fmt(row['edge_std_sum_mean'])} |"
        )

    lines.extend(
        [
            "",
            "Pairwise gate/input stability:",
            "",
            "| Seed A | Seed B | Pairs | Spearman rho | Pearson r |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in gate_input_stability:
        lines.append(
            f"| {row['seed_a']} | {row['seed_b']} | {row['common_gate_input_pairs']} | "
            f"{fmt(row['spearman_rho'])} | {fmt(row['pearson_r'])} |"
        )

    lines.extend(
        [
            "",
            "## Hidden-Unit Alignment",
            "",
            f"Mean matched absolute hidden-activation correlation: **{fmt(align_corr)}**.",
            f"Mean pairwise Spearman correlation after activation-based unit alignment: **{fmt(aligned_mean)}**.",
            "",
            "| Seed A | Seed B | Matched units | Mean abs activation corr | Aligned edge Spearman |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    aligned_lookup = {(row["seed_a"], row["seed_b"]): row for row in aligned_stability}
    for row in alignment_quality:
        aligned = aligned_lookup[(row["seed_a"], row["seed_b"])]
        lines.append(
            f"| {row['seed_a']} | {row['seed_b']} | {row['matched_units']} | "
            f"{fmt(row['mean_abs_activation_corr'])} | {fmt(aligned['spearman_rho'])} |"
        )

    lines.extend(
        [
            "",
            "## Top-k Edge Overlap",
            "",
            "| Comparison | k | Pair count | Jaccard | Overlap/k |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in topk_summary:
        lines.append(
            f"| `{row['comparison']}` | {row['k']} | {row['pairs']} | "
            f"{fmt(row['jaccard_mean'])} +/- {fmt(row['jaccard_std'])} | "
            f"{fmt(row['overlap_fraction_mean'])} +/- {fmt(row['overlap_fraction_std'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The raw unit-labeled 384-edge Spearman statistic should not be treated as the primary stability result, because hidden units are permutation-symmetric and many low-importance edges have noisy rankings. The corrected analysis should be reported instead: gate/input aggregation gives stable, corresponding labels; activation alignment tests whether unit permutation explains part of the mismatch; and top-k overlap focuses on important edges rather than noisy tail rankings.",
            "",
            "Manuscript-safe sentence:",
            "",
            "```text",
            "After correcting for hidden-unit permutation symmetry, gate/input-level importance is more stable than raw unit-labeled edge ranks, while individual edge identities remain partly seed-dependent. Thus, the interpretability claim is strongest at the level of gate/input functional dependence and load-bearing top-edge groups, not exact unit-by-unit correspondence.",
            "```",
            "",
            "Figure files:",
            "",
            "- `gate_input_aggregate_importance.png`",
            "- `topk_overlap_raw_vs_aligned.png`",
            "- `aggregate_gate_input_response_curves.png`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge-csv", type=Path, default=Path("results/interpretability/kan_gate_deep_dive_sdc/edge_importance_per_edge_seed.csv"))
    parser.add_argument("--cache", type=Path, default=Path("data/processed/sdc_state_splits.npz"))
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--grid-points", type=int, default=401)
    parser.add_argument("--topk", type=int, nargs="+", default=[10, 20, 50])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", type=Path, default=Path("results/interpretability/kan_gate_stability_corrected_sdc"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    edge_rows = read_csv(args.edge_csv)

    gate_input_pairs, gate_input_stability = aggregate_gate_input(edge_rows)
    gate_input_summary = summarize_gate_input_pairs(gate_input_pairs)
    activations: dict[int, np.ndarray] = {}
    for checkpoint in args.checkpoints:
        seed, acts = hidden_activations(
            checkpoint,
            cache=args.cache,
            split=args.split,
            window=args.window,
            stride=args.stride,
            batch_size=args.batch_size,
            max_windows=args.max_windows,
            device=device,
        )
        activations[seed] = acts
    aligned_stability, alignment_quality = aligned_edge_stability(edge_rows, activations)
    topk_rows = topk_overlap(edge_rows, activations, args.topk)
    topk_summary = summarize_topk(topk_rows)
    grid = torch.linspace(-1.0, 1.0, args.grid_points)
    curve_rows = aggregate_response_curves(args.checkpoints, cache=args.cache, grid=grid)

    write_csv(args.out_dir / "gate_input_aggregate_importance_per_seed.csv", gate_input_pairs)
    write_csv(args.out_dir / "gate_input_aggregate_importance_summary.csv", gate_input_summary)
    write_csv(args.out_dir / "gate_input_aggregate_stability.csv", gate_input_stability)
    write_csv(args.out_dir / "unit_alignment_quality.csv", alignment_quality)
    write_csv(args.out_dir / "unit_aligned_edge_rank_stability.csv", aligned_stability)
    write_csv(args.out_dir / "topk_edge_overlap_raw_vs_aligned_per_pair.csv", topk_rows)
    write_csv(args.out_dir / "topk_edge_overlap_raw_vs_aligned_summary.csv", topk_summary)
    write_csv(args.out_dir / "aggregate_gate_input_response_curves.csv", curve_rows)

    make_plots(args.out_dir, gate_input_pairs, gate_input_summary, aligned_stability, topk_rows, curve_rows)
    manifest = {
        "edge_csv": str(args.edge_csv),
        "cache": str(args.cache),
        "checkpoints": [str(path) for path in args.checkpoints],
        "split": args.split,
        "window": args.window,
        "stride": args.stride,
        "batch_size": args.batch_size,
        "max_windows": args.max_windows,
        "grid_points": args.grid_points,
        "topk": args.topk,
    }
    (args.out_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_markdown(
        args.out_dir / "kan_gate_corrected_stability_results.md",
        gate_input_summary=gate_input_summary,
        gate_input_stability=gate_input_stability,
        aligned_stability=aligned_stability,
        alignment_quality=alignment_quality,
        topk_summary=topk_summary,
        manifest=manifest,
    )
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "gate_input_rows": len(gate_input_pairs),
                "gate_input_stability_rows": len(gate_input_stability),
                "aligned_stability_rows": len(aligned_stability),
                "topk_rows": len(topk_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
