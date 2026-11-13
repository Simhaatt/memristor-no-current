"""Evaluate load-bearing gate/input interventions on trained sequence models.

This script complements descriptive KAN spline inspection with interventions:

* KAN-edge ablations zero selected learned edge functions in clipped KAN-GRU.
* Feature-zero ablations replace one normalized input feature with its train
  mean value, i.e. zero in normalized coordinates, for both KAN-GRU and GRU.

The output is a drop-from-baseline table over saved checkpoints. No training is
performed.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from dataclasses import asdict
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
from memristorkan.models.recurrent import ClippedKANGRURegressor, GRURegressor
from run_sdc_sequence import (  # type: ignore
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
        raise ValueError("checkpoint does not contain a variant dictionary")
    allowed = {"name", "model_name", "hidden_dim", "grid_size", "gauge_every_steps", "moment_mode"}
    return SequenceVariant(**{key: value for key, value in variant.items() if key in allowed})


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


def evaluate_model(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds.append(model(x).cpu())
            targets.append(y.cpu())
    return sequence_metrics(torch.cat(preds, dim=0), torch.cat(targets, dim=0))


def zero_kan_edges(model: ClippedKANGRURegressor, specs: list[tuple[str, int, str]]) -> None:
    """Zero selected KAN edge functions in place.

    specs entries are ``(layer_name, physical_input_index, gate_name)`` where
    gate_name is one of ``update``, ``reset``, or ``candidate``.
    """

    hidden = model.hidden_dim
    with torch.no_grad():
        for layer_name, input_idx, gate_name in specs:
            if layer_name == "gates":
                layer = model.cell.gates
                if gate_name == "update":
                    out_slice = slice(0, hidden)
                elif gate_name == "reset":
                    out_slice = slice(hidden, 2 * hidden)
                else:
                    raise ValueError(f"bad gate for gates layer: {gate_name}")
            elif layer_name == "candidate":
                layer = model.cell.candidate
                if gate_name != "candidate":
                    raise ValueError(f"bad gate for candidate layer: {gate_name}")
                out_slice = slice(0, hidden)
            else:
                raise ValueError(f"unknown KAN layer: {layer_name}")
            layer.scale_base[input_idx, out_slice] = 0.0
            layer.scale_sp[input_idx, out_slice] = 0.0


def kan_interventions(feature_to_idx: dict[str, int]) -> dict[str, list[tuple[str, int, str]]]:
    return {
        "none": [],
        "kan_update_vtm1_zero_edges": [("gates", feature_to_idx["V_t_minus_1"], "update")],
        "kan_reset_dv_zero_edges": [("gates", feature_to_idx["dV_t"], "reset")],
        "kan_candidate_dv_zero_edges": [("candidate", feature_to_idx["dV_t"], "candidate")],
        "kan_sweep_all_gates_zero_edges": [
            ("gates", feature_to_idx["sweep_direction"], "update"),
            ("gates", feature_to_idx["sweep_direction"], "reset"),
            ("candidate", feature_to_idx["sweep_direction"], "candidate"),
        ],
        "kan_top_edges_zero": [
            ("gates", feature_to_idx["V_t_minus_1"], "update"),
            ("gates", feature_to_idx["dV_t"], "reset"),
            ("candidate", feature_to_idx["dV_t"], "candidate"),
        ],
        "kan_all_physical_edges_zero": [
            (layer, idx, gate)
            for idx in feature_to_idx.values()
            for layer, gate in [("gates", "update"), ("gates", "reset"), ("candidate", "candidate")]
        ],
    }


def feature_zero_interventions(feature_to_idx: dict[str, int]) -> dict[str, int | None]:
    return {
        "none": None,
        **{f"feature_zero_{name}": idx for name, idx in feature_to_idx.items()},
    }


def evaluate_checkpoint_group(
    *,
    label: str,
    checkpoint_paths: list[Path],
    cache: Path,
    split: str,
    window: int,
    stride: int,
    batch_size: int,
    device: torch.device,
    intervention_type: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for checkpoint_path in checkpoint_paths:
        payload = load_checkpoint(checkpoint_path, device)
        variant = variant_from_checkpoint(payload)
        feature_mode = str(payload.get("feature_mode", "no_current"))
        feature_names = list(payload.get("feature_names", feature_names_for_mode(feature_mode)))
        feature_to_idx = {name: idx for idx, name in enumerate(feature_names)}
        base_dataset = WindowedCurveDataset(
            cache,
            split,
            window=window,
            stride=stride,
            feature_mode=feature_mode,
        )
        seed = int(payload.get("seed", -1))
        base_model = build_sequence_model(variant, input_dim=base_dataset.feature_dim).to(device)
        base_model.load_state_dict(payload["model_state_dict"])
        base_model.eval()

        if intervention_type == "kan_edges":
            if not isinstance(base_model, ClippedKANGRURegressor):
                raise TypeError("kan_edges intervention requires clipped KAN-GRU checkpoints")
            interventions = kan_interventions(feature_to_idx)
            loader = DataLoader(base_dataset, batch_size=batch_size, shuffle=False)
            baseline_metrics: dict[str, float] | None = None
            for name, specs in interventions.items():
                model = copy.deepcopy(base_model).to(device)
                if specs:
                    zero_kan_edges(model, specs)
                metrics = evaluate_model(model, loader, device)
                if name == "none":
                    baseline_metrics = metrics
                assert baseline_metrics is not None
                rows.append(
                    {
                        "group": label,
                        "variant": variant.name,
                        "seed": seed,
                        "intervention_type": intervention_type,
                        "intervention": name,
                        "feature_mode": feature_mode,
                        "checkpoint_path": str(checkpoint_path),
                        "r2": metrics["r2"],
                        "rmse": metrics["rmse"],
                        "mae": metrics["mae"],
                        "final_state_mae": metrics["final_state_mae"],
                        "r2_drop": baseline_metrics["r2"] - metrics["r2"],
                        "rmse_increase": metrics["rmse"] - baseline_metrics["rmse"],
                    }
                )
        elif intervention_type == "feature_zero":
            interventions = feature_zero_interventions(feature_to_idx)
            baseline_metrics = None
            for name, feature_idx in interventions.items():
                dataset = FeatureZeroDataset(base_dataset, feature_idx)
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
                metrics = evaluate_model(base_model, loader, device)
                if name == "none":
                    baseline_metrics = metrics
                assert baseline_metrics is not None
                rows.append(
                    {
                        "group": label,
                        "variant": variant.name,
                        "seed": seed,
                        "intervention_type": intervention_type,
                        "intervention": name,
                        "feature_mode": feature_mode,
                        "checkpoint_path": str(checkpoint_path),
                        "r2": metrics["r2"],
                        "rmse": metrics["rmse"],
                        "mae": metrics["mae"],
                        "final_state_mae": metrics["final_state_mae"],
                        "r2_drop": baseline_metrics["r2"] - metrics["r2"],
                        "rmse_increase": metrics["rmse"] - baseline_metrics["rmse"],
                    }
                )
        else:
            raise ValueError(f"unknown intervention_type: {intervention_type}")
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["group"]),
            str(row["variant"]),
            str(row["intervention_type"]),
            str(row["intervention"]),
        )
        grouped.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for (group, variant, intervention_type, intervention), items in sorted(grouped.items()):
        def values(field: str) -> list[float]:
            return [float(item[field]) for item in items]

        out.append(
            {
                "group": group,
                "variant": variant,
                "intervention_type": intervention_type,
                "intervention": intervention,
                "runs": len(items),
                "r2_mean": mean(values("r2")),
                "r2_std": pstdev(values("r2")) if len(items) > 1 else 0.0,
                "rmse_mean": mean(values("rmse")),
                "rmse_std": pstdev(values("rmse")) if len(items) > 1 else 0.0,
                "final_state_mae_mean": mean(values("final_state_mae")),
                "r2_drop_mean": mean(values("r2_drop")),
                "r2_drop_std": pstdev(values("r2_drop")) if len(items) > 1 else 0.0,
                "rmse_increase_mean": mean(values("rmse_increase")),
            }
        )
    out.sort(key=lambda row: (row["group"], row["intervention_type"], -float(row["r2_drop_mean"])))
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float) -> str:
    return f"{float(value):.4f}"


def write_markdown(path: Path, summary_rows: list[dict[str, Any]], per_seed_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Gate And Feature Intervention Results",
        "",
        "These are evaluation-only interventions on saved checkpoints. No model was retrained.",
        "",
        "## KAN Edge Ablations",
        "",
        "| Intervention | Runs | R2 | R2 drop | RMSE | RMSE increase |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in [r for r in summary_rows if r["intervention_type"] == "kan_edges"]:
        lines.append(
            f"| `{row['intervention']}` | {row['runs']} | "
            f"{fmt(row['r2_mean'])} +/- {fmt(row['r2_std'])} | "
            f"{fmt(row['r2_drop_mean'])} +/- {fmt(row['r2_drop_std'])} | "
            f"{fmt(row['rmse_mean'])} | {fmt(row['rmse_increase_mean'])} |"
        )

    lines.extend(
        [
            "",
            "## Feature-Zero Interventions",
            "",
            "| Model | Intervention | Runs | R2 | R2 drop | RMSE | RMSE increase |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in [r for r in summary_rows if r["intervention_type"] == "feature_zero"]:
        lines.append(
            f"| `{row['variant']}` | `{row['intervention']}` | {row['runs']} | "
            f"{fmt(row['r2_mean'])} +/- {fmt(row['r2_std'])} | "
            f"{fmt(row['r2_drop_mean'])} +/- {fmt(row['r2_drop_std'])} | "
            f"{fmt(row['rmse_mean'])} | {fmt(row['rmse_increase_mean'])} |"
        )

    top_kan = [
        row for row in summary_rows if row["intervention_type"] == "kan_edges" and row["intervention"] != "none"
    ][:5]
    top_feature = [
        row for row in summary_rows if row["intervention_type"] == "feature_zero" and row["intervention"] != "none"
    ][:8]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "KAN edge ablation turns the descriptive spline ranking into a load-bearing test. Large R2 drops mean that the corresponding learned spline edges are not merely visually active; the checkpoint depends on them at inference time.",
            "",
            "Top KAN edge drops:",
            "",
        ]
    )
    for row in top_kan:
        lines.append(f"- `{row['intervention']}`: R2 drop {fmt(row['r2_drop_mean'])}")
    lines.extend(["", "Top feature-zero drops:", ""])
    for row in top_feature:
        lines.append(f"- `{row['variant']}` / `{row['intervention']}`: R2 drop {fmt(row['r2_drop_mean'])}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("data/processed/sdc_state_splits.npz"))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--kan-checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--gru-checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/interpretability/gate_interventions_sdc"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    rows: list[dict[str, Any]] = []
    rows.extend(
        evaluate_checkpoint_group(
            label="clipped_kan_gru_h32",
            checkpoint_paths=args.kan_checkpoints,
            cache=args.cache,
            split=args.split,
            window=args.window,
            stride=args.stride,
            batch_size=args.batch_size,
            device=device,
            intervention_type="kan_edges",
        )
    )
    rows.extend(
        evaluate_checkpoint_group(
            label="clipped_kan_gru_h32",
            checkpoint_paths=args.kan_checkpoints,
            cache=args.cache,
            split=args.split,
            window=args.window,
            stride=args.stride,
            batch_size=args.batch_size,
            device=device,
            intervention_type="feature_zero",
        )
    )
    rows.extend(
        evaluate_checkpoint_group(
            label="gru_h116",
            checkpoint_paths=args.gru_checkpoints,
            cache=args.cache,
            split=args.split,
            window=args.window,
            stride=args.stride,
            batch_size=args.batch_size,
            device=device,
            intervention_type="feature_zero",
        )
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = summarize(rows)
    write_csv(args.out_dir / "gate_interventions_per_seed.csv", rows)
    write_csv(args.out_dir / "gate_interventions_summary.csv", summary_rows)
    write_markdown(args.out_dir / "gate_intervention_results.md", summary_rows, rows)
    (args.out_dir / "analysis_manifest.json").write_text(
        json.dumps(
            {
                "cache": str(args.cache),
                "split": args.split,
                "window": args.window,
                "stride": args.stride,
                "batch_size": args.batch_size,
                "kan_checkpoints": [str(path) for path in args.kan_checkpoints],
                "gru_checkpoints": [str(path) for path in args.gru_checkpoints],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"out_dir": str(args.out_dir), "rows": len(rows), "summary_rows": len(summary_rows)}, indent=2))


if __name__ == "__main__":
    main()
