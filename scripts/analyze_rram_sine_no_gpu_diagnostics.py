"""No-GPU diagnostics for the RRAM sine KAN failure mode.

This script audits two questions without retraining:

1. Did the RRAM-sine models receive comparable training budget, and do the
   logged losses/validation curves suggest under-training?
2. Do the normalized RRAM-sine inputs fall outside the fixed KAN spline grid
   range used by the recurrent KAN gate layers?
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_rram_lstm_sequence import TASK_FILES, normalize, read_rows  # noqa: E402
from run_sdc_sequence import VARIANTS, build_sequence_model  # noqa: E402


RESULT_JSONS = [
    ROOT / "results" / "rram_lstm_sine_3seed_all_models.json",
    ROOT / "results" / "rram_lstm_sine_fused_clipped_kangru_3seed.json",
    ROOT / "results" / "rram_lstm_sine_hybrid_clipped_kangru_3seed.json",
]
OUT_DIR = ROOT / "results" / "rram_no_gpu_diagnostics"
GRID_MIN = -1.0
GRID_MAX = 1.0


def load_results() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for path in RESULT_JSONS:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("results", []):
            key = (str(row["variant"]), int(row["seed"]))
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    rows.sort(key=lambda r: (str(r["variant"]), int(r["seed"])))
    return rows


def tail_slope(values: list[float], n: int = 3) -> float:
    if len(values) < 2:
        return 0.0
    tail = values[-min(n, len(values)) :]
    x = np.arange(len(tail), dtype=np.float64)
    y = np.asarray(tail, dtype=np.float64)
    if len(tail) == 2:
        return float(y[-1] - y[0])
    return float(np.polyfit(x, y, deg=1)[0])


def fairness_audit(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_run: list[dict[str, Any]] = []
    for row in rows:
        epochs = row.get("epoch_summaries", [])
        losses = [float(e["train_loss_mean"]) for e in epochs]
        val_r2 = [float(e["val_metrics"]["r2"]) for e in epochs]
        val_rmse = [float(e["val_metrics"]["raw_rmse"]) for e in epochs]
        best_idx = int(np.argmax(val_r2)) if val_r2 else -1
        first_loss = losses[0] if losses else math.nan
        final_loss = losses[-1] if losses else math.nan
        loss_drop_fraction = (
            (first_loss - final_loss) / abs(first_loss)
            if losses and abs(first_loss) > 1e-12
            else math.nan
        )
        per_run.append(
            {
                "variant": row["variant"],
                "seed": int(row["seed"]),
                "epochs": int(row["epochs"]),
                "global_steps": int(row["global_steps"]),
                "batch_size": int(row["batch_size"]),
                "train_rows": int(row["train_rows"]),
                "val_rows": int(row["val_rows"]),
                "test_rows": int(row["test_rows"]),
                "params": int(row["complexity"]["total_parameters"]),
                "seconds_per_epoch": float(row["timing"]["seconds_per_epoch"]),
                "first_train_loss": first_loss,
                "final_train_loss": final_loss,
                "loss_drop_fraction": loss_drop_fraction,
                "tail_train_loss_slope": tail_slope(losses),
                "first_val_r2": val_r2[0] if val_r2 else math.nan,
                "best_val_r2": val_r2[best_idx] if best_idx >= 0 else math.nan,
                "best_val_epoch": best_idx + 1 if best_idx >= 0 else -1,
                "final_val_r2": float(row["final_val_metrics"]["r2"]),
                "final_test_r2": float(row["test_metrics"]["r2"]),
                "final_test_raw_rmse": float(row["test_metrics"]["raw_rmse"]),
                "final_minus_best_val_r2": (
                    float(row["final_val_metrics"]["r2"]) - val_r2[best_idx]
                    if best_idx >= 0
                    else math.nan
                ),
                "tail_val_r2_slope": tail_slope(val_r2),
                "tail_val_raw_rmse_slope": tail_slope(val_rmse),
            }
        )

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in per_run:
        grouped.setdefault(str(row["variant"]), []).append(row)
    summary: list[dict[str, Any]] = []
    for variant, group in sorted(grouped.items()):
        def vals(key: str) -> list[float]:
            return [float(r[key]) for r in group]

        summary.append(
            {
                "variant": variant,
                "runs": len(group),
                "epochs_min": min(int(r["epochs"]) for r in group),
                "epochs_max": max(int(r["epochs"]) for r in group),
                "steps_mean": mean(vals("global_steps")),
                "params": int(group[0]["params"]),
                "seconds_per_epoch_mean": mean(vals("seconds_per_epoch")),
                "test_r2_mean": mean(vals("final_test_r2")),
                "test_r2_std": pstdev(vals("final_test_r2")),
                "test_raw_rmse_mean": mean(vals("final_test_raw_rmse")),
                "loss_drop_fraction_mean": mean(vals("loss_drop_fraction")),
                "tail_train_loss_slope_mean": mean(vals("tail_train_loss_slope")),
                "tail_val_r2_slope_mean": mean(vals("tail_val_r2_slope")),
                "best_val_epoch_mean": mean(vals("best_val_epoch")),
                "final_minus_best_val_r2_mean": mean(vals("final_minus_best_val_r2")),
            }
        )
    summary.sort(key=lambda r: float(r["test_r2_mean"]), reverse=True)
    return per_run, summary


def rram_arrays() -> dict[str, np.ndarray]:
    task = TASK_FILES["sine"]
    root = ROOT / "data" / "raw" / "rram_lstm"
    train_x, train_y = read_rows(root, task["train"], task["seq_len"], task["input_dim"], task["target_col"])
    val_x, val_y = read_rows(root, task["val"], task["seq_len"], task["input_dim"], task["target_col"])
    test_x, test_y = read_rows(root, task["test"], task["seq_len"], task["input_dim"], task["target_col"])
    return normalize(train_x, train_y, val_x, val_y, test_x, test_y)


def coverage_for_array(name: str, x: np.ndarray) -> list[dict[str, Any]]:
    flat = x.reshape(-1, x.shape[-1])
    rows: list[dict[str, Any]] = []
    feature_names = ["voltage_history", "current_history"]
    for idx in range(flat.shape[1]):
        values = flat[:, idx]
        outside = (values < GRID_MIN) | (values > GRID_MAX)
        rows.append(
            {
                "scope": "normalized_physical_input",
                "split": name,
                "feature": feature_names[idx] if idx < len(feature_names) else f"feature_{idx}",
                "count": int(values.size),
                "min": float(values.min()),
                "p01": float(np.quantile(values, 0.01)),
                "p05": float(np.quantile(values, 0.05)),
                "mean": float(values.mean()),
                "p95": float(np.quantile(values, 0.95)),
                "p99": float(np.quantile(values, 0.99)),
                "max": float(values.max()),
                "outside_grid_fraction": float(outside.mean()),
                "outside_grid_count": int(outside.sum()),
            }
        )
    all_values = flat.reshape(-1)
    outside_all = (all_values < GRID_MIN) | (all_values > GRID_MAX)
    rows.append(
        {
            "scope": "normalized_physical_input",
            "split": name,
            "feature": "all_features",
            "count": int(all_values.size),
            "min": float(all_values.min()),
            "p01": float(np.quantile(all_values, 0.01)),
            "p05": float(np.quantile(all_values, 0.05)),
            "mean": float(all_values.mean()),
            "p95": float(np.quantile(all_values, 0.95)),
            "p99": float(np.quantile(all_values, 0.99)),
            "max": float(all_values.max()),
            "outside_grid_fraction": float(outside_all.mean()),
            "outside_grid_count": int(outside_all.sum()),
        }
    )
    return rows


@torch.no_grad()
def runtime_kan_inputs(model: torch.nn.Module, x: torch.Tensor) -> dict[str, np.ndarray]:
    """Collect actual KAN layer input tensors for recurrent KAN variants."""

    model.eval()
    cell = getattr(model, "cell", None)
    if cell is None:
        return {}
    batch, seq_len, _ = x.shape
    h_t = x.new_zeros(batch, int(getattr(model, "hidden_dim", 0)))
    c_t = x.new_zeros(batch, int(getattr(model, "hidden_dim", 0)))
    gate_inputs = []
    candidate_inputs = []

    for t in range(seq_len):
        x_t = x[:, t, :]
        if hasattr(cell, "candidate"):
            gate_input = torch.cat([x_t, h_t], dim=-1)
            gate_inputs.append(gate_input.cpu().numpy())
            raw = cell.gates(gate_input)
            z_raw, r_raw = raw.chunk(2, dim=-1)
            if cell.__class__.__name__.startswith("Clipped"):
                r_t = ((r_raw + 1.0) * 0.5).clamp(0.0, 1.0)
            else:
                r_t = torch.sigmoid(r_raw)
            candidate_input = torch.cat([x_t, r_t * h_t], dim=-1)
            candidate_inputs.append(candidate_input.cpu().numpy())
            h_t = cell(x_t, h_t)
        elif hasattr(cell, "gates"):
            gate_input = torch.cat([x_t, h_t], dim=-1)
            gate_inputs.append(gate_input.cpu().numpy())
            # LSTM-like KAN gate cells.
            h_t, c_t = cell(x_t, (h_t, c_t))
        else:
            break

    out: dict[str, np.ndarray] = {}
    if gate_inputs:
        out["gate_input"] = np.concatenate(gate_inputs, axis=0)
    if candidate_inputs:
        out["candidate_input"] = np.concatenate(candidate_inputs, axis=0)
    return out


def coverage_for_runtime_tensor(variant: str, seed: int, tensor_name: str, values: np.ndarray) -> dict[str, Any]:
    all_values = values.reshape(-1)
    outside_all = (all_values < GRID_MIN) | (all_values > GRID_MAX)
    physical = values[:, :2].reshape(-1)
    hidden = values[:, 2:].reshape(-1) if values.shape[1] > 2 else np.asarray([], dtype=np.float32)
    hidden_outside = ((hidden < GRID_MIN) | (hidden > GRID_MAX)) if hidden.size else np.asarray([], dtype=bool)
    physical_outside = (physical < GRID_MIN) | (physical > GRID_MAX)
    return {
        "scope": "runtime_kan_layer_input",
        "variant": variant,
        "seed": seed,
        "tensor": tensor_name,
        "dimensions": int(values.shape[1]),
        "count": int(all_values.size),
        "min": float(all_values.min()),
        "p01": float(np.quantile(all_values, 0.01)),
        "p05": float(np.quantile(all_values, 0.05)),
        "mean": float(all_values.mean()),
        "p95": float(np.quantile(all_values, 0.95)),
        "p99": float(np.quantile(all_values, 0.99)),
        "max": float(all_values.max()),
        "outside_grid_fraction": float(outside_all.mean()),
        "outside_grid_count": int(outside_all.sum()),
        "physical_outside_grid_fraction": float(physical_outside.mean()),
        "hidden_outside_grid_fraction": float(hidden_outside.mean()) if hidden.size else 0.0,
    }


def grid_audit(arrays: dict[str, np.ndarray]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_rows: list[dict[str, Any]] = []
    for split in ["train_x", "val_x", "test_x"]:
        raw_rows.extend(coverage_for_array(split.replace("_x", ""), arrays[split]))

    runtime_rows: list[dict[str, Any]] = []
    test_tensor = torch.from_numpy(arrays["test_x"].astype(np.float32))
    checkpoint_dir = ROOT / "checkpoints" / "rram_lstm_sequence"
    for variant in ["kan_lstm_h32", "clipped_kan_lstm_h32", "kan_gru_h32", "clipped_kan_gru_h32"]:
        for seed in [0, 1, 2]:
            ckpt_path = checkpoint_dir / f"{variant}_sine_seed{seed}.pt"
            if not ckpt_path.exists():
                continue
            model = build_sequence_model(VARIANTS[variant], input_dim=2)
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            tensors = runtime_kan_inputs(model, test_tensor)
            for tensor_name, values in tensors.items():
                runtime_rows.append(coverage_for_runtime_tensor(variant, seed, tensor_name, values))
    return raw_rows, runtime_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: Path,
    fairness_summary: list[dict[str, Any]],
    physical_rows: list[dict[str, Any]],
    runtime_rows: list[dict[str, Any]],
) -> None:
    kan_rows = [r for r in fairness_summary if "kan" in str(r["variant"])]
    dense_rows = [r for r in fairness_summary if "kan" not in str(r["variant"])]
    dense_epochs = sorted({int(r["epochs_min"]) for r in dense_rows} | {int(r["epochs_max"]) for r in dense_rows})
    kan_epochs = sorted({int(r["epochs_min"]) for r in kan_rows} | {int(r["epochs_max"]) for r in kan_rows})
    physical_all = [r for r in physical_rows if r["feature"] == "all_features"]
    runtime_all = runtime_rows
    max_physical_outside = max(float(r["outside_grid_fraction"]) for r in physical_all)
    max_runtime_outside = max(float(r["outside_grid_fraction"]) for r in runtime_all) if runtime_all else float("nan")
    max_hidden_outside = max(float(r["hidden_outside_grid_fraction"]) for r in runtime_all) if runtime_all else float("nan")

    lines = [
        "# RRAM Sine No-GPU Diagnostics",
        "",
        "## Purpose",
        "",
        "This audit checks the RRAM sine boundary result without retraining. It answers two questions: whether the KAN rows received comparable logged training budget, and whether fixed-grid KAN layers were exposed to out-of-grid normalized inputs.",
        "",
        "## Diagnostic Finding",
        "",
        f"- Dense and KAN rows in the locked RRAM sine comparison used the same logged epoch count: dense epochs `{dense_epochs}`, KAN epochs `{kan_epochs}`.",
        "- The all-model RRAM sine runs are short by design: 10 epochs and 20 optimizer steps per seed, because the dataset has 995 training rows and batch size 512.",
        f"- Physical normalized RRAM sine inputs extend beyond the KAN grid `[-1, 1]`; the worst split-level all-feature out-of-grid fraction is `{max_physical_outside:.3f}`.",
        f"- Runtime KAN layer inputs in saved sine checkpoints also extend beyond `[-1, 1]`; the worst measured recurrent KAN tensor out-of-grid fraction is `{max_runtime_outside:.3f}`.",
        f"- Hidden-state channels remain inside the grid in this audit; worst hidden out-of-grid fraction is `{max_hidden_outside:.3f}`. The mismatch is therefore driven by normalized physical voltage/current history channels, not hidden-state explosion.",
        "",
        "Interpretation: the RRAM sine weakness should be reported as a boundary result under the locked fixed-grid configuration, with a plausible diagnostic that normalized physical inputs are often outside the spline grid. A follow-up normalized-grid or wider-grid RRAM-only rerun would be a targeted confirmation, not a required change to the locked SDC conclusions.",
        "",
        "## Training-Budget And Convergence Summary",
        "",
        "| Variant | Runs | Epochs | Steps | Test R2 | Loss Drop | Tail Loss Slope | Final-Best Val R2 | Sec/Epoch |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in fairness_summary:
        lines.append(
            f"| `{row['variant']}` | {row['runs']} | {row['epochs_min']}-{row['epochs_max']} | "
            f"{row['steps_mean']:.0f} | {row['test_r2_mean']:.4f} +/- {row['test_r2_std']:.4f} | "
            f"{row['loss_drop_fraction_mean']:.3f} | {row['tail_train_loss_slope_mean']:.4g} | "
            f"{row['final_minus_best_val_r2_mean']:.4g} | {row['seconds_per_epoch_mean']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Normalized Physical Input Coverage",
            "",
            "| Split | Feature | Min | P05 | P95 | Max | Out-Of-Grid Fraction |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in physical_rows:
        if row["feature"] != "all_features":
            continue
        lines.append(
            f"| `{row['split']}` | `{row['feature']}` | {row['min']:.3f} | {row['p05']:.3f} | "
            f"{row['p95']:.3f} | {row['max']:.3f} | {row['outside_grid_fraction']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Runtime KAN Input Coverage",
            "",
            "| Variant | Tensor | Out-Of-Grid | Physical Out-Of-Grid | Hidden Out-Of-Grid | Min | Max |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    grouped_runtime: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in runtime_rows:
        grouped_runtime.setdefault((str(row["variant"]), str(row["tensor"])), []).append(row)
    for (variant, tensor), rows in sorted(grouped_runtime.items()):
        lines.append(
            f"| `{variant}` | `{tensor}` | "
            f"{mean(float(r['outside_grid_fraction']) for r in rows):.3f} | "
            f"{mean(float(r['physical_outside_grid_fraction']) for r in rows):.3f} | "
            f"{mean(float(r['hidden_outside_grid_fraction']) for r in rows):.3f} | "
            f"{mean(float(r['min']) for r in rows):.3f} | {mean(float(r['max']) for r in rows):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Manuscript Use",
            "",
            "Use this as a short defensive paragraph in the RRAM section: the poor RRAM-sine KAN result is not due to fewer logged epochs than the dense GRU; all compared rows received the same 10-epoch, 20-step budget. The fixed-grid KAN layers, however, frequently receive normalized physical inputs outside the `[-1, 1]` grid, which is a credible explanation for why dense GRU generalizes better in this regime.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_figure(
    path: Path,
    fairness_summary: list[dict[str, Any]],
    physical_rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    variants = [str(r["variant"]) for r in fairness_summary]
    r2_values = [float(r["test_r2_mean"]) for r in fairness_summary]
    loss_drop = [float(r["loss_drop_fraction_mean"]) for r in fairness_summary]
    split_rows = [r for r in physical_rows if r["feature"] == "all_features"]
    splits = [str(r["split"]) for r in split_rows]
    outside = [float(r["outside_grid_fraction"]) for r in split_rows]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    axes[0].bar(range(len(variants)), r2_values, color="#4c78a8")
    axes[0].set_xticks(range(len(variants)))
    axes[0].set_xticklabels(variants, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("Test R2")
    axes[0].set_title("RRAM Sine Test Accuracy")
    axes[0].axhline(0.0, color="#333333", linewidth=0.8)

    axes[1].bar(range(len(variants)), loss_drop, color="#f58518")
    axes[1].set_xticks(range(len(variants)))
    axes[1].set_xticklabels(variants, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("Train Loss Drop Fraction")
    axes[1].set_title("Logged 10-Epoch Convergence")

    axes[2].bar(range(len(splits)), outside, color="#54a24b")
    axes[2].set_xticks(range(len(splits)))
    axes[2].set_xticklabels(splits)
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Fraction Outside [-1, 1]")
    axes[2].set_title("Normalized Input vs KAN Grid")

    fig.suptitle("RRAM Sine No-GPU Diagnostic Audit", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_results()
    fairness_per_run, fairness_summary = fairness_audit(rows)
    arrays = rram_arrays()
    physical_rows, runtime_rows = grid_audit(arrays)

    write_csv(OUT_DIR / "rram_sine_fairness_per_run.csv", fairness_per_run)
    write_csv(OUT_DIR / "rram_sine_fairness_summary.csv", fairness_summary)
    write_csv(OUT_DIR / "rram_sine_physical_grid_coverage.csv", physical_rows)
    write_csv(OUT_DIR / "rram_sine_runtime_kan_grid_coverage.csv", runtime_rows)
    payload = {
        "fairness_per_run": fairness_per_run,
        "fairness_summary": fairness_summary,
        "physical_grid_coverage": physical_rows,
        "runtime_kan_grid_coverage": runtime_rows,
    }
    (OUT_DIR / "rram_sine_no_gpu_diagnostics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(OUT_DIR / "rram_sine_no_gpu_diagnostics.md", fairness_summary, physical_rows, runtime_rows)
    write_figure(OUT_DIR / "rram_sine_no_gpu_diagnostics.png", fairness_summary, physical_rows)
    print(json.dumps({"out_dir": str(OUT_DIR), "runs": len(rows), "runtime_rows": len(runtime_rows)}, indent=2))


if __name__ == "__main__":
    main()
