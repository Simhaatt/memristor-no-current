"""Build a manuscript-ready result package for the memristor benchmark paper."""

from __future__ import annotations

import csv
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
SCRIPT_DIR = ROOT / "scripts"
for path in (SRC_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_sdc_sequence import WindowedCurveDataset, build_sequence_model
from evaluate_sequence_checkpoints import load_checkpoint, variant_from_checkpoint


OUT = ROOT / "results" / "manuscript_ready"
FIG = OUT / "figures"
TABLE = OUT / "tables"
REPORT = OUT / "reports"
PROV = OUT / "provenance"
SCRIPT_OUT = OUT / "scripts"
ZIP_OUT = ROOT / "results" / "manuscript_ready_upload.zip"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def model_label(name: str) -> str:
    labels = {
        "static_mlp_h200": "Static MLP",
        "lstm_h100": "LSTM",
        "gru_h116": "GRU",
        "kan_lstm_h32": "KAN-LSTM",
        "clipped_kan_lstm_h32": "Clipped KAN-LSTM",
        "kan_gru_h32": "KAN-GRU",
        "clipped_kan_gru_h32": "Clipped KAN-GRU",
        "fused_clipped_kan_gru_h32": "Fused KAN-GRU",
        "hybrid_clipped_kan_gru_h32": "Hybrid KAN-GRU",
        "compact_only": "Compact only",
    }
    return labels.get(name, name.replace("_", " "))


def save_bar(
    labels: list[str],
    values: list[float],
    path: Path,
    *,
    title: str,
    ylabel: str,
    errors: list[float] | None = None,
    rotation: int = 35,
    color: str = "#3B6EA8",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(8, len(labels) * 0.75), 4.8))
    x = np.arange(len(labels))
    plt.bar(x, values, yerr=errors, color=color, capsize=3 if errors else 0)
    plt.xticks(x, labels, rotation=rotation, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_main_sdc() -> None:
    rows = read_csv(ROOT / "results" / "final_tables" / "sdc_locked_test_results.csv")
    keep = [row for row in rows if row["feature_mode"] == "no_current"]
    keep = sorted(keep, key=lambda row: float(row["teacher_r2_mean"]), reverse=True)
    save_bar(
        [model_label(row["variant"]) for row in keep],
        [float(row["teacher_r2_mean"]) for row in keep],
        FIG / "figure_01_main_sdc_no_current_r2.png",
        title="Original SDC No-Current Test R2",
        ylabel="Test R2",
        errors=[float(row["teacher_r2_std"]) for row in keep],
    )


def plot_runtime_tradeoff() -> None:
    acc_rows = {
        row["variant"]: row
        for row in read_csv(ROOT / "results" / "final_tables" / "sdc_locked_test_results.csv")
        if row["feature_mode"] == "no_current"
    }
    timing_rows = {
        row["variant"]: row
        for row in read_csv(ROOT / "results" / "final_tables" / "sdc_training_complexity_timing.csv")
        if row["feature_mode"] == "no_current"
    }
    variants = [variant for variant in acc_rows if variant in timing_rows]
    plt.figure(figsize=(8.8, 5.2))
    for variant in variants:
        x = float(timing_rows[variant]["seconds_per_epoch_mean"])
        y = float(acc_rows[variant]["teacher_r2_mean"])
        params = float(acc_rows[variant]["params"])
        plt.scatter(x, y, s=max(35, params / 350), alpha=0.8)
        plt.text(x * 1.03, y, model_label(variant), fontsize=8, va="center")
    plt.xscale("log")
    plt.xlabel("Seconds per epoch (log scale)")
    plt.ylabel("Test R2")
    plt.title("Accuracy vs Training Time on Original SDC")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG / "figure_02_sdc_accuracy_runtime_tradeoff.png", dpi=220)
    plt.close()


def plot_four_dopant() -> None:
    rows = read_csv(ROOT / "results" / "final_tables" / "sdc4dopant_corrected_test_summary.csv")
    folds = ["pooled", "holdout_c", "holdout_sn", "holdout_cr", "holdout_w"]
    variants = ["static_mlp_h200", "lstm_h100", "gru_h116", "clipped_kan_gru_h32"]
    values: dict[str, list[float]] = {variant: [] for variant in variants}
    for fold in folds:
        for variant in variants:
            row = next(r for r in rows if r["fold"] == fold and r["variant"] == variant)
            values[variant].append(float(row["test_r2_mean"]))
    x = np.arange(len(folds))
    width = 0.18
    plt.figure(figsize=(9.5, 5.2))
    for idx, variant in enumerate(variants):
        plt.bar(x + (idx - 1.5) * width, values[variant], width, label=model_label(variant))
    plt.xticks(x, folds)
    plt.ylabel("Test R2")
    plt.title("Four-Dopant SDC Sine Leave-One-Family Results")
    plt.legend(fontsize=8)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG / "figure_03_four_dopant_lodo_r2.png", dpi=220)
    plt.close()


def plot_rram() -> None:
    rows = read_csv(ROOT / "results" / "final_tables" / "rram_external_validation_100epoch_summary.csv")
    rows = [r for r in rows if r["task"] in {"random_walk", "sine"}]
    tasks = ["random_walk", "sine"]
    variants = ["clipped_kan_gru_h32", "clipped_kan_lstm_h32", "gru_h116", "lstm_h100"]
    x = np.arange(len(tasks))
    width = 0.18
    plt.figure(figsize=(8.8, 5.0))
    for idx, variant in enumerate(variants):
        vals = []
        for task in tasks:
            found = [r for r in rows if r["task"] == task and r["variant"] == variant]
            vals.append(float(found[0]["test_r2_mean"]) if found else np.nan)
        plt.bar(x + (idx - 1.5) * width, vals, width, label=model_label(variant))
    plt.xticks(x, ["RRAM random-walk", "RRAM sine"])
    plt.ylabel("Test R2")
    plt.title("RRAM External Validation, 100 Epochs")
    plt.legend(fontsize=8)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG / "figure_04_rram_external_r2.png", dpi=220)
    plt.close()


def plot_rram_adjusted_grid_distillation() -> None:
    rows = read_csv(ROOT / "results" / "final_tables" / "rram_sine_adjusted_grid_distillation_summary.csv")
    selected = [
        ("adjusted_grid_supervised", "clipped_kan_gru_h32_wide", ""),
        ("adjusted_grid_supervised", "gru_h116", ""),
        ("offline_distillation", "distilled_gru_h64", "0.25"),
        ("offline_distillation", "distilled_gru_h116", "0.25"),
        ("adjusted_grid_supervised", "fused_clipped_kan_gru_h32_wide", ""),
        ("adjusted_grid_supervised", "kan_gru_h32_wide", ""),
        ("adjusted_grid_supervised", "kan_lstm_h32_wide", ""),
    ]
    labels: list[str] = []
    values: list[float] = []
    errors: list[float] = []
    colors: list[str] = []
    for run_type, variant, weight in selected:
        row = next(
            r
            for r in rows
            if r["run_type"] == run_type
            and r["variant"] == variant
            and (not weight or r["distill_weight"] == weight)
        )
        label = model_label(variant)
        if run_type == "offline_distillation":
            label = f"{label}\nw={weight}"
            colors.append("#72B7B2")
        else:
            colors.append("#4C78A8" if "kan" in variant else "#F58518")
        labels.append(label)
        values.append(float(row["test_r2_mean"]))
        errors.append(float(row["test_r2_std"]))
    x = np.arange(len(labels))
    plt.figure(figsize=(10.2, 5.2))
    plt.bar(x, values, yerr=errors, color=colors, capsize=3)
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Test R2")
    plt.title("RRAM Sine Adjusted-Grid And Distillation Follow-Up")
    plt.ylim(max(0.90, min(values) - 0.02), 1.0)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG / "figure_14_rram_sine_adjusted_grid_distillation.png", dpi=220)
    plt.close()


def plot_rram_sine_narrow100() -> None:
    original = read_csv(ROOT / "results" / "final_tables" / "rram_external_validation_final.csv")
    narrow = read_csv(ROOT / "results" / "final_tables" / "rram_sine_narrow100_summary.csv")
    wide = read_csv(ROOT / "results" / "final_tables" / "rram_sine_adjusted_grid_distillation_summary.csv")

    variants = [
        ("gru_h116", "GRU"),
        ("kan_gru_h32", "KAN-GRU"),
        ("clipped_kan_gru_h32", "Clipped KAN-GRU"),
    ]
    settings = ["Original 10 epoch", "Narrow [-1,1] 100 epoch", "Wide [-2,2] 100 epoch"]
    values: dict[str, list[float]] = {}
    errors: dict[str, list[float]] = {}
    for variant, label in variants:
        original_row = next(r for r in original if r["task"] == "sine" and r["variant"] == variant)
        narrow_row = next(r for r in narrow if r["variant"] == variant)
        wide_variant = f"{variant}_wide" if "kan" in variant else variant
        wide_row = next(
            r
            for r in wide
            if r["run_type"] == "adjusted_grid_supervised" and r["variant"] == wide_variant
        )
        values[label] = [
            float(original_row["test_r2_mean"]),
            float(narrow_row["test_r2_mean"]),
            float(wide_row["test_r2_mean"]),
        ]
        errors[label] = [
            float(original_row["test_r2_std"]),
            float(narrow_row["test_r2_std"]),
            float(wide_row["test_r2_std"]),
        ]

    x = np.arange(len(settings))
    width = 0.23
    colors = ["#F58518", "#54A24B", "#4C78A8"]
    plt.figure(figsize=(9.2, 5.2))
    for idx, (label, vals) in enumerate(values.items()):
        plt.bar(
            x + (idx - 1) * width,
            vals,
            width,
            yerr=errors[label],
            label=label,
            color=colors[idx],
            capsize=3,
        )
    plt.xticks(x, settings, rotation=12, ha="right")
    plt.ylabel("RRAM sine test R2")
    plt.title("RRAM Sine Causal Ablation: Budget vs Grid Range")
    plt.ylim(0.20, 1.0)
    plt.legend(fontsize=8)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG / "figure_17_rram_sine_narrow100_causal_ablation.png", dpi=220)
    plt.close()


def plot_sdc_distillation() -> None:
    rows = read_csv(ROOT / "results" / "final_tables" / "sdc_distillation_summary.csv")
    selected = [
        ("distilled_gru_h116", "0.25"),
        ("supervised_gru_h116", ""),
        ("distilled_gru_h64", "0.25"),
        ("supervised_gru_h64", ""),
        ("distilled_gru_h32", "0.5"),
        ("supervised_gru_h32", ""),
    ]
    labels: list[str] = []
    values: list[float] = []
    errors: list[float] = []
    colors: list[str] = []
    for variant, weight in selected:
        row = next(r for r in rows if r["variant"] == variant and (not weight or r["distill_weight"] == weight))
        label = model_label(variant)
        if variant.startswith("distilled"):
            label = f"{label}\nw={weight}"
            colors.append("#72B7B2")
        else:
            colors.append("#F58518")
        labels.append(label)
        values.append(float(row["test_r2_mean"]))
        errors.append(float(row["test_r2_std"]))
    plt.figure(figsize=(9.6, 5.0))
    x = np.arange(len(labels))
    plt.bar(x, values, yerr=errors, color=colors, capsize=3)
    plt.axhline(0.9944756031036377, color="#4C78A8", linestyle="--", linewidth=1.5, label="Clipped KAN-GRU teacher")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Original SDC no-current test R2")
    plt.title("SDC Teacher Distillation Into Dense GRU Students")
    plt.ylim(0.94, 1.0)
    plt.legend(fontsize=8)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG / "figure_16_sdc_teacher_distillation.png", dpi=220)
    plt.close()


def plot_perturbation() -> None:
    files = [
        ROOT / "results" / "test_evaluation" / "clipped_kangru_3seed_sdc_no_current_test_eval.summary.csv",
        ROOT / "results" / "test_evaluation" / "clipped_kangru_3seed_shuffle_time_test_eval.summary.csv",
        ROOT / "results" / "test_evaluation" / "clipped_kangru_3seed_reverse_time_test_eval.summary.csv",
    ]
    rows = [read_csv(path)[0] for path in files]
    labels = ["Original order", "Shuffle time", "Reverse time"]
    save_bar(
        labels,
        [float(row["teacher_r2_mean"]) for row in rows],
        FIG / "figure_05_temporal_perturbation_r2.png",
        title="Temporal Perturbation of Clipped KAN-GRU",
        ylabel="Test R2",
        errors=[float(row["teacher_r2_std"]) for row in rows],
        rotation=0,
        color="#A84E3B",
    )


def plot_d0132_hybrid() -> None:
    rows = read_csv(ROOT / "results" / "d0132_hybrid_residual" / "d0132_hybrid_residual_summary.csv")
    selected = [
        ("device_transfer", "VTEAM", "compact_only"),
        ("device_transfer", "VTEAM", "clipped_kan_gru_h32"),
        ("within_d0132", "VTEAM", "compact_only"),
        ("within_d0132", "VTEAM", "gru_h116"),
        ("within_d0132", "published_model", "compact_only"),
        ("within_d0132", "published_model", "clipped_kan_gru_h32"),
    ]
    labels: list[str] = []
    values: list[float] = []
    for protocol, compact, model in selected:
        row = next(r for r in rows if r["protocol"] == protocol and r["compact"] == compact and r["model"] == model)
        labels.append(f"{protocol}\n{compact}+{model_label(model)}")
        values.append(float(row["r2_mean"]))
    save_bar(
        labels,
        values,
        FIG / "figure_06_d0132_hybrid_residual_r2.png",
        title="D0132 Compact/VTEAM Plus ML Residual",
        ylabel="R2",
        rotation=25,
        color="#4C8C57",
    )


def plot_transfer() -> None:
    rows = read_csv(ROOT / "results" / "transfer" / "sdc_to_sdc4dopant_quickpass_summary.csv")
    variants = ["static_mlp_h200", "gru_h116", "lstm_h100"]
    modes = ["zero_shot", "head_only", "full", "scratch_same_budget"]
    x = np.arange(len(variants))
    width = 0.19
    plt.figure(figsize=(8.8, 5.0))
    for idx, mode in enumerate(modes):
        vals = []
        for variant in variants:
            row = next(r for r in rows if r["variant"] == variant and r["adaptation_mode"] == mode)
            vals.append(float(row["test_r2_mean"]))
        plt.bar(x + (idx - 1.5) * width, vals, width, label=mode)
    plt.xticks(x, [model_label(v) for v in variants])
    plt.ylabel("Holdout W test R2")
    plt.title("Original SDC Checkpoint Transfer to Four-Dopant W")
    plt.legend(fontsize=8)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG / "figure_07_sdc_to_four_dopant_transfer.png", dpi=220)
    plt.close()


def plot_sdc_prediction_trace() -> None:
    device = torch.device("cpu")
    ckpt = ROOT / "results" / "kaggle_imported" / "kangru_sdc_100epoch_seed0" / "checkpoints" / "sequence" / "clipped_kan_gru_h32_no_current_seed0.pt"
    cache = ROOT / "data" / "processed" / "sdc_state_splits.npz"
    payload = load_checkpoint(ckpt, device)
    variant = variant_from_checkpoint(payload)
    dataset = WindowedCurveDataset(cache, "test", window=256, stride=512, feature_mode="no_current")
    x, y = dataset[0]
    model = build_sequence_model(variant, input_dim=dataset.feature_dim).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    with torch.no_grad():
        pred = model(x.unsqueeze(0)).squeeze(0).squeeze(-1).cpu().numpy()
    target = y.squeeze(-1).cpu().numpy()
    plt.figure(figsize=(9.0, 4.6))
    plt.plot(target, label="Actual normalized current", linewidth=2)
    plt.plot(pred, label="Predicted normalized current", linewidth=1.8, alpha=0.85)
    plt.xlabel("Step within test window")
    plt.ylabel("Normalized current")
    plt.title("Representative Original SDC Test Prediction: Clipped KAN-GRU")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG / "figure_08_sdc_prediction_trace.png", dpi=220)
    plt.close()


def copy_existing_figures() -> None:
    copies = {
        ROOT / "results" / "interpretability" / "kan_gate_splines_sdc" / "physical_input_spline_importance.png": FIG / "figure_09_kan_gate_physical_input_importance.png",
        ROOT / "results" / "interpretability" / "kan_gate_splines_sdc" / "gate_activation_and_saturation.png": FIG / "figure_10_kan_gate_activation_saturation.png",
        ROOT / "results" / "interpretability" / "gate_interventions_sdc" / "intervention_r2_drop.png": FIG / "figure_11_gate_intervention_r2_drop.png",
        ROOT / "results" / "compact_model_baselines" / "d0132" / "d0132_compact_baseline_nrmse.png": FIG / "figure_12_d0132_compact_baseline_nrmse.png",
        ROOT / "results" / "rram_no_gpu_diagnostics" / "rram_sine_no_gpu_diagnostics.png": FIG / "figure_13_rram_sine_no_gpu_diagnostic.png",
        ROOT / "results" / "sdc_grid_coverage" / "sdc_no_current_grid_coverage.png": FIG / "figure_15_sdc_no_current_grid_coverage.png",
        ROOT / "results" / "interpretability" / "kan_gate_deep_dive_sdc" / "top_learned_edge_functions.png": FIG / "figure_18_top_learned_kan_edge_functions.png",
        ROOT / "results" / "interpretability" / "kan_gate_deep_dive_sdc" / "topk_vs_random_edge_ablation.png": FIG / "figure_19_topk_vs_random_kan_edge_ablation.png",
        ROOT / "results" / "interpretability" / "kan_gate_deep_dive_sdc" / "dense_gru_vs_kan_input_ablation.png": FIG / "figure_20_dense_gru_vs_kan_input_ablation.png",
        ROOT / "results" / "interpretability" / "kan_gate_deep_dive_sdc" / "parametric_fit_examples.png": FIG / "figure_21_parametric_fits_to_dominant_kan_edges.png",
        ROOT / "results" / "interpretability" / "kan_gate_stability_corrected_sdc" / "gate_input_aggregate_importance.png": FIG / "figure_22_corrected_gate_input_importance.png",
        ROOT / "results" / "interpretability" / "kan_gate_stability_corrected_sdc" / "topk_overlap_raw_vs_aligned.png": FIG / "figure_23_topk_overlap_raw_vs_aligned.png",
        ROOT / "results" / "interpretability" / "kan_gate_stability_corrected_sdc" / "aggregate_gate_input_response_curves.png": FIG / "figure_24_aggregate_gate_input_response_curves.png",
    }
    for src, dst in copies.items():
        if src.exists():
            copy_file(src, dst)


def copy_tables_and_reports() -> None:
    table_files = [
        ROOT / "results" / "final_tables" / "sdc_locked_test_results.csv",
        ROOT / "results" / "final_tables" / "sdc_training_complexity_timing.csv",
        ROOT / "results" / "final_tables" / "sdc4dopant_corrected_test_summary.csv",
        ROOT / "results" / "final_tables" / "sdc4dopant_corrected_test_gaps.csv",
        ROOT / "results" / "final_tables" / "rram_external_validation_final.csv",
        ROOT / "results" / "final_tables" / "rram_external_validation_100epoch_summary.csv",
        ROOT / "results" / "final_tables" / "rram_external_validation_100epoch_per_run.csv",
        ROOT / "results" / "d0132_hybrid_residual" / "d0132_hybrid_residual_summary.csv",
        ROOT / "results" / "d0132_resistance_sequence" / "d0132_resistance_sequence_summary.csv",
        ROOT / "results" / "compact_model_baselines" / "d0132" / "d0132_compact_baseline_summary.csv",
        ROOT / "results" / "transfer" / "sdc_to_sdc4dopant_quickpass_summary.csv",
        ROOT / "results" / "transfer" / "sdc_to_sdc4dopant_kan_smoke_summary.csv",
        ROOT / "results" / "interpretability" / "gate_interventions_sdc" / "gate_interventions_summary.csv",
        ROOT / "results" / "interpretability" / "kan_gate_splines_sdc" / "edge_importance_summary.csv",
        ROOT / "results" / "interpretability" / "kan_gate_deep_dive_sdc" / "edge_importance_ranked_mean.csv",
        ROOT / "results" / "interpretability" / "kan_gate_deep_dive_sdc" / "edge_rank_consistency.csv",
        ROOT / "results" / "interpretability" / "kan_gate_deep_dive_sdc" / "edge_ablation_topk_vs_random_summary.csv",
        ROOT / "results" / "interpretability" / "kan_gate_deep_dive_sdc" / "input_feature_ablation_summary.csv",
        ROOT / "results" / "interpretability" / "kan_gate_deep_dive_sdc" / "parametric_edge_fits.csv",
        ROOT / "results" / "interpretability" / "kan_gate_stability_corrected_sdc" / "gate_input_aggregate_importance_summary.csv",
        ROOT / "results" / "interpretability" / "kan_gate_stability_corrected_sdc" / "gate_input_aggregate_stability.csv",
        ROOT / "results" / "interpretability" / "kan_gate_stability_corrected_sdc" / "unit_alignment_quality.csv",
        ROOT / "results" / "interpretability" / "kan_gate_stability_corrected_sdc" / "unit_aligned_edge_rank_stability.csv",
        ROOT / "results" / "interpretability" / "kan_gate_stability_corrected_sdc" / "topk_edge_overlap_raw_vs_aligned_summary.csv",
        ROOT / "results" / "rram_no_gpu_diagnostics" / "rram_sine_fairness_summary.csv",
        ROOT / "results" / "rram_no_gpu_diagnostics" / "rram_sine_fairness_per_run.csv",
        ROOT / "results" / "rram_no_gpu_diagnostics" / "rram_sine_physical_grid_coverage.csv",
        ROOT / "results" / "rram_no_gpu_diagnostics" / "rram_sine_runtime_kan_grid_coverage.csv",
        ROOT / "results" / "final_tables" / "rram_sine_adjusted_grid_distillation_summary.csv",
        ROOT / "results" / "final_tables" / "rram_sine_adjusted_grid_distillation_per_run.csv",
        ROOT / "results" / "final_tables" / "rram_sine_narrow100_summary.csv",
        ROOT / "results" / "sdc_grid_coverage" / "sdc_no_current_physical_grid_coverage.csv",
        ROOT / "results" / "sdc_grid_coverage" / "sdc_no_current_runtime_kan_grid_coverage.csv",
        ROOT / "results" / "final_tables" / "sdc_distillation_summary.csv",
        ROOT / "results" / "final_tables" / "sdc_distillation_per_run.csv",
        ROOT / "results" / "training_budget_audit" / "training_budget_summary.csv",
        ROOT / "results" / "training_budget_audit" / "training_budget_runs.csv",
    ]
    report_files = [
        ROOT / "results" / "final_tables" / "memristor_modeling_detailed_report.md",
        ROOT / "results" / "final_tables" / "methods_parameter_audit.md",
        ROOT / "results" / "final_tables" / "sdc4dopant_corrected_test_results.md",
        ROOT / "results" / "final_tables" / "rram_external_validation_final.md",
        ROOT / "results" / "final_tables" / "rram_external_validation_100epoch.md",
        ROOT / "results" / "compact_model_baselines" / "d0132" / "d0132_compact_model_baseline.md",
        ROOT / "results" / "d0132_resistance_sequence" / "d0132_resistance_sequence_results.md",
        ROOT / "results" / "d0132_hybrid_residual" / "d0132_hybrid_residual_results.md",
        ROOT / "results" / "transfer" / "sdc_to_sdc4dopant_quickpass.md",
        ROOT / "results" / "transfer" / "sdc_to_sdc4dopant_kan_smoke.md",
        ROOT / "results" / "interpretability" / "gate_interventions_sdc" / "gate_intervention_results.md",
        ROOT / "results" / "interpretability" / "kan_gate_splines_sdc" / "kan_gate_spline_interpretability.md",
        ROOT / "results" / "interpretability" / "kan_gate_deep_dive_sdc" / "kan_gate_deep_dive_results.md",
        ROOT / "results" / "interpretability" / "kan_gate_stability_corrected_sdc" / "kan_gate_corrected_stability_results.md",
        ROOT / "results" / "rram_no_gpu_diagnostics" / "rram_sine_no_gpu_diagnostics.md",
        ROOT / "results" / "final_tables" / "rram_sine_adjusted_grid_distillation_results.md",
        ROOT / "results" / "final_tables" / "rram_sine_narrow100_results.md",
        ROOT / "results" / "sdc_grid_coverage" / "sdc_no_current_grid_coverage.md",
        ROOT / "results" / "final_tables" / "sdc_distillation_results.md",
        ROOT / "results" / "training_budget_audit" / "training_budget_audit.md",
    ]
    provenance_files = [
        ROOT / "results" / "sdc4dopant_preparation_summary.md",
        ROOT / "results" / "sdc4dopant_preparation_summary.json",
        ROOT / "results" / "external_dataset_inspection" / "southampton_download_suitability_summary.md",
        ROOT / "results" / "test_evaluation" / "sdc_per_curve_clipped_kangru_vs_gru_h116.md",
        ROOT / "results" / "test_evaluation" / "sdc_per_curve_clipped_kangru_vs_gru_h116.paired.csv",
        ROOT / "results" / "rram_no_gpu_diagnostics" / "rram_sine_no_gpu_diagnostics.json",
        ROOT / "results" / "final_tables" / "rram_sine_adjusted_grid_distillation.json",
        ROOT / "results" / "final_tables" / "rram_external_validation_100epoch.json",
        ROOT / "results" / "final_tables" / "rram_sine_narrow100.json",
        ROOT / "results" / "sdc_grid_coverage" / "sdc_no_current_grid_coverage.json",
        ROOT / "results" / "final_tables" / "sdc_distillation.json",
        ROOT / "results" / "interpretability" / "kan_gate_deep_dive_sdc" / "analysis_manifest.json",
        ROOT / "results" / "interpretability" / "kan_gate_stability_corrected_sdc" / "analysis_manifest.json",
    ]
    script_files = [
        ROOT / "scripts" / "finetune_sdc_to_sdc_cache.py",
        ROOT / "scripts" / "evaluate_d0132_compact_baseline.py",
        ROOT / "scripts" / "run_d0132_resistance_sequence.py",
        ROOT / "scripts" / "run_d0132_hybrid_residual.py",
        ROOT / "scripts" / "analyze_kan_gate_splines.py",
        ROOT / "scripts" / "evaluate_gate_interventions.py",
        ROOT / "scripts" / "run_kan_gate_deep_dive.py",
        ROOT / "scripts" / "analyze_kan_gate_stability_corrected.py",
        ROOT / "scripts" / "analyze_rram_sine_no_gpu_diagnostics.py",
        ROOT / "scripts" / "analyze_sdc_grid_coverage.py",
        ROOT / "scripts" / "audit_training_budgets.py",
        ROOT / "scripts" / "build_rram_external100_package.py",
        ROOT / "scripts" / "build_manuscript_ready_package.py",
    ]
    for path in table_files:
        if path.exists():
            copy_file(path, TABLE / path.name)
    for path in report_files:
        if path.exists():
            copy_file(path, REPORT / path.name)
    for path in provenance_files:
        if path.exists():
            copy_file(path, PROV / path.name)
    for path in script_files:
        if path.exists():
            copy_file(path, SCRIPT_OUT / path.name)


def write_readme_and_master() -> None:
    detailed = (ROOT / "results" / "final_tables" / "memristor_modeling_detailed_report.md").read_text(encoding="utf-8")
    figure_list = "\n".join(
        f"- `figures/{path.name}`" for path in sorted(FIG.glob("*.png"))
    )
    table_list = "\n".join(
        f"- `tables/{path.name}`" for path in sorted(TABLE.glob("*.csv"))
    )
    master = f"""# Manuscript-Ready Memristor Modeling Package

## Recommended Manuscript Title

No-Current Memory Inference for Memristor Dynamics: A Cross-Regime Benchmark of Recurrent and KAN-Gated Models

## Package Contents

This package contains the final manuscript source material: results tables,
figures, detailed experiment narrative, dataset/provenance notes, and scripts
needed to reproduce the final analyses.

## Core Claim

The paper is a benchmark/protocol paper. The main positive model result is that
clipped KAN-GRU is strongest on the original SDC no-current
state-characterisation benchmark. The broader contribution is the cross-regime
evaluation showing where that advantage holds and where dense recurrent,
static, or compact-model approaches are more appropriate.

## Figure Inventory

{figure_list}

## Table Inventory

{table_list}

## Manuscript Section Map

1. Introduction: motivate no-current memory inference and why measured
   previous current can create a teacher-forcing shortcut.
2. Datasets: original SDC, four-dopant SDC sine, RRAM-LSTM, Southampton D0132.
3. Models: static MLP, LSTM, GRU, KAN-LSTM, clipped KAN-LSTM, KAN-GRU,
   clipped KAN-GRU, fused/hybrid KAN-GRU, compact/VTEAM references.
4. Training and evaluation: curve-wise splits, 100 epochs for locked SDC runs,
   seeds, rollout/teacher-forced interpretation, timing/complexity logging.
5. Results: original SDC main result, perturbation, interpretability,
   four-dopant boundary, RRAM short-budget and grid/budget diagnostics,
   D0132 compact/hybrid boundary, transfer-learning boundary.
6. Discussion: clipped KAN-GRU is a strong original-SDC specialist; model
   advantage is regime-dependent; compact models remain important when
   calibrated.

## Final Detailed Report

{detailed}
"""
    (OUT / "MANUSCRIPT_READY_REPORT.md").write_text(master, encoding="utf-8")

    readme = """# Memristor Manuscript Upload Package

Open `MANUSCRIPT_READY_REPORT.md` first. It contains the complete result
narrative and points to every table and figure in this package.

Use `figures/` for manuscript figures and `tables/` for manuscript tables.
Use `reports/` for detailed subsection reports and `provenance/` for dataset
and evaluation traceability.

The final zip uses forward-slash archive paths and is safe to upload to Kaggle
or share as a manuscript-support package.
"""
    (OUT / "README_UPLOAD.md").write_text(readme, encoding="utf-8")


def make_zip() -> None:
    if ZIP_OUT.exists():
        ZIP_OUT.unlink()
    files = sorted(path for path in OUT.rglob("*") if path.is_file())
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, path.relative_to(OUT).as_posix())
    with zipfile.ZipFile(ZIP_OUT) as zf:
        bad = [name for name in zf.namelist() if "\\" in name]
        if bad:
            raise RuntimeError(f"bad archive paths: {bad[:3]}")


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    FIG.mkdir(parents=True)
    TABLE.mkdir(parents=True)
    REPORT.mkdir(parents=True)
    PROV.mkdir(parents=True)
    SCRIPT_OUT.mkdir(parents=True)

    plot_main_sdc()
    plot_runtime_tradeoff()
    plot_four_dopant()
    plot_rram()
    plot_rram_adjusted_grid_distillation()
    plot_rram_sine_narrow100()
    plot_sdc_distillation()
    plot_perturbation()
    plot_d0132_hybrid()
    plot_transfer()
    plot_sdc_prediction_trace()
    copy_existing_figures()
    copy_tables_and_reports()
    write_readme_and_master()
    make_zip()
    print(json.dumps({"package_dir": str(OUT), "zip": str(ZIP_OUT), "bytes": ZIP_OUT.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
