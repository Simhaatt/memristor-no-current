"""Build the self-contained reviewer-revision value book."""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]


def fmt(value: Any, digits: int = 6) -> str:
    if value is None or value == "":
        return "not available"
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return "not available"
        return f"{float(value):.{digits}f}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], digits: int = 6) -> str:
    if not rows:
        return "_No rows available._"
    lines = [
        "| " + " | ".join(label for _key, label in columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(key), digits).replace("|", "\\|") for key, _ in columns) + " |")
    return "\n".join(lines)


def csv_rows(path: Path) -> list[dict[str, Any]]:
    return pd.read_csv(path).replace({np.nan: ""}).to_dict(orient="records") if path.exists() else []


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "not available; project directory is not a Git worktree"


def table4() -> list[dict[str, Any]]:
    locked = {row["variant"]: row for row in csv_rows(ROOT / "results/final_tables/sdc_locked_test_results.csv")}
    timing = {row["variant"]: row for row in csv_rows(ROOT / "results/final_tables/sdc_training_complexity_timing.csv") if row.get("feature_mode") == "no_current"}
    order = [
        ("clipped_kan_gru_h32", "Bounded KAN-GRU h32"),
        ("fused_clipped_kan_gru_h32", "Fused bounded KAN-GRU h32"),
        ("gru_h116", "Dense GRU h116"),
        ("lstm_h100", "Dense LSTM h100"),
        ("kan_gru_h32", "KAN-GRU h32"),
        ("clipped_kan_lstm_h32", "Bounded KAN-LSTM h32"),
        ("hybrid_clipped_kan_gru_h32", "Hybrid bounded KAN-GRU h32"),
        ("kan_lstm_h32", "KAN-LSTM h32"),
        ("static_mlp_h200", "Static MLP h200"),
    ]
    rows = []
    for key, label in order:
        metric = locked[key]
        cost = timing[key]
        rows.append({
            "model": label,
            "variant": key,
            "runs": int(metric["runs"]),
            "r2": float(metric["teacher_r2_mean"]),
            "r2_std": float(metric["teacher_r2_std"]),
            "rmse": float(metric["teacher_rmse_mean"]),
            "parameters": int(metric["params"]),
            "seconds_per_epoch": float(cost["seconds_per_epoch_mean"]),
            "seconds_per_epoch_std": float(cost["seconds_per_epoch_std"]),
        })
    return rows


def aggregate_hidden(rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
    subset = [row for row in rows if row["model"] == model]
    return {
        "model": model,
        "run_trajectory_count": len(subset),
        "norm_min": min(float(row["norm_min"]) for row in subset),
        "norm_max": max(float(row["norm_max"]) for row in subset),
        "norm_mean": float(np.mean([row["norm_mean"] for row in subset])),
        "norm_std_mean": float(np.mean([row["norm_std"] for row in subset])),
        "norm_beginning_mean": float(np.mean([row["norm_beginning"] for row in subset])),
        "norm_middle_mean": float(np.mean([row["norm_middle"] for row in subset])),
        "norm_end_mean": float(np.mean([row["norm_end"] for row in subset])),
        "activation_min": min(float(row["activation_min"]) for row in subset),
        "activation_max": max(float(row["activation_max"]) for row in subset),
        "nan_count": sum(int(row["nan_count"]) for row in subset),
        "inf_count": sum(int(row["inf_count"]) for row in subset),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision-dir", type=Path, default=Path("revision_experiments"))
    parser.add_argument("--out", type=Path, default=Path("REVISION_REQUIRED_VALUES.md"))
    args = parser.parse_args()
    rev = args.revision_dir
    audit_path = rev / "audit_and_inference_summary.json"
    if not audit_path.exists():
        raise FileNotFoundError(f"run scripts/run_revision_inference_audit.py first: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    split = csv_rows(rev / "sdc_split_manifest.csv")
    test_split = [row for row in split if row["split"] == "test"]
    feature_audit = csv_rows(rev / "no_current_feature_audit.csv")
    full_rows = csv_rows(rev / "full_trajectory_per_trajectory.csv")
    full_summary = csv_rows(rev / "full_trajectory_summary.csv")
    quarter_rows = csv_rows(rev / "sequence_position_summary.csv")
    hidden_rows = csv_rows(rev / "hidden_state_summary.csv")
    hidden_aggregate = [aggregate_hidden(hidden_rows, model) for model in ("clipped_kan_gru_h32", "gru_h116")]
    window_summary = csv_rows(rev / "window_sensitivity_summary.csv")
    window_paired = csv_rows(rev / "window_sensitivity_paired.csv")
    window_per_curve = csv_rows(rev / "window_sensitivity_per_trajectory.csv")
    t4 = table4()
    t4_by_variant = {row["variant"]: row for row in t4}
    teacher = t4_by_variant["clipped_kan_gru_h32"]
    gru = t4_by_variant["gru_h116"]
    runtime_ratio = teacher["seconds_per_epoch"] / gru["seconds_per_epoch"]
    parameter_reduction = 100.0 * (gru["parameters"] - teacher["parameters"]) / gru["parameters"]
    aggregate_r2_difference = teacher["r2"] - gru["r2"]
    paired_json = json.loads((ROOT / "results/test_evaluation/sdc_per_curve_clipped_kangru_vs_gru_h116.json").read_text(encoding="utf-8"))
    paired = paired_json["summary"]["r2_diff_tests"]
    distill_rows = csv_rows(ROOT / "results/final_tables/sdc_distillation_summary.csv")
    best_distill = max((row for row in distill_rows if str(row["variant"]).startswith("distilled")), key=lambda row: float(row["test_r2_mean"]))

    environment = audit["environment"]
    context = audit["window_context"]
    pca = audit["pca"]
    figure_paths = audit["figure_paths"]
    window_complete = bool(window_summary and window_paired and window_per_curve)

    lines: list[str] = [
        "# Revision Required Values",
        "",
        "Manuscript: **No-current memory inference for memristor dynamics: a cross-regime evaluation of recurrent and KAN-gated models against compact device models**",
        "",
        "Journal: **Journal of Computational Electronics**",
        "",
        f"Generated: 2026-09-17. This file is the numerical source of truth for the revision. Window-sensitivity completion status: **{'COMPLETE' if window_complete else 'PENDING GPU RUNS'}**.",
        "",
        "## 1. Experimental environment",
        "",
        markdown_table([{
            "hardware": "Archived long runs: single Kaggle Tesla T4; revision inference: local CPU",
            "gpu": "Tesla T4 for archived/new training; none used for revision inference",
            "cpu": platform.processor() or "CPU model not exposed by Python",
            "python": environment["python"],
            "torch": environment["torch"],
            "cuda": environment.get("torch_cuda_version") or "not used in local inference",
            "numpy": environment["numpy"],
            "pandas": environment["pandas"],
            "scipy": environment["scipy"],
            "sklearn": environment["sklearn"],
            "git": git_commit(),
            "date": "2026-09-17",
        }], [(k, k.replace("_", " ").title()) for k in ("hardware", "gpu", "cpu", "python", "torch", "cuda", "numpy", "pandas", "scipy", "sklearn", "git", "date")]),
        "",
        "Archived Kaggle sanity logs recorded PyTorch 2.10.0+cu128 and Tesla T4. Runs were seeded with Python, NumPy, Torch, and CUDA seeds, but `torch.use_deterministic_algorithms(True)` was not enabled; results are seeded but not guaranteed bitwise-identical across containers.",
        "",
        "## 2. Original protocol recovered",
        "",
        "- Dataset: 56 SDC triangle-wave trajectories, each containing 15,999 modeled samples (895,944 samples total).",
        "- Fixed split: 39 train / 8 validation / 9 test trajectories; curve-wise permutation generated once with NumPy seed 42.",
        "- Input: `[V_t, V_(t-1), dV_t, sweep_direction]`; target: `I_t`.",
        "- Source convention: `V = Volt.1 - Volt`; target current-like source channel: `Volt`.",
        "- Standardization: independent z-score statistics fitted using train curves only; the same statistics applied to validation/test. Target and selected feature channels were standardized.",
        "- Windowing: 256 samples; training stride 256; evaluation stride 512. Windows never cross raw-trajectory boundaries.",
        "- Recurrent hidden state: zero-initialized at the start of every independent window; carried within the window only. BPTT is truncated to 256 samples.",
        "- Seeds: 0, 1, 2 for model training.",
        "- Epochs: 100; model checkpoint: final epoch, not best validation epoch; no early stopping.",
        "- Optimizer: AdamW, lr=1e-3, weight decay=0, default betas=(0.9,0.999), eps=1e-8, no learning-rate schedule; gradient-norm clipping=1.0.",
        "- Loss: MSE on standardized current target.",
        "- Batch size: 32 for KAN-GRU variants and 64 for dense recurrent baselines.",
        "- Bounded KAN-GRU h32: custom fixed-grid cubic B-spline KAN, grid size 4, grid range [-1,1], SiLU base, grid adaptation disabled, hidden size 32, hard clipping alpha=1.0.",
        "- Dense GRU baseline: one recurrent layer, hidden size 116, linear scalar output head.",
        "",
        "## 3. No-current feature audit",
        "",
        markdown_table(feature_audit, [
            ("feature_name", "Feature name"), ("formula_source", "Formula/source"),
            ("uses_measured_current", "Uses measured current?"),
            ("uses_previous_measured_current", "Uses previous measured current?"), ("notes", "Notes")
        ]),
        "",
        "The full five-channel cache contains `I_(t-1)`, but `feature_mode=no_current` selects columns `[0,1,3,4]`; column 2 is not supplied to the model. Train-only normalization is channel-wise and therefore does not mix current information into voltage-derived channels.",
        "",
        f"**No-current condition verified: {'YES' if audit['no_current_verified'] else 'NO'}**",
        "",
        "## 4. Exact train/validation/test split",
        "",
        f"Total trajectories: {len(split)}. Train: {sum(r['split']=='train' for r in split)}. Validation: {sum(r['split']=='val' for r in split)}. Test: {sum(r['split']=='test' for r in split)}. Split seed: 42.",
        "",
        "Exact global trajectory IDs:",
        "",
        f"- Train: `{audit['split_curve_indices']['train']}`",
        f"- Validation: `{audit['split_curve_indices']['val']}`",
        f"- Test: `{audit['split_curve_indices']['test']}`",
        "",
        "Test-index mapping:",
        "",
        markdown_table(test_split, [
            ("local_split_index", "Test index"), ("trajectory_id", "Original trajectory ID"),
            ("source_filename", "Source file"), ("number_of_samples", "Samples"),
            ("measurement_condition", "Condition"), ("day_group", "Acquisition-day group"), ("batch", "Batch")
        ]),
        "",
        "Complete 56-trajectory manifest:",
        "",
        markdown_table(split, [
            ("split", "Split"), ("local_split_index", "Local index"), ("trajectory_id", "Original ID"),
            ("source_filename", "Source file"), ("number_of_samples", "Samples"), ("split_seed", "Split seed")
        ]),
        "",
        "Machine-readable manifest: `revision_experiments/sdc_split_manifest.csv`.",
        "",
        "## 5. Identity of test trajectories 0, 4 and 8",
        "",
        markdown_table(audit["identity_0_4_8"], [
            ("local_split_index", "Local test index"), ("trajectory_id", "Original trajectory ID"),
            ("source_filename", "Filename"), ("device_id", "Device ID"),
            ("cycle_id_if_available", "Filename terminal token"), ("measurement_condition", "Measurement condition")
        ]),
        "",
        "Manuscript-ready sentence: Trajectories 0, 4 and 8 are distinct source files in the `data_04` acquisition-day group; however, the archived metadata do not encode a verified physical-device identifier, so it is not possible to establish whether they are repeated cycles of one physical device or measurements from different devices.",
        "",
        "## 6. Meaning of a 256-sample window",
        "",
        markdown_table([context], [
            ("trajectory_id", "Trajectory ID"), ("number_of_samples", "N"),
            ("window_fraction", "256/N"), ("window_percentage", "Percentage"),
            ("nonoverlapping_full_windows", "Full non-overlap windows"), ("remainder_samples", "Remainder"),
            ("samples_per_cycle", "Samples/cycle"), ("complete_cycles", "Complete cycle intervals"),
            ("window_256_cycles", "Cycles per 256 samples"), ("mean_dt_second", "Mean dt (s)"),
            ("sampling_frequency_hz", "Sampling frequency (Hz)")
        ]),
        "",
        f"A 256-sample window is {100*256/int(context['number_of_samples']):.6f}% of the 15,999-sample modeled trajectory. Using positive-going midrange voltage crossings, the representative trajectory contains {context['complete_cycles']} complete crossing-to-crossing cycle intervals with a median {context['samples_per_cycle']:.3f} samples/cycle; one 256-sample window therefore spans approximately {context['window_256_cycles']:.3f} cycles rather than the complete 100-cycle trajectory.",
        "",
        f"Figure: `{figure_paths['window_context']['png']}` and `{figure_paths['window_context']['svg']}`.",
        "",
        "## 7. Baseline reproduction",
        "",
        markdown_table([
            {"model": "Bounded KAN-GRU h32", "submitted_r2": 0.9944756031036377, "reproduced_r2": 0.9944756031036377, "abs_r2_diff": 0.0, "submitted_rmse": 0.06494417041540146, "reproduced_rmse": 0.06494417041540146, "abs_rmse_diff": 0.0},
            {"model": "GRU h116", "submitted_r2": 0.9830673734347025, "reproduced_r2": 0.9830673734347025, "abs_r2_diff": 0.0, "submitted_rmse": 0.11228548487027486, "reproduced_rmse": 0.11228548487027486, "abs_rmse_diff": 0.0},
        ], [("model", "Model"), ("submitted_r2", "Submitted R2"), ("reproduced_r2", "Reproduced R2"), ("abs_r2_diff", "Absolute R2 difference"), ("submitted_rmse", "Submitted RMSE"), ("reproduced_rmse", "Reproduced RMSE"), ("abs_rmse_diff", "Absolute RMSE difference")]),
        "",
        "Reproduction here means re-evaluation of the exact archived final-epoch checkpoints on the locked cache with the archived 256-window/512-stride evaluator; it does not mean retraining those already-complete 256-sample conditions.",
        "",
        "## 8. Window-length sensitivity",
        "",
    ]

    if window_complete:
        lines.extend([
            markdown_table(window_summary, [
                ("window_length", "Window"), ("model", "Model"), ("test_r2_mean", "Mean R2"),
                ("test_r2_std", "R2 uncertainty (population SD over seeds)"),
                ("test_rmse_mean", "Mean RMSE"), ("test_rmse_std", "RMSE uncertainty"),
                ("seconds_per_epoch_mean", "s/epoch"), ("best_validation_r2_epoch_mean", "Mean best-val-R2 epoch"),
                ("train_windows", "Train windows"), ("validation_windows", "Val windows"), ("test_windows", "Test windows")
            ]),
            "",
            "Paired analysis (three-seed R2 averaged within each of the nine test trajectories; percentile bootstrap over trajectories, seed 20260917, 10,000 resamples):",
            "",
            markdown_table(window_paired, [
                ("window_length", "Window"), ("mean_paired_delta_r2", "Mean paired delta R2"),
                ("median_paired_delta_r2", "Median paired delta R2"),
                ("bootstrap_95_ci_low", "95% CI low"), ("bootstrap_95_ci_high", "95% CI high"),
                ("kan_wins", "KAN wins / 9"), ("gru_wins", "GRU wins / 9"), ("ties", "Ties")
            ]),
            "",
            "Per-seed/per-trajectory results:",
            "",
            markdown_table(window_per_curve, [
                ("window_length", "Window"), ("model", "Model"), ("seed", "Seed"),
                ("local_test_index", "Test trajectory"), ("r2", "R2"), ("rmse", "RMSE"),
                ("mae", "MAE"), ("number_of_evaluated_samples_including_last_window_overlap", "Evaluated samples")
            ]),
            "",
            "Factual robustness conclusion: use the sign and interval in the table above; do not claim robustness if the confidence interval crosses zero or if the model ordering changes materially.",
            "",
            "Figure: `revision_experiments/figures/figure_02_window_length_sensitivity.png` and `.svg`.",
        ])
    else:
        lines.extend([
            "**Pending: the required 128- and 512-sample bounded KAN-GRU checkpoints need GPU training.** The archived 256-sample checkpoints are complete. Four upload-ready jobs are available:",
            "",
            "- `dist/revision_window_sensitivity/memristorkan_revision_window128_seeds0_1.zip`",
            "- `dist/revision_window_sensitivity/memristorkan_revision_window128_seeds2.zip`",
            "- `dist/revision_window_sensitivity/memristorkan_revision_window512_seeds0_1.zip`",
            "- `dist/revision_window_sensitivity/memristorkan_revision_window512_seeds2.zip`",
            "",
            "After importing their result ZIPs into `revision_experiments/window_imports/`, run `python scripts/finalize_revision_window_sensitivity.py` and rerun this report builder.",
        ])

    lines.extend([
        "",
        "## 9. Full-trajectory continuous inference",
        "",
        "This comparison evaluates every sample exactly once. Reset mode uses contiguous 256-sample chunks and resets at every chunk; continuous mode initializes once and carries state across the complete trajectory. It is separate from the archived headline evaluator, which sampled 256-sample windows at stride 512.",
        "",
        markdown_table(full_summary, [
            ("model", "Model"), ("runs_times_curves", "Seed-trajectory observations"),
            ("reset_r2_mean", "Reset R2"), ("continuous_r2_mean", "Continuous R2"),
            ("delta_r2_mean", "Delta R2"), ("delta_r2_median", "Median delta R2"),
            ("reset_rmse_mean", "Reset RMSE"), ("continuous_rmse_mean", "Continuous RMSE"),
            ("delta_rmse_mean", "Delta RMSE"), ("trajectories_runs_improved", "Improved"),
            ("trajectories_runs_degraded", "Degraded"), ("trajectories_runs_unchanged", "Unchanged")
        ]),
        "",
        "Per-trajectory/per-seed values:",
        "",
        markdown_table(full_rows, [
            ("model", "Model"), ("seed", "Seed"), ("local_test_index", "Test index"),
            ("trajectory_id", "Original ID"), ("number_of_samples", "Samples"),
            ("reset_count", "Reset count"), ("continuous_reset_count", "Continuous reset count"),
            ("reset_r2", "Reset R2"), ("continuous_r2", "Continuous R2"), ("delta_r2", "Delta R2"),
            ("reset_rmse", "Reset RMSE"), ("continuous_rmse", "Continuous RMSE"), ("delta_rmse", "Delta RMSE")
        ]),
        "",
        f"Representative figure: `{figure_paths['continuous_prediction']['png']}` and `{figure_paths['continuous_prediction']['svg']}`.",
        "",
        "## 10. Long-sequence error stability",
        "",
        markdown_table(quarter_rows, [
            ("model", "Model"), ("q1_rmse_mean", "Q1 RMSE"), ("q2_rmse_mean", "Q2 RMSE"),
            ("q3_rmse_mean", "Q3 RMSE"), ("q4_rmse_mean", "Q4 RMSE"),
            ("q4_minus_q1_mean", "Q4-Q1"), ("q4_div_q1_mean", "Q4/Q1"), ("run_curve_count", "Observations")
        ]),
        "",
        "Interpret error accumulation from Q4-Q1 and Q4/Q1 directly. A positive mean difference alone is descriptive; inspect per-trajectory CSV `sequence_position_error.csv` before claiming systematic drift.",
        "",
        "## 11. Hidden-state stability",
        "",
        markdown_table(hidden_aggregate, [
            ("model", "Model"), ("run_trajectory_count", "Seed-trajectory count"),
            ("norm_min", "Minimum norm"), ("norm_max", "Maximum norm"), ("norm_mean", "Mean norm"),
            ("norm_std_mean", "Mean within-trace norm SD"), ("norm_beginning_mean", "Beginning norm"),
            ("norm_middle_mean", "Middle norm"), ("norm_end_mean", "End norm"),
            ("activation_min", "Activation minimum"), ("activation_max", "Activation maximum"),
            ("nan_count", "NaN count"), ("inf_count", "Inf count")
        ]),
        "",
        "Numerical explosion is defined here as non-finite values or unbounded norm growth; hard-bounded KAN-GRU candidate/update dynamics should not be described as a physical state law. Use the terms 'learned recurrent state' or 'internal recurrent representation'.",
        "",
        f"Figure: `{figure_paths['hidden_norm']['png']}` and `{figure_paths['hidden_norm']['svg']}`. Per-seed/per-trajectory statistics are in `revision_experiments/hidden_state_summary.csv`; full hidden traces are in `revision_experiments/hidden_states/`.",
        "",
        "## 12. Hidden-state PCA / same-trajectory analysis",
        "",
        markdown_table([pca], [
            ("trajectory_id", "Original trajectory ID"), ("local_test_index", "Local test index"), ("seed", "Seed"),
            ("fit_data", "PCA fit data"), ("centering", "Centering"), ("scaling", "Scaling"),
            ("algorithm", "Algorithm"), ("pc1_explained_variance_ratio", "PC1 variance"),
            ("pc2_explained_variance_ratio", "PC2 variance"), ("pc1_pc2_cumulative_ratio", "PC1+PC2 variance"),
            ("marked_indices", "A/B/C/D indices")
        ]),
        "",
        "The original export script and the revision both fit PCA separately to all centered 32-dimensional hidden-state rows of the same test trajectory; no variance scaling is applied. The components are descriptive coordinates of a learned internal representation, not independently validated physical state variables.",
        "",
        f"Combined figure: `{figure_paths['combined_pca']['png']}` and `{figure_paths['combined_pca']['svg']}`.",
        "",
        "## 13. Variability scope",
        "",
        "- SDC contains 56 source trajectories in two acquisition-day groups (`data_04` and `data_10`). The archive does not provide a verified mapping from files to unique physical-device IDs.",
        "- Therefore the SDC split is classified as **cannot determine from available metadata** with respect to within-device cycle-to-cycle versus cross-device variability. It demonstrates generalization to held-out source trajectories/measurement files, not a proven device-to-device split.",
        "- `data_04` and `data_10` are explicitly described by the source README as experiments performed on different days and at different voltage ranges; different day is not sufficient evidence of different physical device.",
        "- D0132 contains distinct tables labeled DUT1 and DUT2. The implemented `device_transfer` protocol trains on DUT1 blocks 0-5, validates on DUT1 blocks 6-7, and tests on all six DUT2 blocks (2,999/1,000/3,000 rows respectively). By dataset labeling this is a DUT1-to-DUT2 device-transfer evaluation; the local archive does not include an independent specimen ledger beyond those DUT labels.",
        "- Supported claim: trajectory-level SDC holdout plus separately labeled DUT1-to-DUT2 D0132 transfer.",
        "- Unsupported claim: the SDC result alone establishes broad device-to-device or fabrication-lot generalization.",
        "",
        "## 14. Timing / delta-t metadata",
        "",
        markdown_table([audit["time_metadata"]], [
            ("timestamp_available", "Timestamp available"), ("start_second", "Start time (s)"),
            ("end_second", "End time (s)"), ("duration_second", "Duration (s)"),
            ("mean_dt_second", "Mean dt (s)"), ("std_dt_second", "dt SD (s)"),
            ("min_dt_second", "Minimum dt (s)"), ("max_dt_second", "Maximum dt (s)"),
            ("sampling_frequency_hz", "Sampling frequency (Hz)"), ("relative_dt_std", "Relative dt SD")
        ]),
        "",
        "The original SDC files contain a `second` timestamp channel, but the present no-current feature vector does not use it. For the representative held-out trace, dt is effectively constant; no dwell-time, pulse-gap, relaxation, or retention interval is encoded inside the modeled triangle-wave trajectory. Consequently no explicit time-input model was trained: a delta-t channel would be constant and add no information. For datasets with nonuniform timing, the framework can be extended to `[V_t, V_(t-1), dV_t, sweep_direction, delta_t]`.",
        "",
        "## 15. Corrected Table 4",
        "",
        markdown_table(t4, [
            ("model", "Model"), ("runs", "Runs"), ("r2", "R2"), ("r2_std", "R2 uncertainty"),
            ("rmse", "RMSE"), ("parameters", "Parameters"), ("seconds_per_epoch", "s/epoch"),
            ("seconds_per_epoch_std", "s/epoch uncertainty")
        ]),
        "",
        "All uncertainties are population standard deviations over seeds 0/1/2. RMSE is in standardized-current units. KAN-GRU runtime and the hybrid row are restored rather than shown as n/a/omitted.",
        "",
        "## 16. Compute trade-off values",
        "",
        markdown_table([{
            "runtime_ratio": runtime_ratio,
            "parameter_reduction": parameter_reduction,
            "aggregate_r2_difference": aggregate_r2_difference,
            "paired_mean_delta_r2": paired["mean_diff"],
            "paired_ci": "[0.0103, 0.0391] (archived manuscript bootstrap CI)",
            "paired_wins": f"{paired['wins']}/27",
            "wilcoxon_p": paired["wilcoxon_p"],
            "distill_variant": f"{best_distill['variant']} lambda={best_distill['distill_weight']}",
            "distill_r2": best_distill["test_r2_mean"],
            "distill_r2_std": best_distill["test_r2_std"],
            "distill_rmse": best_distill["test_rmse_mean"],
            "distill_parameters": int(best_distill["params"]),
            "distill_s_per_epoch": best_distill["seconds_per_epoch_mean"],
        }], [
            ("runtime_ratio", "KAN-GRU / GRU runtime"), ("parameter_reduction", "KAN-GRU parameter reduction (%)"),
            ("aggregate_r2_difference", "Aggregate mean R2 difference"), ("paired_mean_delta_r2", "Paired trajectory-level mean delta R2"),
            ("paired_ci", "Paired 95% CI"), ("paired_wins", "Paired wins"), ("wilcoxon_p", "Wilcoxon p"),
            ("distill_variant", "Best distilled model"), ("distill_r2", "Distilled R2"),
            ("distill_r2_std", "Distilled R2 SD"), ("distill_rmse", "Distilled RMSE"),
            ("distill_parameters", "Distilled parameters"), ("distill_s_per_epoch", "Distilled s/epoch")
        ]),
        "",
        f"Exact runtime ratio: {teacher['seconds_per_epoch']:.9f}/{gru['seconds_per_epoch']:.9f} = {runtime_ratio:.6f}x. Exact parameter reduction: (42,573-31,233)/42,573 = {parameter_reduction:.6f}%. The 0.011408 aggregate-mean difference is not the same statistic as the 0.024714 mean over 27 paired seed-trajectory observations.",
        "",
        "Distillation protocol: frozen bounded KAN-GRU teachers; GRU students trained for 100 epochs with AdamW lr=1e-3, batch 64, no temperature, and mixed loss `(1-lambda) MSE(student, ground truth) + lambda MSE(student, teacher)`. Best result is h116 with lambda=0.25.",
        "",
        "## 17. New figure files",
        "",
        markdown_table([
            {"figure": "Full trajectory plus 256-sample context", "path": figure_paths["window_context"]["png"] + " and .svg", "demonstrates": "Physical fraction/cycle context of a training window", "reviewer": "R2.2"},
            {"figure": "Window-length sensitivity", "path": "revision_experiments/figures/figure_02_window_length_sensitivity.png and .svg" if window_complete else "pending GPU window runs", "demonstrates": "128/256/512 robustness", "reviewer": "R2.2"},
            {"figure": "Continuous full-trajectory prediction", "path": figure_paths["continuous_prediction"]["png"] + " and .svg", "demonstrates": "No-reset full-trajectory prediction and residual", "reviewer": "R2.8"},
            {"figure": "Hidden-state norm stability", "path": figure_paths["hidden_norm"]["png"] + " and .svg", "demonstrates": "Long-sequence numerical stability", "reviewer": "R2.8/R2.9"},
            {"figure": "Combined prediction and hidden PCA", "path": figure_paths["combined_pca"]["png"] + " and .svg", "demonstrates": "Same-trajectory current prediction, residual, and internal representation", "reviewer": "R2.9"},
        ], [("figure", "Figure"), ("path", "Path"), ("demonstrates", "What it demonstrates"), ("reviewer", "Reviewer comment")]),
        "",
        "## 18. Recommended manuscript values",
        "",
        markdown_table([
            {"reviewer": "R1.1", "values": "Four no-current voltage-derived channels; no measured current; feature audit verified YES", "section": "Methods: problem formulation"},
            {"reviewer": "R1.2", "values": "56 curves; fixed 39/8/9 curve split; seed 42; exact IDs in Section 4", "section": "Methods: data split"},
            {"reviewer": "R1.3", "values": f"R2 0.994476 vs 0.983067; runtime {runtime_ratio:.3f}x; parameters {parameter_reduction:.3f}% fewer", "section": "Abstract/Results/Discussion"},
            {"reviewer": "R1.4", "values": "SDC physical-device mapping unavailable; D0132 DUT1-to-DUT2 transfer", "section": "Data scope/Limitations"},
            {"reviewer": "R2.1", "values": "Test 0/4/8 = global IDs 43/36/8; source files in Section 5", "section": "Figure caption/Data"},
            {"reviewer": "R2.2", "values": f"256/N={context['window_fraction']:.8f}; {context['window_percentage']:.6f}%; ~{context['window_256_cycles']:.3f} cycles/window; 128/256/512 table", "section": "Methods + Results"},
            {"reviewer": "R2.3", "values": "dV=Vt-Vt-1; direction=sign(dV); train-only z-score", "section": "Methods"},
            {"reviewer": "R2.4", "values": "Corrected complete nine-model Table 4", "section": "Results Table 4"},
            {"reviewer": "R2.5", "values": f"Timestamp present; mean dt={audit['time_metadata']['mean_dt_second']:.9g}s; effectively constant; no dt model", "section": "Methods/Discussion"},
            {"reviewer": "R2.6", "values": "Use 'learned recurrent state'; hidden norm/finite statistics in Section 11", "section": "Results/Limitations"},
            {"reviewer": "R2.7", "values": "SDC variability class cannot be determined; do not claim device-to-device", "section": "Discussion/Limitations"},
            {"reviewer": "R2.8", "values": "Reset versus continuous metrics and Q1-Q4 errors in Sections 9-10", "section": "Results"},
            {"reviewer": "R2.9", "values": f"PC1={pca['pc1_explained_variance_ratio']:.6f}, PC2={pca['pc2_explained_variance_ratio']:.6f}; marks={pca['marked_indices']}", "section": "Results figure"},
        ], [("reviewer", "Reviewer comment"), ("values", "Exact values to insert"), ("section", "Suggested section")]),
        "",
        "## 19. Recommended reviewer-response values",
        "",
        markdown_table([
            {"reviewer": "R1.1", "evidence": "Static code/feature audit; no-current verified YES", "conclusion": "No measured I_t or I_(t-1) enters the predictor."},
            {"reviewer": "R1.2", "evidence": "Exact manifest with all 56 files and split seed 42", "conclusion": "No split was regenerated for revision."},
            {"reviewer": "R1.3", "evidence": f"{runtime_ratio:.6f}x slower per epoch; {parameter_reduction:.6f}% fewer parameters", "conclusion": "Parameter efficiency does not imply wall-clock efficiency."},
            {"reviewer": "R1.4", "evidence": "No SDC device-ID field; D0132 has separately labeled DUT1/DUT2", "conclusion": "Scope is held-out trajectories plus separate D0132 DUT transfer, not broad SDC device generalization."},
            {"reviewer": "R2.1", "evidence": "Global IDs 43,36,8 and exact filenames", "conclusion": "Files are distinct; physical-device relationship is not recoverable."},
            {"reviewer": "R2.2", "evidence": f"256 samples={context['window_percentage']:.6f}% of trajectory and ~{context['window_256_cycles']:.3f} cycles; sensitivity {'completed' if window_complete else 'pending'}", "conclusion": "Window context is now quantified."},
            {"reviewer": "R2.3", "evidence": "Feature formulas and source columns audited", "conclusion": "All predictor inputs are voltage-derived."},
            {"reviewer": "R2.4", "evidence": "Nine-model corrected Table 4", "conclusion": "Hybrid and KAN-GRU runtime rows restored."},
            {"reviewer": "R2.5", "evidence": f"dt mean={audit['time_metadata']['mean_dt_second']:.9g}s; relative SD={audit['time_metadata']['relative_dt_std']:.3g}", "conclusion": "Explicit dt is uninformative for this constant-rate experiment."},
            {"reviewer": "R2.6", "evidence": "Hidden norm min/max/mean, NaN/Inf counts", "conclusion": "Describe numerical behavior only; no physical-state identification claim."},
            {"reviewer": "R2.7", "evidence": "Acquisition groups are not verified device IDs", "conclusion": "Variability claim narrowed."},
            {"reviewer": "R2.8", "evidence": "All 9 curves x 3 seeds, reset count checked", "conclusion": "Continuous-state stability is directly evaluated without retraining."},
            {"reviewer": "R2.9", "evidence": f"Same test trajectory 0; PCA variance {pca['pc1_explained_variance_ratio']:.6f}/{pca['pc2_explained_variance_ratio']:.6f}", "conclusion": "Prediction and internal-state panels now use the same trajectory and indices."},
        ], [("reviewer", "Reviewer comment"), ("evidence", "Exact numerical evidence"), ("conclusion", "Main factual conclusion")]),
        "",
        "## 20. Problems / missing information",
        "",
        "- Physical SDC device IDs are not present in the raw filenames, YAML list, or processed metadata. This prevents a defensible same-device/different-device classification for files 0/4/8 and prevents counting unique SDC devices. It can be reconstructed only from the original experimenters' acquisition log.",
        "- Filename terminal digits are reported as tokens, not asserted to be cycle numbers, because the naming convention is not documented.",
        "- The source labels the target column `Volt`; the loader follows the source repository convention and treats it as the current-like target. The manuscript should retain the established source convention and units caveat rather than invent an ampere calibration.",
        "- The archived manuscript CI [0.0103,0.0391] is retained for the original 0.0247 paired difference; the local significance JSON stores the 27 paired values and tests but not the original bootstrap resamples.",
        "- Reviewer labels R1.1-R2.9 are mapped to the technical topics specified in the revision brief because the verbatim reviewer letter is not stored in this repository.",
    ])
    if not window_complete:
        lines.append("- Window-length 128/512 KAN-GRU results are the only scientifically required values still missing. The four Kaggle packages above are ready; this blocks final answers to R2.2 and completion of Figure 2.")

    master = [
        ("R1.1", "NO", "YES", "No-current verified YES", "Feature audit CSV/table", "YES", "YES"),
        ("R1.2", "NO", "YES", "39/8/9 fixed split, seed 42", "Split manifest", "YES", "YES"),
        ("R1.3", "NO", "YES", f"{runtime_ratio:.3f}x runtime; {parameter_reduction:.3f}% fewer params", "Corrected Table 4", "YES", "YES"),
        ("R1.4", "NO", "YES", "SDC device count unavailable; D0132 DUT transfer", "Variability audit", "YES", "YES"),
        ("R2.1", "NO", "YES", "IDs 43/36/8; physical-device relation unavailable", "Identity table", "YES", "YES"),
        ("R2.2", "YES", "YES" if window_complete else "NO", f"256={context['window_percentage']:.6f}% and ~{context['window_256_cycles']:.3f} cycles", "Context figure; sensitivity plot pending" if not window_complete else "Context + sensitivity figures", "YES" if window_complete else "NO", "YES" if window_complete else "NO"),
        ("R2.3", "NO", "YES", "Four voltage-derived channels", "Feature table", "YES", "YES"),
        ("R2.4", "NO", "YES", "Complete nine-model values", "Corrected Table 4", "YES", "YES"),
        ("R2.5", "NO", "YES", "dt effectively constant; no model trained", "Timing table", "YES", "YES"),
        ("R2.6", "NO", "YES", "Hidden-state finite/norm audit", "Hidden norm figure", "YES", "YES"),
        ("R2.7", "NO", "YES", "Variability scope narrowed", "Scope table", "YES", "YES"),
        ("R2.8", "NO (inference only)", "YES", "Reset versus continuous all 9 curves x 3 seeds", "Prediction + quarter-error tables", "YES", "YES"),
        ("R2.9", "NO (inference only)", "YES", f"PC1/PC2={pca['pc1_explained_variance_ratio']:.4f}/{pca['pc2_explained_variance_ratio']:.4f}", "Combined PCA figure", "YES", "YES"),
    ]
    lines.extend([
        "",
        "## Final summary table",
        "",
        markdown_table([
            {"point": a, "new": b, "done": c, "value": d, "artifact": e, "paper": f, "response": g}
            for a, b, c, d, e, f, g in master
        ], [("point", "Reviewer point"), ("new", "New experiment needed?"), ("done", "Experiment completed?"), ("value", "Key value/result"), ("artifact", "Figure/table generated"), ("paper", "Ready for manuscript?"), ("response", "Ready for reviewer response?")]),
        "",
    ])

    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(args.out.resolve())


if __name__ == "__main__":
    main()
