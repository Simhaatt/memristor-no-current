# Reproduction map

Run commands from the repository root after editable installation. All original CLI flags are available with `--help`. The `scripts/` names and historical checkpoint locations are retained so the original workflow remains inspectable.

| Result | Released CSVs | Generator/evaluator |
|---|---|---|
| Main SDC table | `results/final_tables/sdc_locked_test_results.csv` | `scripts/evaluate_sequence_checkpoints.py`, `scripts/finalize_sdc_tables.py` |
| Training time and model size | `results/final_tables/sdc_training_complexity_timing.csv` | `scripts/finalize_sdc_tables.py`; original training logs |
| 128/256/512 sensitivity and CI | `revision_experiments/window_sensitivity_*.csv` | `scripts/finalize_revision_window_sensitivity.py` |
| Full trajectory / quarter errors / hidden norms | `reviewer_experiments/*.csv` | `scripts/run_reviewer_full_trajectory.py` |
| Paired t/Wilcoxon/sign tests | `results/test_evaluation/sdc_per_curve_clipped_kangru_vs_gru_h116.json` | `scripts/evaluate_sdc_per_curve_significance.py` |
| Four-dopant folds | `results/final_tables/sdc4dopant_corrected_test_summary.csv` | `scripts/evaluate_sequence_checkpoints.py`; each cache evaluated separately |
| External RRAM, 100 epochs | `results/final_tables/rram_external_validation_100epoch_*.csv` | `scripts/run_rram_lstm_sequence.py` |
| RRAM grid / distillation | `results/final_tables/rram_sine*csv` | `scripts/run_rram_sine_grid_distill_batch.py` |
| SDC distillation | `results/final_tables/sdc_distillation*csv` | `scripts/run_sdc_distillation.py` |
| Gate curves and interventions | `results/interpretability/` | `scripts/run_kan_gate_deep_dive.py`, `scripts/analyze_kan_gate_stability_corrected.py`, `scripts/evaluate_gate_interventions.py` |
| D0132 compact / residual | `results/d0132_hybrid_residual/`, `results/compact_model_baselines/` | `scripts/run_d0132_hybrid_residual.py`, `scripts/evaluate_d0132_compact_baseline.py` |

## Quick checks without data or GPU

```bash
python analysis/verify_release.py
python analysis/make_figures.py
```

The first verifies checksums and the 18-run/162-trajectory result matrix. The second rebuilds the sensitivity R2/RMSE figure from CSV. Every RMSE in that figure uses standardized target units.

## Original SDC training example

```bash
python data/download.py --all
python data/preprocess.py --regime all
python scripts/run_sdc_sequence.py --cache data/processed/sdc_state_splits.npz --variants clipped_kan_gru_h32 --seeds 0 1 2 --epochs 100 --window 256 --stride 256 --eval-stride 512 --feature-mode no_current --rollout-eval --batch-size 32 --lr 0.001 --weight-decay 0 --grad-clip 1 --device cuda --save-models --out regenerated/sdc_kan.json
```

Dense GRU h116 uses batch 64; model definitions and capacities are in `src/memristorkan/models/recurrent.py`. Cache statistics use only training curves. Hidden state resets at independent training windows. The complete-trajectory evaluator initializes it once per held-out trajectory.

For a full re-evaluation of the sensitivity matrix:

```bash
python scripts/finalize_revision_window_sensitivity.py --device cpu
python scripts/run_reviewer_full_trajectory.py --device cpu
```

These use the archived final-epoch checkpoints. They require regenerated caches and downloaded original SDC files. The evaluation stride is twice the window length, and sample/update budgets differ; details are recorded in the result tables.

## Other regimes

Four-dopant training uses the same SDC runner with the `sdc4dopant_pooled.npz` or `sdc4dopant_holdout_{w,sn,cr,c}.npz` cache. The exact archived metadata specify the folds. The reported Cr fold contains the available seeds only; consult `runs` fields rather than assuming three seeds in every fold.

RRAM runs use `--task random_walk`, `--task sine`, or `--task anneal` with `scripts/run_rram_lstm_sequence.py --epochs 100 --seeds 0 1 2`. D0132 uses the supplied measured/compact/VTEAM tables. Protocols and model selection are defined by their original runners, and differ from SDC; training logs contain the actual settings.

## Figures and completeness

The historical figure assembler is `scripts/build_manuscript_ready_package.py`. It combines original analysis CSVs and some exported measurement traces. Those raw-derived traces are not republished here; regenerate them with `scripts/export_manuscript_trace_csvs.py` after obtaining the source data. Some export routines retrain the D0132 residual component because that historical runner did not save every learned residual checkpoint. This is listed as an artifact gap rather than silently presenting a retrained model as the original weights.

Published figure numbers must be checked against the final proof: that document is not included here. The existing generated images are reference artifacts, not a claim that all typeset figures have been matched. Axis units should be taken from the CSV/loader convention above. Do not relabel standardized targets as amperes without performing the physical conversion.
