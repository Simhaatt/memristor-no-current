# Reviewer Experiment Results

Scope: Reviewer 2.2 window-length sensitivity and Reviewer 2.8 continuous complete-trajectory inference using the original submitted SDC formulation.

Completion: window sensitivity **COMPLETE**; full-trajectory inference **COMPLETE**.

## 1. Experimental setup

- Dataset: SDC state-characterisation; 56 trajectories of 15,999 modeled samples.
- Fixed split: 39 train / 8 validation / 9 test; split seed 42; test IDs `[43, 14, 54, 2, 36, 33, 1, 13, 8]`.
- Model seeds: 0, 1, 2.
- Original input formulation: `[V_t, V_(t-1), dV_t, sweep_direction]`; target `I_t`.
- Features and target: independently z-standardized with statistics fitted to training trajectories only.
- Loss: MSE on standardized target. Optimizer: AdamW, lr=1e-3, weight decay=0, default betas (0.9, 0.999), eps=1e-8, no scheduler, gradient-norm clip 1.0.
- Training: 100 epochs, final-epoch checkpoint, no early stopping. The best-validation epoch is reported diagnostically and was not used for checkpoint selection.
- Batch size: 32 for bounded KAN-GRU h32; 64 for dense GRU h116.
- Training hardware: Kaggle Tesla T4. Full-trajectory inference hardware: local CPU.
- Report-builder software: Python 3.14.3, PyTorch 2.12.0+cpu, NumPy 2.4.2. Archived Kaggle logs report PyTorch 2.10.0+cu128.
- Exact 256-sample checkpoints:
  - `results/kaggle_imported/kangru_sdc_100epoch_seed0/checkpoints/sequence/clipped_kan_gru_h32_no_current_seed0.pt`
  - `results/kaggle_imported/clipped_kangru_sdc_100epoch_seed1_2/checkpoints/sequence/clipped_kan_gru_h32_no_current_seed1.pt`
  - `results/kaggle_imported/clipped_kangru_sdc_100epoch_seed1_2/checkpoints/sequence/clipped_kan_gru_h32_no_current_seed2.pt`
  - `checkpoints/kaggle_imported/no_current_baselines_100epoch_3seed/sequence/gru_h116_no_current_seed0.pt`
  - `checkpoints/kaggle_imported/no_current_baselines_100epoch_3seed/sequence/gru_h116_no_current_seed1.pt`
  - `checkpoints/kaggle_imported/no_current_baselines_100epoch_3seed/sequence/gru_h116_no_current_seed2.pt`

## 2. Baseline 256-sample reproduction

| Model | Submitted R2 | Reproduced R2 | Difference | Submitted RMSE | Reproduced RMSE | Difference |
|---|---|---|---|---|---|---|
| Bounded KAN-GRU h32 | 0.994476 | 0.994476 | 0 | 0.064944 | 0.064944 | 0 |
| Dense GRU h116 | 0.983067 | 0.983067 | 0 | 0.112285 | 0.112285 | 0 |

The archived final-epoch checkpoints evaluated on the locked cache reproduce the submitted aggregate values exactly; retraining the 256-sample reference was unnecessary.

## 3. Window-length sensitivity - complete values

| Window | Model | Mean R2 | SD/uncertainty | Mean RMSE | RMSE SD | Mean MAE | Runtime/epoch |
|---|---|---|---|---|---|---|---|
| 128 | Bounded KAN-GRU h32 | 0.995437 | 0.000164 | 0.059322 | 0.001072 | 0.033821 | 150.582918 |
| 256 | Bounded KAN-GRU h32 | 0.994476 | 0.000165 | 0.064944 | 0.000976 | 0.038528 | 141.574131 |
| 512 | Bounded KAN-GRU h32 | 0.988630 | 0.003742 | 0.091899 | 0.015603 | 0.056498 | 149.083130 |
| 128 | Dense GRU h116 | 0.990827 | 0.000478 | 0.084092 | 0.002194 | 0.045720 | 0.846136 |
| 256 | Dense GRU h116 | 0.983067 | 0.005571 | 0.112285 | 0.017959 | 0.063353 | 0.612773 |
| 512 | Dense GRU h116 | 0.957201 | 0.010845 | 0.179449 | 0.022465 | 0.110999 | 0.723444 |

Uncertainty is the population standard deviation across seeds 0/1/2. RMSE and MAE are in standardized-target units.

Window-dependent exposure and compute accounting:

| Window | Model | Epochs | Train stride | Eval stride | Batch size | Samples/batch | Train windows | Validation windows | Test windows | Evaluated samples/trajectory | Batches/epoch | Optimizer steps | s/epoch | Mean total training s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | Bounded KAN-GRU h32 | 100 | 128 | 256 | 32 | 4096 | 4875 | 504 | 567 | 8064 | 153 | 15300 | 150.582918 | 15058.291797 |
| 256 | Bounded KAN-GRU h32 | 100 | 256 | 512 | 32 | 8192 | 2457 | 256 | 288 | 8192 | 77 | 7700 | 141.574131 | 14157.413090 |
| 512 | Bounded KAN-GRU h32 | 100 | 512 | 1024 | 32 | 16384 | 1248 | 136 | 153 | 8704 | 39 | 3900 | 149.083130 | 14908.312978 |
| 128 | Dense GRU h116 | 100 | 128 | 256 | 64 | 8192 | 4875 | 504 | 567 | 8064 | 77 | 7700 | 0.846136 | 84.613614 |
| 256 | Dense GRU h116 | 100 | 256 | 512 | 64 | 16384 | 2457 | 256 | 288 | 8192 | 39 | 3900 | 0.612773 | 61.277255 |
| 512 | Dense GRU h116 | 100 | 512 | 1024 | 64 | 32768 | 1248 | 136 | 153 | 8704 | 20 | 2000 | 0.723444 | 72.344388 |

The train/evaluation strides scale with window length, so each condition evaluates a slightly different sample count per trajectory: 8,064 at 128, 8,192 at 256, and 8,704 at 512. With 100 epochs fixed, the KAN-GRU receives 15,300, 7,700, and 3,900 optimizer steps, respectively (GRU: 7,700, 3,900, and 2,000). Therefore this is sensitivity of the complete fixed-epoch training protocol to sequence length, not an isolated sequence-length effect under a perfectly matched update budget.

## 4. Per-seed window results

| Model | Window | Seed | R2 | RMSE | Best epoch | Epochs | Batch | Train windows | Batches/epoch | Optimizer steps | Runtime/epoch | Total runtime |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Bounded KAN-GRU h32 | 128 | 0 | 0.995271 | 0.060398 | 99 | 100 | 32 | 4875 | 153 | 15300 | 156.649157 | 15664.915723 |
| Bounded KAN-GRU h32 | 128 | 1 | 0.995378 | 0.059709 | 75 | 100 | 32 | 4875 | 153 | 15300 | 156.773117 | 15677.311659 |
| Bounded KAN-GRU h32 | 128 | 2 | 0.995660 | 0.057860 | 99 | 100 | 32 | 4875 | 153 | 15300 | 138.326480 | 13832.648010 |
| Bounded KAN-GRU h32 | 256 | 0 | 0.994699 | 0.063624 | 98 | 100 | 32 | 2457 | 77 | 7700 | 141.321757 | 14132.175731 |
| Bounded KAN-GRU h32 | 256 | 1 | 0.994423 | 0.065257 | 100 | 100 | 32 | 2457 | 77 | 7700 | 141.523999 | 14152.399917 |
| Bounded KAN-GRU h32 | 256 | 2 | 0.994304 | 0.065951 | 99 | 100 | 32 | 2457 | 77 | 7700 | 141.876636 | 14187.663622 |
| Bounded KAN-GRU h32 | 512 | 0 | 0.988809 | 0.092477 | 92 | 100 | 32 | 1248 | 39 | 3900 | 149.471650 | 14947.165009 |
| Bounded KAN-GRU h32 | 512 | 1 | 0.993120 | 0.072507 | 100 | 100 | 32 | 1248 | 39 | 3900 | 149.286627 | 14928.662749 |
| Bounded KAN-GRU h32 | 512 | 2 | 0.983960 | 0.110714 | 99 | 100 | 32 | 1248 | 39 | 3900 | 148.491112 | 14849.111175 |
| Dense GRU h116 | 128 | 0 | 0.990814 | 0.084183 | 89 | 100 | 64 | 4875 | 77 | 7700 | 0.888430 | 88.842958 |
| Dense GRU h116 | 128 | 1 | 0.990249 | 0.086733 | 92 | 100 | 64 | 4875 | 77 | 7700 | 0.887874 | 88.787419 |
| Dense GRU h116 | 128 | 2 | 0.991419 | 0.081361 | 71 | 100 | 64 | 4875 | 77 | 7700 | 0.762105 | 76.210466 |
| Dense GRU h116 | 256 | 0 | 0.986506 | 0.101513 | 99 | 100 | 64 | 2457 | 39 | 3900 | 0.593169 | 59.316929 |
| Dense GRU h116 | 256 | 1 | 0.987487 | 0.097754 | 100 | 100 | 64 | 2457 | 39 | 3900 | 0.621651 | 62.165062 |
| Dense GRU h116 | 256 | 2 | 0.975210 | 0.137590 | 99 | 100 | 64 | 2457 | 39 | 3900 | 0.623498 | 62.349773 |
| Dense GRU h116 | 512 | 0 | 0.942413 | 0.209781 | 86 | 100 | 64 | 1248 | 20 | 2000 | 0.719389 | 71.938885 |
| Dense GRU h116 | 512 | 1 | 0.968115 | 0.156097 | 98 | 100 | 64 | 1248 | 20 | 2000 | 0.716237 | 71.623662 |
| Dense GRU h116 | 512 | 2 | 0.961076 | 0.172470 | 99 | 100 | 64 | 1248 | 20 | 2000 | 0.734706 | 73.470617 |

## 5. Per-trajectory window results

| Model | Window | Seed | Trajectory ID | R2 | RMSE | MAE | Evaluated samples | Best epoch | Checkpoint |
|---|---|---|---|---|---|---|---|---|---|
| Bounded KAN-GRU h32 | 128 | 0 | 43 | 0.992197 | 0.080167 | 0.037953 | 8064 | 99 | revision_experiments\window_imports\results_reviewer_window128_seeds0_1\checkpoints\window128_kan_seeds0_1\clipped_kan_gru_h32_no_current_seed0.pt |
| Bounded KAN-GRU h32 | 128 | 0 | 14 | 0.926839 | 0.071942 | 0.031468 | 8064 | 99 | revision_experiments\window_imports\results_reviewer_window128_seeds0_1\checkpoints\window128_kan_seeds0_1\clipped_kan_gru_h32_no_current_seed0.pt |
| Bounded KAN-GRU h32 | 128 | 0 | 54 | 0.998932 | 0.033352 | 0.023882 | 8064 | 99 | revision_experiments\window_imports\results_reviewer_window128_seeds0_1\checkpoints\window128_kan_seeds0_1\clipped_kan_gru_h32_no_current_seed0.pt |
| Bounded KAN-GRU h32 | 128 | 0 | 2 | 0.983681 | 0.061607 | 0.037387 | 8064 | 99 | revision_experiments\window_imports\results_reviewer_window128_seeds0_1\checkpoints\window128_kan_seeds0_1\clipped_kan_gru_h32_no_current_seed0.pt |
| Bounded KAN-GRU h32 | 128 | 0 | 36 | 0.990296 | 0.086850 | 0.049156 | 8064 | 99 | revision_experiments\window_imports\results_reviewer_window128_seeds0_1\checkpoints\window128_kan_seeds0_1\clipped_kan_gru_h32_no_current_seed0.pt |
| Bounded KAN-GRU h32 | 128 | 0 | 33 | 0.998913 | 0.043205 | 0.030280 | 8064 | 99 | revision_experiments\window_imports\results_reviewer_window128_seeds0_1\checkpoints\window128_kan_seeds0_1\clipped_kan_gru_h32_no_current_seed0.pt |
| Bounded KAN-GRU h32 | 128 | 0 | 1 | 0.988099 | 0.052331 | 0.035477 | 8064 | 99 | revision_experiments\window_imports\results_reviewer_window128_seeds0_1\checkpoints\window128_kan_seeds0_1\clipped_kan_gru_h32_no_current_seed0.pt |
| Bounded KAN-GRU h32 | 128 | 0 | 13 | 0.977049 | 0.046384 | 0.027864 | 8064 | 99 | revision_experiments\window_imports\results_reviewer_window128_seeds0_1\checkpoints\window128_kan_seeds0_1\clipped_kan_gru_h32_no_current_seed0.pt |
| Bounded KAN-GRU h32 | 128 | 0 | 8 | 0.998954 | 0.044961 | 0.031767 | 8064 | 99 | revision_experiments\window_imports\results_reviewer_window128_seeds0_1\checkpoints\window128_kan_seeds0_1\clipped_kan_gru_h32_no_current_seed0.pt |
| Bounded KAN-GRU h32 | 128 | 1 | 43 | 0.993041 | 0.075707 | 0.037310 | 8064 | 75 | revision_experiments\window_imports\results_reviewer_window128_seeds0_1\checkpoints\window128_kan_seeds0_1\clipped_kan_gru_h32_no_current_seed1.pt |
| Bounded KAN-GRU h32 | 128 | 1 | 14 | 0.921952 | 0.074306 | 0.034619 | 8064 | 75 | revision_experiments\window_imports\results_reviewer_window128_seeds0_1\checkpoints\window128_kan_seeds0_1\clipped_kan_gru_h32_no_current_seed1.pt |
| Bounded KAN-GRU h32 | 128 | 1 | 54 | 0.998871 | 0.034292 | 0.024523 | 8064 | 75 | revision_experiments\window_imports\results_reviewer_window128_seeds0_1\checkpoints\window128_kan_seeds0_1\clipped_kan_gru_h32_no_current_seed1.pt |
| Bounded KAN-GRU h32 | 128 | 1 | 2 | 0.984428 | 0.060180 | 0.039684 | 8064 | 75 | revision_experiments\window_imports\results_reviewer_window128_seeds0_1\checkpoints\window128_kan_seeds0_1\clipped_kan_gru_h32_no_current_seed1.pt |
| Bounded KAN-GRU h32 | 128 | 1 | 36 | 0.993565 | 0.070728 | 0.036798 | 8064 | 75 | revision_experiments\window_imports\results_reviewer_window128_seeds0_1\checkpoints\window128_kan_seeds0_1\clipped_kan_gru_h32_no_current_seed1.pt |
| Bounded KAN-GRU h32 | 128 | 1 | 33 | 0.999056 | 0.040262 | 0.029279 | 8064 | 75 | revision_experiments\window_imports\results_reviewer_window128_seeds0_1\checkpoints\window128_kan_seeds0_1\clipped_kan_gru_h32_no_current_seed1.pt |
| Bounded KAN-GRU h32 | 128 | 1 | 1 | 0.984859 | 0.059026 | 0.039364 | 8064 | 75 | revision_experiments\window_imports\results_reviewer_window128_seeds0_1\checkpoints\window128_kan_seeds0_1\clipped_kan_gru_h32_no_current_seed1.pt |
| Bounded KAN-GRU h32 | 128 | 1 | 13 | 0.958990 | 0.062003 | 0.039285 | 8064 | 75 | revision_experiments\window_imports\results_reviewer_window128_seeds0_1\checkpoints\window128_kan_seeds0_1\clipped_kan_gru_h32_no_current_seed1.pt |
| Bounded KAN-GRU h32 | 128 | 1 | 8 | 0.998921 | 0.045655 | 0.031940 | 8064 | 75 | revision_experiments\window_imports\results_reviewer_window128_seeds0_1\checkpoints\window128_kan_seeds0_1\clipped_kan_gru_h32_no_current_seed1.pt |
| Bounded KAN-GRU h32 | 128 | 2 | 43 | 0.991709 | 0.082637 | 0.040791 | 8064 | 99 | revision_experiments\window_imports\results_reviewer_window128_seeds2\checkpoints\window128_kan_seeds2\clipped_kan_gru_h32_no_current_seed2.pt |
| Bounded KAN-GRU h32 | 128 | 2 | 14 | 0.926910 | 0.071906 | 0.033119 | 8064 | 99 | revision_experiments\window_imports\results_reviewer_window128_seeds2\checkpoints\window128_kan_seeds2\clipped_kan_gru_h32_no_current_seed2.pt |
| Bounded KAN-GRU h32 | 128 | 2 | 54 | 0.998999 | 0.032287 | 0.022138 | 8064 | 99 | revision_experiments\window_imports\results_reviewer_window128_seeds2\checkpoints\window128_kan_seeds2\clipped_kan_gru_h32_no_current_seed2.pt |
| Bounded KAN-GRU h32 | 128 | 2 | 2 | 0.984870 | 0.059320 | 0.038210 | 8064 | 99 | revision_experiments\window_imports\results_reviewer_window128_seeds2\checkpoints\window128_kan_seeds2\clipped_kan_gru_h32_no_current_seed2.pt |
| Bounded KAN-GRU h32 | 128 | 2 | 36 | 0.993111 | 0.073176 | 0.034329 | 8064 | 99 | revision_experiments\window_imports\results_reviewer_window128_seeds2\checkpoints\window128_kan_seeds2\clipped_kan_gru_h32_no_current_seed2.pt |
| Bounded KAN-GRU h32 | 128 | 2 | 33 | 0.999276 | 0.035264 | 0.026647 | 8064 | 99 | revision_experiments\window_imports\results_reviewer_window128_seeds2\checkpoints\window128_kan_seeds2\clipped_kan_gru_h32_no_current_seed2.pt |
| Bounded KAN-GRU h32 | 128 | 2 | 1 | 0.988128 | 0.052266 | 0.036416 | 8064 | 99 | revision_experiments\window_imports\results_reviewer_window128_seeds2\checkpoints\window128_kan_seeds2\clipped_kan_gru_h32_no_current_seed2.pt |
| Bounded KAN-GRU h32 | 128 | 2 | 13 | 0.976952 | 0.046482 | 0.028916 | 8064 | 99 | revision_experiments\window_imports\results_reviewer_window128_seeds2\checkpoints\window128_kan_seeds2\clipped_kan_gru_h32_no_current_seed2.pt |
| Bounded KAN-GRU h32 | 128 | 2 | 8 | 0.998924 | 0.045593 | 0.034570 | 8064 | 99 | revision_experiments\window_imports\results_reviewer_window128_seeds2\checkpoints\window128_kan_seeds2\clipped_kan_gru_h32_no_current_seed2.pt |
| Bounded KAN-GRU h32 | 256 | 0 | 43 | 0.992945 | 0.075837 | 0.038741 | 8192 | 98 | results\kaggle_imported\kangru_sdc_100epoch_seed0\checkpoints\sequence\clipped_kan_gru_h32_no_current_seed0.pt |
| Bounded KAN-GRU h32 | 256 | 0 | 14 | 0.933185 | 0.068597 | 0.035592 | 8192 | 98 | results\kaggle_imported\kangru_sdc_100epoch_seed0\checkpoints\sequence\clipped_kan_gru_h32_no_current_seed0.pt |
| Bounded KAN-GRU h32 | 256 | 0 | 54 | 0.998131 | 0.043841 | 0.028659 | 8192 | 98 | results\kaggle_imported\kangru_sdc_100epoch_seed0\checkpoints\sequence\clipped_kan_gru_h32_no_current_seed0.pt |
| Bounded KAN-GRU h32 | 256 | 0 | 2 | 0.982254 | 0.063945 | 0.043984 | 8192 | 98 | results\kaggle_imported\kangru_sdc_100epoch_seed0\checkpoints\sequence\clipped_kan_gru_h32_no_current_seed0.pt |
| Bounded KAN-GRU h32 | 256 | 0 | 36 | 0.993510 | 0.070692 | 0.042664 | 8192 | 98 | results\kaggle_imported\kangru_sdc_100epoch_seed0\checkpoints\sequence\clipped_kan_gru_h32_no_current_seed0.pt |
| Bounded KAN-GRU h32 | 256 | 0 | 33 | 0.998173 | 0.055745 | 0.029586 | 8192 | 98 | results\kaggle_imported\kangru_sdc_100epoch_seed0\checkpoints\sequence\clipped_kan_gru_h32_no_current_seed0.pt |
| Bounded KAN-GRU h32 | 256 | 0 | 1 | 0.985564 | 0.057328 | 0.040048 | 8192 | 98 | results\kaggle_imported\kangru_sdc_100epoch_seed0\checkpoints\sequence\clipped_kan_gru_h32_no_current_seed0.pt |
| Bounded KAN-GRU h32 | 256 | 0 | 13 | 0.956618 | 0.063526 | 0.039529 | 8192 | 98 | results\kaggle_imported\kangru_sdc_100epoch_seed0\checkpoints\sequence\clipped_kan_gru_h32_no_current_seed0.pt |
| Bounded KAN-GRU h32 | 256 | 0 | 8 | 0.997628 | 0.067361 | 0.038616 | 8192 | 98 | results\kaggle_imported\kangru_sdc_100epoch_seed0\checkpoints\sequence\clipped_kan_gru_h32_no_current_seed0.pt |
| Bounded KAN-GRU h32 | 256 | 1 | 43 | 0.993879 | 0.070640 | 0.037243 | 8192 | 100 | results\kaggle_imported\clipped_kangru_sdc_100epoch_seed1_2\checkpoints\sequence\clipped_kan_gru_h32_no_current_seed1.pt |
| Bounded KAN-GRU h32 | 256 | 1 | 14 | 0.941878 | 0.063980 | 0.035731 | 8192 | 100 | results\kaggle_imported\clipped_kangru_sdc_100epoch_seed1_2\checkpoints\sequence\clipped_kan_gru_h32_no_current_seed1.pt |
| Bounded KAN-GRU h32 | 256 | 1 | 54 | 0.998043 | 0.044863 | 0.029893 | 8192 | 100 | results\kaggle_imported\clipped_kangru_sdc_100epoch_seed1_2\checkpoints\sequence\clipped_kan_gru_h32_no_current_seed1.pt |
| Bounded KAN-GRU h32 | 256 | 1 | 2 | 0.983648 | 0.061382 | 0.042010 | 8192 | 100 | results\kaggle_imported\clipped_kangru_sdc_100epoch_seed1_2\checkpoints\sequence\clipped_kan_gru_h32_no_current_seed1.pt |
| Bounded KAN-GRU h32 | 256 | 1 | 36 | 0.994861 | 0.062904 | 0.036234 | 8192 | 100 | results\kaggle_imported\clipped_kangru_sdc_100epoch_seed1_2\checkpoints\sequence\clipped_kan_gru_h32_no_current_seed1.pt |
| Bounded KAN-GRU h32 | 256 | 1 | 33 | 0.997967 | 0.058802 | 0.033384 | 8192 | 100 | results\kaggle_imported\clipped_kangru_sdc_100epoch_seed1_2\checkpoints\sequence\clipped_kan_gru_h32_no_current_seed1.pt |
| Bounded KAN-GRU h32 | 256 | 1 | 1 | 0.983849 | 0.060637 | 0.041509 | 8192 | 100 | results\kaggle_imported\clipped_kangru_sdc_100epoch_seed1_2\checkpoints\sequence\clipped_kan_gru_h32_no_current_seed1.pt |
| Bounded KAN-GRU h32 | 256 | 1 | 13 | 0.917730 | 0.087482 | 0.052219 | 8192 | 100 | results\kaggle_imported\clipped_kangru_sdc_100epoch_seed1_2\checkpoints\sequence\clipped_kan_gru_h32_no_current_seed1.pt |
| Bounded KAN-GRU h32 | 256 | 1 | 8 | 0.997534 | 0.068690 | 0.038586 | 8192 | 100 | results\kaggle_imported\clipped_kangru_sdc_100epoch_seed1_2\checkpoints\sequence\clipped_kan_gru_h32_no_current_seed1.pt |
| Bounded KAN-GRU h32 | 256 | 2 | 43 | 0.990877 | 0.086242 | 0.044978 | 8192 | 99 | results\kaggle_imported\clipped_kangru_sdc_100epoch_seed1_2\checkpoints\sequence\clipped_kan_gru_h32_no_current_seed2.pt |
| Bounded KAN-GRU h32 | 256 | 2 | 14 | 0.949297 | 0.059757 | 0.030498 | 8192 | 99 | results\kaggle_imported\clipped_kangru_sdc_100epoch_seed1_2\checkpoints\sequence\clipped_kan_gru_h32_no_current_seed2.pt |
| Bounded KAN-GRU h32 | 256 | 2 | 54 | 0.998287 | 0.041970 | 0.030199 | 8192 | 99 | results\kaggle_imported\clipped_kangru_sdc_100epoch_seed1_2\checkpoints\sequence\clipped_kan_gru_h32_no_current_seed2.pt |
| Bounded KAN-GRU h32 | 256 | 2 | 2 | 0.988608 | 0.051233 | 0.034106 | 8192 | 99 | results\kaggle_imported\clipped_kangru_sdc_100epoch_seed1_2\checkpoints\sequence\clipped_kan_gru_h32_no_current_seed2.pt |
| Bounded KAN-GRU h32 | 256 | 2 | 36 | 0.993308 | 0.071786 | 0.035826 | 8192 | 99 | results\kaggle_imported\clipped_kangru_sdc_100epoch_seed1_2\checkpoints\sequence\clipped_kan_gru_h32_no_current_seed2.pt |
| Bounded KAN-GRU h32 | 256 | 2 | 33 | 0.996819 | 0.073544 | 0.054485 | 8192 | 99 | results\kaggle_imported\clipped_kangru_sdc_100epoch_seed1_2\checkpoints\sequence\clipped_kan_gru_h32_no_current_seed2.pt |
| Bounded KAN-GRU h32 | 256 | 2 | 1 | 0.989257 | 0.049455 | 0.032831 | 8192 | 99 | results\kaggle_imported\clipped_kangru_sdc_100epoch_seed1_2\checkpoints\sequence\clipped_kan_gru_h32_no_current_seed2.pt |
| Bounded KAN-GRU h32 | 256 | 2 | 13 | 0.974954 | 0.048269 | 0.027528 | 8192 | 99 | results\kaggle_imported\clipped_kangru_sdc_100epoch_seed1_2\checkpoints\sequence\clipped_kan_gru_h32_no_current_seed2.pt |
| Bounded KAN-GRU h32 | 256 | 2 | 8 | 0.995602 | 0.091725 | 0.065576 | 8192 | 99 | results\kaggle_imported\clipped_kangru_sdc_100epoch_seed1_2\checkpoints\sequence\clipped_kan_gru_h32_no_current_seed2.pt |
| Bounded KAN-GRU h32 | 512 | 0 | 43 | 0.990660 | 0.087313 | 0.047476 | 8704 | 92 | revision_experiments\window_imports\results_reviewer_window512_seeds0_1\checkpoints\window512_kan_seeds0_1\clipped_kan_gru_h32_no_current_seed0.pt |
| Bounded KAN-GRU h32 | 512 | 0 | 14 | 0.898397 | 0.084663 | 0.045657 | 8704 | 92 | revision_experiments\window_imports\results_reviewer_window512_seeds0_1\checkpoints\window512_kan_seeds0_1\clipped_kan_gru_h32_no_current_seed0.pt |
| Bounded KAN-GRU h32 | 512 | 0 | 54 | 0.996543 | 0.059577 | 0.033925 | 8704 | 92 | revision_experiments\window_imports\results_reviewer_window512_seeds0_1\checkpoints\window512_kan_seeds0_1\clipped_kan_gru_h32_no_current_seed0.pt |
| Bounded KAN-GRU h32 | 512 | 0 | 2 | 0.971809 | 0.080695 | 0.053474 | 8704 | 92 | revision_experiments\window_imports\results_reviewer_window512_seeds0_1\checkpoints\window512_kan_seeds0_1\clipped_kan_gru_h32_no_current_seed0.pt |
| Bounded KAN-GRU h32 | 512 | 0 | 36 | 0.992012 | 0.078437 | 0.047698 | 8704 | 92 | revision_experiments\window_imports\results_reviewer_window512_seeds0_1\checkpoints\window512_kan_seeds0_1\clipped_kan_gru_h32_no_current_seed0.pt |
| Bounded KAN-GRU h32 | 512 | 0 | 33 | 0.996992 | 0.071571 | 0.035429 | 8704 | 92 | revision_experiments\window_imports\results_reviewer_window512_seeds0_1\checkpoints\window512_kan_seeds0_1\clipped_kan_gru_h32_no_current_seed0.pt |
| Bounded KAN-GRU h32 | 512 | 0 | 1 | 0.978778 | 0.069563 | 0.050364 | 8704 | 92 | revision_experiments\window_imports\results_reviewer_window512_seeds0_1\checkpoints\window512_kan_seeds0_1\clipped_kan_gru_h32_no_current_seed0.pt |
| Bounded KAN-GRU h32 | 512 | 0 | 13 | 0.688976 | 0.169754 | 0.084524 | 8704 | 92 | revision_experiments\window_imports\results_reviewer_window512_seeds0_1\checkpoints\window512_kan_seeds0_1\clipped_kan_gru_h32_no_current_seed0.pt |
| Bounded KAN-GRU h32 | 512 | 0 | 8 | 0.996251 | 0.084767 | 0.046853 | 8704 | 92 | revision_experiments\window_imports\results_reviewer_window512_seeds0_1\checkpoints\window512_kan_seeds0_1\clipped_kan_gru_h32_no_current_seed0.pt |
| Bounded KAN-GRU h32 | 512 | 1 | 43 | 0.992110 | 0.080252 | 0.044319 | 8704 | 100 | revision_experiments\window_imports\results_reviewer_window512_seeds0_1\checkpoints\window512_kan_seeds0_1\clipped_kan_gru_h32_no_current_seed1.pt |
| Bounded KAN-GRU h32 | 512 | 1 | 14 | 0.919246 | 0.075478 | 0.040582 | 8704 | 100 | revision_experiments\window_imports\results_reviewer_window512_seeds0_1\checkpoints\window512_kan_seeds0_1\clipped_kan_gru_h32_no_current_seed1.pt |
| Bounded KAN-GRU h32 | 512 | 1 | 54 | 0.995932 | 0.064628 | 0.032070 | 8704 | 100 | revision_experiments\window_imports\results_reviewer_window512_seeds0_1\checkpoints\window512_kan_seeds0_1\clipped_kan_gru_h32_no_current_seed1.pt |
| Bounded KAN-GRU h32 | 512 | 1 | 2 | 0.977378 | 0.072287 | 0.051264 | 8704 | 100 | revision_experiments\window_imports\results_reviewer_window512_seeds0_1\checkpoints\window512_kan_seeds0_1\clipped_kan_gru_h32_no_current_seed1.pt |
| Bounded KAN-GRU h32 | 512 | 1 | 36 | 0.992555 | 0.075726 | 0.045807 | 8704 | 100 | revision_experiments\window_imports\results_reviewer_window512_seeds0_1\checkpoints\window512_kan_seeds0_1\clipped_kan_gru_h32_no_current_seed1.pt |
| Bounded KAN-GRU h32 | 512 | 1 | 33 | 0.997127 | 0.069951 | 0.030479 | 8704 | 100 | revision_experiments\window_imports\results_reviewer_window512_seeds0_1\checkpoints\window512_kan_seeds0_1\clipped_kan_gru_h32_no_current_seed1.pt |
| Bounded KAN-GRU h32 | 512 | 1 | 1 | 0.980167 | 0.067247 | 0.048523 | 8704 | 100 | revision_experiments\window_imports\results_reviewer_window512_seeds0_1\checkpoints\window512_kan_seeds0_1\clipped_kan_gru_h32_no_current_seed1.pt |
| Bounded KAN-GRU h32 | 512 | 1 | 13 | 0.955843 | 0.063962 | 0.038700 | 8704 | 100 | revision_experiments\window_imports\results_reviewer_window512_seeds0_1\checkpoints\window512_kan_seeds0_1\clipped_kan_gru_h32_no_current_seed1.pt |
| Bounded KAN-GRU h32 | 512 | 1 | 8 | 0.996590 | 0.080841 | 0.039166 | 8704 | 100 | revision_experiments\window_imports\results_reviewer_window512_seeds0_1\checkpoints\window512_kan_seeds0_1\clipped_kan_gru_h32_no_current_seed1.pt |
| Bounded KAN-GRU h32 | 512 | 2 | 43 | 0.977280 | 0.136183 | 0.086546 | 8704 | 99 | revision_experiments\window_imports\results_reviewer_window512_seeds2\checkpoints\window512_kan_seeds2\clipped_kan_gru_h32_no_current_seed2.pt |
| Bounded KAN-GRU h32 | 512 | 2 | 14 | 0.899868 | 0.084048 | 0.050861 | 8704 | 99 | revision_experiments\window_imports\results_reviewer_window512_seeds2\checkpoints\window512_kan_seeds2\clipped_kan_gru_h32_no_current_seed2.pt |
| Bounded KAN-GRU h32 | 512 | 2 | 54 | 0.991433 | 0.093792 | 0.067041 | 8704 | 99 | revision_experiments\window_imports\results_reviewer_window512_seeds2\checkpoints\window512_kan_seeds2\clipped_kan_gru_h32_no_current_seed2.pt |
| Bounded KAN-GRU h32 | 512 | 2 | 2 | 0.941196 | 0.116546 | 0.089367 | 8704 | 99 | revision_experiments\window_imports\results_reviewer_window512_seeds2\checkpoints\window512_kan_seeds2\clipped_kan_gru_h32_no_current_seed2.pt |
| Bounded KAN-GRU h32 | 512 | 2 | 36 | 0.985628 | 0.105212 | 0.077841 | 8704 | 99 | revision_experiments\window_imports\results_reviewer_window512_seeds2\checkpoints\window512_kan_seeds2\clipped_kan_gru_h32_no_current_seed2.pt |
| Bounded KAN-GRU h32 | 512 | 2 | 33 | 0.991874 | 0.117633 | 0.090363 | 8704 | 99 | revision_experiments\window_imports\results_reviewer_window512_seeds2\checkpoints\window512_kan_seeds2\clipped_kan_gru_h32_no_current_seed2.pt |
| Bounded KAN-GRU h32 | 512 | 2 | 1 | 0.949584 | 0.107217 | 0.084627 | 8704 | 99 | revision_experiments\window_imports\results_reviewer_window512_seeds2\checkpoints\window512_kan_seeds2\clipped_kan_gru_h32_no_current_seed2.pt |
| Bounded KAN-GRU h32 | 512 | 2 | 13 | 0.947405 | 0.069806 | 0.048689 | 8704 | 99 | revision_experiments\window_imports\results_reviewer_window512_seeds2\checkpoints\window512_kan_seeds2\clipped_kan_gru_h32_no_current_seed2.pt |
| Bounded KAN-GRU h32 | 512 | 2 | 8 | 0.989014 | 0.145097 | 0.113799 | 8704 | 99 | revision_experiments\window_imports\results_reviewer_window512_seeds2\checkpoints\window512_kan_seeds2\clipped_kan_gru_h32_no_current_seed2.pt |
| Dense GRU h116 | 128 | 0 | 43 | 0.987690 | 0.100693 | 0.062341 | 8064 | 89 | revision_experiments\window_imports\results_reviewer_window128_seeds0_1\checkpoints\window128_gru_seeds0_1\gru_h116_no_current_seed0.pt |
| Dense GRU h116 | 128 | 0 | 14 | 0.889788 | 0.088298 | 0.049937 | 8064 | 89 | revision_experiments\window_imports\results_reviewer_window128_seeds0_1\checkpoints\window128_gru_seeds0_1\gru_h116_no_current_seed0.pt |
| Dense GRU h116 | 128 | 0 | 54 | 0.997733 | 0.048598 | 0.035436 | 8064 | 89 | revision_experiments\window_imports\results_reviewer_window128_seeds0_1\checkpoints\window128_gru_seeds0_1\gru_h116_no_current_seed0.pt |
| Dense GRU h116 | 128 | 0 | 2 | 0.982232 | 0.064285 | 0.043346 | 8064 | 89 | revision_experiments\window_imports\results_reviewer_window128_seeds0_1\checkpoints\window128_gru_seeds0_1\gru_h116_no_current_seed0.pt |
| Dense GRU h116 | 128 | 0 | 36 | 0.994008 | 0.068246 | 0.044826 | 8064 | 89 | revision_experiments\window_imports\results_reviewer_window128_seeds0_1\checkpoints\window128_gru_seeds0_1\gru_h116_no_current_seed0.pt |
| Dense GRU h116 | 128 | 0 | 33 | 0.994083 | 0.100790 | 0.048076 | 8064 | 89 | revision_experiments\window_imports\results_reviewer_window128_seeds0_1\checkpoints\window128_gru_seeds0_1\gru_h116_no_current_seed0.pt |
| Dense GRU h116 | 128 | 0 | 1 | 0.979963 | 0.067902 | 0.047255 | 8064 | 89 | revision_experiments\window_imports\results_reviewer_window128_seeds0_1\checkpoints\window128_gru_seeds0_1\gru_h116_no_current_seed0.pt |
| Dense GRU h116 | 128 | 0 | 13 | 0.928663 | 0.081776 | 0.046449 | 8064 | 89 | revision_experiments\window_imports\results_reviewer_window128_seeds0_1\checkpoints\window128_gru_seeds0_1\gru_h116_no_current_seed0.pt |
| Dense GRU h116 | 128 | 0 | 8 | 0.993148 | 0.115051 | 0.056678 | 8064 | 89 | revision_experiments\window_imports\results_reviewer_window128_seeds0_1\checkpoints\window128_gru_seeds0_1\gru_h116_no_current_seed0.pt |
| Dense GRU h116 | 128 | 1 | 43 | 0.987321 | 0.102189 | 0.057955 | 8064 | 92 | revision_experiments\window_imports\results_reviewer_window128_seeds0_1\checkpoints\window128_gru_seeds0_1\gru_h116_no_current_seed1.pt |
| Dense GRU h116 | 128 | 1 | 14 | 0.904513 | 0.082189 | 0.043746 | 8064 | 92 | revision_experiments\window_imports\results_reviewer_window128_seeds0_1\checkpoints\window128_gru_seeds0_1\gru_h116_no_current_seed1.pt |
| Dense GRU h116 | 128 | 1 | 54 | 0.997978 | 0.045901 | 0.032764 | 8064 | 92 | revision_experiments\window_imports\results_reviewer_window128_seeds0_1\checkpoints\window128_gru_seeds0_1\gru_h116_no_current_seed1.pt |
| Dense GRU h116 | 128 | 1 | 2 | 0.967849 | 0.086473 | 0.053177 | 8064 | 92 | revision_experiments\window_imports\results_reviewer_window128_seeds0_1\checkpoints\window128_gru_seeds0_1\gru_h116_no_current_seed1.pt |
| Dense GRU h116 | 128 | 1 | 36 | 0.993152 | 0.072960 | 0.040890 | 8064 | 92 | revision_experiments\window_imports\results_reviewer_window128_seeds0_1\checkpoints\window128_gru_seeds0_1\gru_h116_no_current_seed1.pt |
| Dense GRU h116 | 128 | 1 | 33 | 0.994567 | 0.096586 | 0.042087 | 8064 | 92 | revision_experiments\window_imports\results_reviewer_window128_seeds0_1\checkpoints\window128_gru_seeds0_1\gru_h116_no_current_seed1.pt |
| Dense GRU h116 | 128 | 1 | 1 | 0.971784 | 0.080577 | 0.051640 | 8064 | 92 | revision_experiments\window_imports\results_reviewer_window128_seeds0_1\checkpoints\window128_gru_seeds0_1\gru_h116_no_current_seed1.pt |
| Dense GRU h116 | 128 | 1 | 13 | 0.925371 | 0.083641 | 0.043713 | 8064 | 92 | revision_experiments\window_imports\results_reviewer_window128_seeds0_1\checkpoints\window128_gru_seeds0_1\gru_h116_no_current_seed1.pt |
| Dense GRU h116 | 128 | 1 | 8 | 0.993384 | 0.113049 | 0.053997 | 8064 | 92 | revision_experiments\window_imports\results_reviewer_window128_seeds0_1\checkpoints\window128_gru_seeds0_1\gru_h116_no_current_seed1.pt |
| Dense GRU h116 | 128 | 2 | 43 | 0.991684 | 0.082761 | 0.046133 | 8064 | 71 | revision_experiments\window_imports\results_reviewer_window128_seeds2\checkpoints\window128_gru_seeds2\gru_h116_no_current_seed2.pt |
| Dense GRU h116 | 128 | 2 | 14 | 0.899592 | 0.084280 | 0.039574 | 8064 | 71 | revision_experiments\window_imports\results_reviewer_window128_seeds2\checkpoints\window128_gru_seeds2\gru_h116_no_current_seed2.pt |
| Dense GRU h116 | 128 | 2 | 54 | 0.997980 | 0.045875 | 0.030738 | 8064 | 71 | revision_experiments\window_imports\results_reviewer_window128_seeds2\checkpoints\window128_gru_seeds2\gru_h116_no_current_seed2.pt |
| Dense GRU h116 | 128 | 2 | 2 | 0.969804 | 0.083803 | 0.054923 | 8064 | 71 | revision_experiments\window_imports\results_reviewer_window128_seeds2\checkpoints\window128_gru_seeds2\gru_h116_no_current_seed2.pt |
| Dense GRU h116 | 128 | 2 | 36 | 0.993748 | 0.069710 | 0.049326 | 8064 | 71 | revision_experiments\window_imports\results_reviewer_window128_seeds2\checkpoints\window128_gru_seeds2\gru_h116_no_current_seed2.pt |
| Dense GRU h116 | 128 | 2 | 33 | 0.994846 | 0.094065 | 0.033506 | 8064 | 71 | revision_experiments\window_imports\results_reviewer_window128_seeds2\checkpoints\window128_gru_seeds2\gru_h116_no_current_seed2.pt |
| Dense GRU h116 | 128 | 2 | 1 | 0.984292 | 0.060121 | 0.039752 | 8064 | 71 | revision_experiments\window_imports\results_reviewer_window128_seeds2\checkpoints\window128_gru_seeds2\gru_h116_no_current_seed2.pt |
| Dense GRU h116 | 128 | 2 | 13 | 0.922879 | 0.085027 | 0.042558 | 8064 | 71 | revision_experiments\window_imports\results_reviewer_window128_seeds2\checkpoints\window128_gru_seeds2\gru_h116_no_current_seed2.pt |
| Dense GRU h116 | 128 | 2 | 8 | 0.993817 | 0.109291 | 0.043606 | 8064 | 71 | revision_experiments\window_imports\results_reviewer_window128_seeds2\checkpoints\window128_gru_seeds2\gru_h116_no_current_seed2.pt |
| Dense GRU h116 | 256 | 0 | 43 | 0.984284 | 0.113192 | 0.061347 | 8192 | 99 | checkpoints\kaggle_imported\no_current_baselines_100epoch_3seed\sequence\gru_h116_no_current_seed0.pt |
| Dense GRU h116 | 256 | 0 | 14 | 0.880582 | 0.091708 | 0.054487 | 8192 | 99 | checkpoints\kaggle_imported\no_current_baselines_100epoch_3seed\sequence\gru_h116_no_current_seed0.pt |
| Dense GRU h116 | 256 | 0 | 54 | 0.996867 | 0.056767 | 0.039545 | 8192 | 99 | checkpoints\kaggle_imported\no_current_baselines_100epoch_3seed\sequence\gru_h116_no_current_seed0.pt |
| Dense GRU h116 | 256 | 0 | 2 | 0.938196 | 0.119333 | 0.086314 | 8192 | 99 | checkpoints\kaggle_imported\no_current_baselines_100epoch_3seed\sequence\gru_h116_no_current_seed0.pt |
| Dense GRU h116 | 256 | 0 | 36 | 0.990072 | 0.087434 | 0.060731 | 8192 | 99 | checkpoints\kaggle_imported\no_current_baselines_100epoch_3seed\sequence\gru_h116_no_current_seed0.pt |
| Dense GRU h116 | 256 | 0 | 33 | 0.993201 | 0.107527 | 0.047293 | 8192 | 99 | checkpoints\kaggle_imported\no_current_baselines_100epoch_3seed\sequence\gru_h116_no_current_seed0.pt |
| Dense GRU h116 | 256 | 0 | 1 | 0.953358 | 0.103046 | 0.076786 | 8192 | 99 | checkpoints\kaggle_imported\no_current_baselines_100epoch_3seed\sequence\gru_h116_no_current_seed0.pt |
| Dense GRU h116 | 256 | 0 | 13 | 0.907955 | 0.092534 | 0.055720 | 8192 | 99 | checkpoints\kaggle_imported\no_current_baselines_100epoch_3seed\sequence\gru_h116_no_current_seed0.pt |
| Dense GRU h116 | 256 | 0 | 8 | 0.991809 | 0.125183 | 0.057630 | 8192 | 99 | checkpoints\kaggle_imported\no_current_baselines_100epoch_3seed\sequence\gru_h116_no_current_seed0.pt |
| Dense GRU h116 | 256 | 1 | 43 | 0.987819 | 0.099649 | 0.051380 | 8192 | 100 | checkpoints\kaggle_imported\no_current_baselines_100epoch_3seed\sequence\gru_h116_no_current_seed1.pt |
| Dense GRU h116 | 256 | 1 | 14 | 0.888852 | 0.088475 | 0.047825 | 8192 | 100 | checkpoints\kaggle_imported\no_current_baselines_100epoch_3seed\sequence\gru_h116_no_current_seed1.pt |
| Dense GRU h116 | 256 | 1 | 54 | 0.996260 | 0.062024 | 0.040727 | 8192 | 100 | checkpoints\kaggle_imported\no_current_baselines_100epoch_3seed\sequence\gru_h116_no_current_seed1.pt |
| Dense GRU h116 | 256 | 1 | 2 | 0.938015 | 0.119507 | 0.077607 | 8192 | 100 | checkpoints\kaggle_imported\no_current_baselines_100epoch_3seed\sequence\gru_h116_no_current_seed1.pt |
| Dense GRU h116 | 256 | 1 | 36 | 0.992401 | 0.076494 | 0.046985 | 8192 | 100 | checkpoints\kaggle_imported\no_current_baselines_100epoch_3seed\sequence\gru_h116_no_current_seed1.pt |
| Dense GRU h116 | 256 | 1 | 33 | 0.993348 | 0.106355 | 0.043865 | 8192 | 100 | checkpoints\kaggle_imported\no_current_baselines_100epoch_3seed\sequence\gru_h116_no_current_seed1.pt |
| Dense GRU h116 | 256 | 1 | 1 | 0.958872 | 0.096764 | 0.071881 | 8192 | 100 | checkpoints\kaggle_imported\no_current_baselines_100epoch_3seed\sequence\gru_h116_no_current_seed1.pt |
| Dense GRU h116 | 256 | 1 | 13 | 0.907940 | 0.092541 | 0.053299 | 8192 | 100 | checkpoints\kaggle_imported\no_current_baselines_100epoch_3seed\sequence\gru_h116_no_current_seed1.pt |
| Dense GRU h116 | 256 | 1 | 8 | 0.992146 | 0.122580 | 0.051116 | 8192 | 100 | checkpoints\kaggle_imported\no_current_baselines_100epoch_3seed\sequence\gru_h116_no_current_seed1.pt |
| Dense GRU h116 | 256 | 2 | 43 | 0.947804 | 0.206279 | 0.129081 | 8192 | 99 | checkpoints\kaggle_imported\no_current_baselines_100epoch_3seed\sequence\gru_h116_no_current_seed2.pt |
| Dense GRU h116 | 256 | 2 | 14 | 0.893704 | 0.086523 | 0.049352 | 8192 | 99 | checkpoints\kaggle_imported\no_current_baselines_100epoch_3seed\sequence\gru_h116_no_current_seed2.pt |
| Dense GRU h116 | 256 | 2 | 54 | 0.995331 | 0.069299 | 0.051758 | 8192 | 99 | checkpoints\kaggle_imported\no_current_baselines_100epoch_3seed\sequence\gru_h116_no_current_seed2.pt |
| Dense GRU h116 | 256 | 2 | 2 | 0.960733 | 0.095118 | 0.066942 | 8192 | 99 | checkpoints\kaggle_imported\no_current_baselines_100epoch_3seed\sequence\gru_h116_no_current_seed2.pt |
| Dense GRU h116 | 256 | 2 | 36 | 0.919478 | 0.249003 | 0.150451 | 8192 | 99 | checkpoints\kaggle_imported\no_current_baselines_100epoch_3seed\sequence\gru_h116_no_current_seed2.pt |
| Dense GRU h116 | 256 | 2 | 33 | 0.993669 | 0.103762 | 0.046903 | 8192 | 99 | checkpoints\kaggle_imported\no_current_baselines_100epoch_3seed\sequence\gru_h116_no_current_seed2.pt |
| Dense GRU h116 | 256 | 2 | 1 | 0.963125 | 0.091624 | 0.067720 | 8192 | 99 | checkpoints\kaggle_imported\no_current_baselines_100epoch_3seed\sequence\gru_h116_no_current_seed2.pt |
| Dense GRU h116 | 256 | 2 | 13 | 0.899661 | 0.096613 | 0.058532 | 8192 | 99 | checkpoints\kaggle_imported\no_current_baselines_100epoch_3seed\sequence\gru_h116_no_current_seed2.pt |
| Dense GRU h116 | 256 | 2 | 8 | 0.991641 | 0.126464 | 0.065263 | 8192 | 99 | checkpoints\kaggle_imported\no_current_baselines_100epoch_3seed\sequence\gru_h116_no_current_seed2.pt |
| Dense GRU h116 | 512 | 0 | 43 | 0.799401 | 0.404648 | 0.333433 | 8704 | 86 | revision_experiments\window_imports\results_reviewer_window512_seeds0_1\checkpoints\window512_gru_seeds0_1\gru_h116_no_current_seed0.pt |
| Dense GRU h116 | 512 | 0 | 14 | 0.832762 | 0.108620 | 0.077988 | 8704 | 86 | revision_experiments\window_imports\results_reviewer_window512_seeds0_1\checkpoints\window512_gru_seeds0_1\gru_h116_no_current_seed0.pt |
| Dense GRU h116 | 512 | 0 | 54 | 0.996280 | 0.061803 | 0.040396 | 8704 | 86 | revision_experiments\window_imports\results_reviewer_window512_seeds0_1\checkpoints\window512_gru_seeds0_1\gru_h116_no_current_seed0.pt |
| Dense GRU h116 | 512 | 0 | 2 | 0.942261 | 0.115485 | 0.078825 | 8704 | 86 | revision_experiments\window_imports\results_reviewer_window512_seeds0_1\checkpoints\window512_gru_seeds0_1\gru_h116_no_current_seed0.pt |
| Dense GRU h116 | 512 | 0 | 36 | 0.820742 | 0.371577 | 0.286980 | 8704 | 86 | revision_experiments\window_imports\results_reviewer_window512_seeds0_1\checkpoints\window512_gru_seeds0_1\gru_h116_no_current_seed0.pt |
| Dense GRU h116 | 512 | 0 | 33 | 0.990328 | 0.128335 | 0.053943 | 8704 | 86 | revision_experiments\window_imports\results_reviewer_window512_seeds0_1\checkpoints\window512_gru_seeds0_1\gru_h116_no_current_seed0.pt |
| Dense GRU h116 | 512 | 0 | 1 | 0.940200 | 0.116769 | 0.080104 | 8704 | 86 | revision_experiments\window_imports\results_reviewer_window512_seeds0_1\checkpoints\window512_gru_seeds0_1\gru_h116_no_current_seed0.pt |
| Dense GRU h116 | 512 | 0 | 13 | 0.863461 | 0.112474 | 0.079561 | 8704 | 86 | revision_experiments\window_imports\results_reviewer_window512_seeds0_1\checkpoints\window512_gru_seeds0_1\gru_h116_no_current_seed0.pt |
| Dense GRU h116 | 512 | 0 | 8 | 0.988233 | 0.150171 | 0.076115 | 8704 | 86 | revision_experiments\window_imports\results_reviewer_window512_seeds0_1\checkpoints\window512_gru_seeds0_1\gru_h116_no_current_seed0.pt |
| Dense GRU h116 | 512 | 1 | 43 | 0.931004 | 0.237315 | 0.141380 | 8704 | 98 | revision_experiments\window_imports\results_reviewer_window512_seeds0_1\checkpoints\window512_gru_seeds0_1\gru_h116_no_current_seed1.pt |
| Dense GRU h116 | 512 | 1 | 14 | 0.813270 | 0.114775 | 0.069241 | 8704 | 98 | revision_experiments\window_imports\results_reviewer_window512_seeds0_1\checkpoints\window512_gru_seeds0_1\gru_h116_no_current_seed1.pt |
| Dense GRU h116 | 512 | 1 | 54 | 0.993320 | 0.082818 | 0.060774 | 8704 | 98 | revision_experiments\window_imports\results_reviewer_window512_seeds0_1\checkpoints\window512_gru_seeds0_1\gru_h116_no_current_seed1.pt |
| Dense GRU h116 | 512 | 1 | 2 | 0.915914 | 0.139365 | 0.104096 | 8704 | 98 | revision_experiments\window_imports\results_reviewer_window512_seeds0_1\checkpoints\window512_gru_seeds0_1\gru_h116_no_current_seed1.pt |
| Dense GRU h116 | 512 | 1 | 36 | 0.941507 | 0.212258 | 0.123399 | 8704 | 98 | revision_experiments\window_imports\results_reviewer_window512_seeds0_1\checkpoints\window512_gru_seeds0_1\gru_h116_no_current_seed1.pt |
| Dense GRU h116 | 512 | 1 | 33 | 0.989012 | 0.136789 | 0.066076 | 8704 | 98 | revision_experiments\window_imports\results_reviewer_window512_seeds0_1\checkpoints\window512_gru_seeds0_1\gru_h116_no_current_seed1.pt |
| Dense GRU h116 | 512 | 1 | 1 | 0.910380 | 0.142949 | 0.110460 | 8704 | 98 | revision_experiments\window_imports\results_reviewer_window512_seeds0_1\checkpoints\window512_gru_seeds0_1\gru_h116_no_current_seed1.pt |
| Dense GRU h116 | 512 | 1 | 13 | 0.861583 | 0.113245 | 0.075857 | 8704 | 98 | revision_experiments\window_imports\results_reviewer_window512_seeds0_1\checkpoints\window512_gru_seeds0_1\gru_h116_no_current_seed1.pt |
| Dense GRU h116 | 512 | 1 | 8 | 0.986173 | 0.162786 | 0.090633 | 8704 | 98 | revision_experiments\window_imports\results_reviewer_window512_seeds0_1\checkpoints\window512_gru_seeds0_1\gru_h116_no_current_seed1.pt |
| Dense GRU h116 | 512 | 2 | 43 | 0.897887 | 0.288704 | 0.224705 | 8704 | 99 | revision_experiments\window_imports\results_reviewer_window512_seeds2\checkpoints\window512_gru_seeds2\gru_h116_no_current_seed2.pt |
| Dense GRU h116 | 512 | 2 | 14 | 0.624833 | 0.162687 | 0.116457 | 8704 | 99 | revision_experiments\window_imports\results_reviewer_window512_seeds2\checkpoints\window512_gru_seeds2\gru_h116_no_current_seed2.pt |
| Dense GRU h116 | 512 | 2 | 54 | 0.995579 | 0.067375 | 0.040326 | 8704 | 99 | revision_experiments\window_imports\results_reviewer_window512_seeds2\checkpoints\window512_gru_seeds2\gru_h116_no_current_seed2.pt |
| Dense GRU h116 | 512 | 2 | 2 | 0.912223 | 0.142391 | 0.116308 | 8704 | 99 | revision_experiments\window_imports\results_reviewer_window512_seeds2\checkpoints\window512_gru_seeds2\gru_h116_no_current_seed2.pt |
| Dense GRU h116 | 512 | 2 | 36 | 0.938265 | 0.218060 | 0.154475 | 8704 | 99 | revision_experiments\window_imports\results_reviewer_window512_seeds2\checkpoints\window512_gru_seeds2\gru_h116_no_current_seed2.pt |
| Dense GRU h116 | 512 | 2 | 33 | 0.987424 | 0.146338 | 0.093077 | 8704 | 99 | revision_experiments\window_imports\results_reviewer_window512_seeds2\checkpoints\window512_gru_seeds2\gru_h116_no_current_seed2.pt |
| Dense GRU h116 | 512 | 2 | 1 | 0.906452 | 0.146048 | 0.119206 | 8704 | 99 | revision_experiments\window_imports\results_reviewer_window512_seeds2\checkpoints\window512_gru_seeds2\gru_h116_no_current_seed2.pt |
| Dense GRU h116 | 512 | 2 | 13 | 0.754823 | 0.150717 | 0.108745 | 8704 | 99 | revision_experiments\window_imports\results_reviewer_window512_seeds2\checkpoints\window512_gru_seeds2\gru_h116_no_current_seed2.pt |
| Dense GRU h116 | 512 | 2 | 8 | 0.989527 | 0.141668 | 0.074421 | 8704 | 99 | revision_experiments\window_imports\results_reviewer_window512_seeds2\checkpoints\window512_gru_seeds2\gru_h116_no_current_seed2.pt |

## 6. Paired KAN-GRU vs GRU analysis

| Window | Mean paired delta R2 | Median paired delta R2 | 95% CI low | 95% CI high | KAN wins | GRU wins | Ties | Paired trajectories |
|---|---|---|---|---|---|---|---|---|
| 128 | 0.011691 | 0.005483 | 0.003696 | 0.022312 | 8 | 1 | 0 | 9 |
| 256 | 0.024714 | 0.026576 | 0.013414 | 0.036671 | 9 | 0 | 0 | 9 |
| 512 | 0.054363 | 0.039994 | 0.024261 | 0.088971 | 8 | 1 | 0 | 9 |

Factual classification: **broadly stable but magnitude varies**.

## 7. Full-trajectory continuous inference - aggregate

| Model | Reset R2 | Reset R2 SD | Continuous R2 | Continuous R2 SD | Delta R2 | Reset RMSE | Reset RMSE SD | Continuous RMSE | Continuous RMSE SD | Delta RMSE |
|---|---|---|---|---|---|---|---|---|---|---|
| Bounded KAN-GRU h32 | 0.981846 | 0.022670 | 0.994185 | 0.006412 | 0.012339 | 0.064588 | 0.012799 | 0.038465 | 0.010266 | -0.026123 |
| Dense GRU h116 | 0.957850 | 0.039497 | 0.986491 | 0.014396 | 0.028641 | 0.106137 | 0.039499 | 0.053120 | 0.017359 | -0.053016 |

Reset mode covers each sample exactly once in contiguous 256-sample chunks. Continuous mode initializes the recurrent state once and carries it through all 15,999 samples.

## 8. Full-trajectory results by seed

| Model | Seed | Trajectories | Reset R2 | Continuous R2 | Delta R2 | Reset RMSE | Continuous RMSE | Delta RMSE |
|---|---|---|---|---|---|---|---|---|
| Bounded KAN-GRU h32 | 0 | 9 | 0.981514 | 0.994225 | 0.012711 | 0.063469 | 0.036296 | -0.027173 |
| Bounded KAN-GRU h32 | 1 | 9 | 0.978371 | 0.992768 | 0.014396 | 0.065221 | 0.038819 | -0.026401 |
| Bounded KAN-GRU h32 | 2 | 9 | 0.985651 | 0.995562 | 0.009911 | 0.065073 | 0.040279 | -0.024794 |
| Dense GRU h116 | 0 | 9 | 0.960202 | 0.981404 | 0.021202 | 0.097534 | 0.058720 | -0.038814 |
| Dense GRU h116 | 1 | 9 | 0.961659 | 0.990247 | 0.028588 | 0.095554 | 0.044516 | -0.051038 |
| Dense GRU h116 | 2 | 9 | 0.951690 | 0.987824 | 0.036134 | 0.125322 | 0.056125 | -0.069197 |

## 9. Full-trajectory results by trajectory

| Trajectory ID | Local index | Model | Seed | Reset R2 | Continuous R2 | Delta R2 | Reset RMSE | Continuous RMSE | Delta RMSE | Samples | Reset count | Continuous reset count |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 6 | Bounded KAN-GRU h32 | 0 | 0.985547 | 0.992116 | 0.006569 | 0.057513 | 0.042477 | -0.015036 | 15999 | 63 | 1 |
| 2 | 3 | Bounded KAN-GRU h32 | 0 | 0.981921 | 0.991204 | 0.009283 | 0.064702 | 0.045131 | -0.019571 | 15999 | 63 | 1 |
| 8 | 8 | Bounded KAN-GRU h32 | 0 | 0.997683 | 0.999152 | 0.001469 | 0.066702 | 0.040358 | -0.026344 | 15999 | 63 | 1 |
| 13 | 7 | Bounded KAN-GRU h32 | 0 | 0.955822 | 0.983665 | 0.027843 | 0.064139 | 0.039002 | -0.025138 | 15999 | 63 | 1 |
| 14 | 1 | Bounded KAN-GRU h32 | 0 | 0.929999 | 0.986128 | 0.056129 | 0.070253 | 0.031274 | -0.038979 | 15999 | 63 | 1 |
| 33 | 5 | Bounded KAN-GRU h32 | 0 | 0.998142 | 0.999361 | 0.001220 | 0.056304 | 0.033008 | -0.023296 | 15999 | 63 | 1 |
| 36 | 4 | Bounded KAN-GRU h32 | 0 | 0.993239 | 0.998321 | 0.005082 | 0.072268 | 0.036013 | -0.036255 | 15999 | 63 | 1 |
| 43 | 0 | Bounded KAN-GRU h32 | 0 | 0.993234 | 0.999004 | 0.005770 | 0.074393 | 0.028540 | -0.045852 | 15999 | 63 | 1 |
| 54 | 2 | Bounded KAN-GRU h32 | 0 | 0.998042 | 0.999077 | 0.001035 | 0.044946 | 0.030859 | -0.014087 | 15999 | 63 | 1 |
| 1 | 6 | Bounded KAN-GRU h32 | 1 | 0.984427 | 0.992319 | 0.007892 | 0.059700 | 0.041927 | -0.017773 | 15999 | 63 | 1 |
| 2 | 3 | Bounded KAN-GRU h32 | 1 | 0.983847 | 0.992711 | 0.008863 | 0.061158 | 0.041084 | -0.020074 | 15999 | 63 | 1 |
| 8 | 8 | Bounded KAN-GRU h32 | 1 | 0.997623 | 0.999183 | 0.001560 | 0.067557 | 0.039598 | -0.027959 | 15999 | 63 | 1 |
| 13 | 7 | Bounded KAN-GRU h32 | 1 | 0.916850 | 0.974679 | 0.057828 | 0.087994 | 0.048558 | -0.039436 | 15999 | 63 | 1 |
| 14 | 1 | Bounded KAN-GRU h32 | 1 | 0.938856 | 0.981020 | 0.042164 | 0.065658 | 0.036581 | -0.029077 | 15999 | 63 | 1 |
| 33 | 5 | Bounded KAN-GRU h32 | 1 | 0.997889 | 0.999112 | 0.001223 | 0.060013 | 0.038926 | -0.021087 | 15999 | 63 | 1 |
| 36 | 4 | Bounded KAN-GRU h32 | 1 | 0.994179 | 0.998433 | 0.004253 | 0.067053 | 0.034795 | -0.032258 | 15999 | 63 | 1 |
| 43 | 0 | Bounded KAN-GRU h32 | 1 | 0.993755 | 0.998497 | 0.004741 | 0.071468 | 0.035066 | -0.036402 | 15999 | 63 | 1 |
| 54 | 2 | Bounded KAN-GRU h32 | 1 | 0.997915 | 0.998955 | 0.001040 | 0.046384 | 0.032836 | -0.013547 | 15999 | 63 | 1 |
| 1 | 6 | Bounded KAN-GRU h32 | 2 | 0.988893 | 0.995210 | 0.006316 | 0.050418 | 0.033111 | -0.017307 | 15999 | 63 | 1 |
| 2 | 3 | Bounded KAN-GRU h32 | 2 | 0.988816 | 0.995305 | 0.006489 | 0.050889 | 0.032971 | -0.017917 | 15999 | 63 | 1 |
| 8 | 8 | Bounded KAN-GRU h32 | 2 | 0.995807 | 0.996837 | 0.001029 | 0.089720 | 0.077935 | -0.011785 | 15999 | 63 | 1 |
| 13 | 7 | Bounded KAN-GRU h32 | 2 | 0.974413 | 0.991070 | 0.016657 | 0.048813 | 0.028837 | -0.019976 | 15999 | 63 | 1 |
| 14 | 1 | Bounded KAN-GRU h32 | 2 | 0.945764 | 0.987621 | 0.041856 | 0.061838 | 0.029543 | -0.032295 | 15999 | 63 | 1 |
| 33 | 5 | Bounded KAN-GRU h32 | 2 | 0.996906 | 0.997866 | 0.000960 | 0.072656 | 0.060338 | -0.012318 | 15999 | 63 | 1 |
| 36 | 4 | Bounded KAN-GRU h32 | 2 | 0.992288 | 0.998471 | 0.006184 | 0.077181 | 0.034361 | -0.042821 | 15999 | 63 | 1 |
| 43 | 0 | Bounded KAN-GRU h32 | 2 | 0.989716 | 0.998804 | 0.009089 | 0.091716 | 0.031272 | -0.060444 | 15999 | 63 | 1 |
| 54 | 2 | Bounded KAN-GRU h32 | 2 | 0.998255 | 0.998870 | 0.000615 | 0.042427 | 0.034142 | -0.008285 | 15999 | 63 | 1 |
| 1 | 6 | Dense GRU h116 | 0 | 0.951983 | 0.957835 | 0.005852 | 0.104831 | 0.098236 | -0.006595 | 15999 | 63 | 1 |
| 2 | 3 | Dense GRU h116 | 0 | 0.942101 | 0.954911 | 0.012810 | 0.115789 | 0.102180 | -0.013609 | 15999 | 63 | 1 |
| 8 | 8 | Dense GRU h116 | 0 | 0.991892 | 0.998552 | 0.006659 | 0.124769 | 0.052735 | -0.072034 | 15999 | 63 | 1 |
| 13 | 7 | Dense GRU h116 | 0 | 0.905219 | 0.971036 | 0.065816 | 0.093947 | 0.051934 | -0.042013 | 15999 | 63 | 1 |
| 14 | 1 | Dense GRU h116 | 0 | 0.881272 | 0.958056 | 0.076784 | 0.091493 | 0.054381 | -0.037112 | 15999 | 63 | 1 |
| 33 | 5 | Dense GRU h116 | 0 | 0.993032 | 0.998895 | 0.005863 | 0.109035 | 0.043421 | -0.065615 | 15999 | 63 | 1 |
| 36 | 4 | Dense GRU h116 | 0 | 0.989371 | 0.996088 | 0.006717 | 0.090608 | 0.054971 | -0.035637 | 15999 | 63 | 1 |
| 43 | 0 | Dense GRU h116 | 0 | 0.990167 | 0.998454 | 0.008286 | 0.089678 | 0.035564 | -0.054115 | 15999 | 63 | 1 |
| 54 | 2 | Dense GRU h116 | 0 | 0.996778 | 0.998809 | 0.002031 | 0.057657 | 0.035061 | -0.022596 | 15999 | 63 | 1 |
| 1 | 6 | Dense GRU h116 | 1 | 0.958112 | 0.981396 | 0.023284 | 0.097913 | 0.065253 | -0.032660 | 15999 | 63 | 1 |
| 2 | 3 | Dense GRU h116 | 1 | 0.936237 | 0.985795 | 0.049558 | 0.121510 | 0.057352 | -0.064159 | 15999 | 63 | 1 |
| 8 | 8 | Dense GRU h116 | 1 | 0.992103 | 0.999114 | 0.007011 | 0.123133 | 0.041242 | -0.081891 | 15999 | 63 | 1 |
| 13 | 7 | Dense GRU h116 | 1 | 0.908091 | 0.975872 | 0.067782 | 0.092513 | 0.047400 | -0.045113 | 15999 | 63 | 1 |
| 14 | 1 | Dense GRU h116 | 1 | 0.888124 | 0.975911 | 0.087787 | 0.088814 | 0.041212 | -0.047602 | 15999 | 63 | 1 |
| 33 | 5 | Dense GRU h116 | 1 | 0.993142 | 0.999124 | 0.005982 | 0.108167 | 0.038657 | -0.069510 | 15999 | 63 | 1 |
| 36 | 4 | Dense GRU h116 | 1 | 0.992047 | 0.997114 | 0.005068 | 0.078379 | 0.047210 | -0.031169 | 15999 | 63 | 1 |
| 43 | 0 | Dense GRU h116 | 1 | 0.991034 | 0.998938 | 0.007904 | 0.085635 | 0.029476 | -0.056159 | 15999 | 63 | 1 |
| 54 | 2 | Dense GRU h116 | 1 | 0.996040 | 0.998955 | 0.002915 | 0.063921 | 0.032843 | -0.031078 | 15999 | 63 | 1 |
| 1 | 6 | Dense GRU h116 | 2 | 0.962929 | 0.976258 | 0.013329 | 0.092111 | 0.073714 | -0.018397 | 15999 | 63 | 1 |
| 2 | 3 | Dense GRU h116 | 2 | 0.963434 | 0.977693 | 0.014259 | 0.092017 | 0.071870 | -0.020147 | 15999 | 63 | 1 |
| 8 | 8 | Dense GRU h116 | 2 | 0.991747 | 0.998018 | 0.006271 | 0.125881 | 0.061683 | -0.064198 | 15999 | 63 | 1 |
| 13 | 7 | Dense GRU h116 | 2 | 0.899162 | 0.972512 | 0.073350 | 0.096903 | 0.050593 | -0.046309 | 15999 | 63 | 1 |
| 14 | 1 | Dense GRU h116 | 2 | 0.894408 | 0.977457 | 0.083049 | 0.086284 | 0.039868 | -0.046416 | 15999 | 63 | 1 |
| 33 | 5 | Dense GRU h116 | 2 | 0.993372 | 0.998850 | 0.005477 | 0.106336 | 0.044299 | -0.062038 | 15999 | 63 | 1 |
| 36 | 4 | Dense GRU h116 | 2 | 0.914035 | 0.997378 | 0.083343 | 0.257682 | 0.045001 | -0.212681 | 15999 | 63 | 1 |
| 43 | 0 | Dense GRU h116 | 2 | 0.950915 | 0.995284 | 0.044369 | 0.200367 | 0.062107 | -0.138261 | 15999 | 63 | 1 |
| 54 | 2 | Dense GRU h116 | 2 | 0.995208 | 0.996962 | 0.001754 | 0.070316 | 0.055990 | -0.014326 | 15999 | 63 | 1 |

## 10. Sequence-position stability

| Model | Q1 RMSE | Q2 RMSE | Q3 RMSE | Q4 RMSE | Q4-Q1 | Q4/Q1 | Q4>Q1 seed-trajectory observations | Seed-trajectory observations | Q4>Q1 seed-averaged trajectories | Test trajectories |
|---|---|---|---|---|---|---|---|---|---|---|
| Bounded KAN-GRU h32 | 0.039964 | 0.037998 | 0.037378 | 0.038106 | -0.001858 | 0.964377 | 9 | 27 | 3 | 9 |
| Dense GRU h116 | 0.056446 | 0.052099 | 0.051197 | 0.051609 | -0.004837 | 0.927631 | 9 | 27 | 1 | 9 |

Does error systematically grow toward the end? **NO**. The earlier count of 9 referred to 9/27 seed-trajectory observations for each model, not nine trajectories. After averaging seeds within trajectory, Q4 exceeded Q1 for 3/9 bounded KAN-GRU trajectories and 1/9 GRU trajectories; aggregate Q4 RMSE was lower than Q1 for both models.

## 11. Hidden-state stability

| Model | Minimum norm | Maximum norm | Mean norm | Mean norm SD | Start norm | Middle norm | End norm | Q2 norm mean | Q4 norm mean | Q4-Q2 norm | NaN count | Inf count | Fraction abs(activation)>=0.99 | Evidence of late drift | Evidence of explosion | Strong saturation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Bounded KAN-GRU h32 | 1.095225 | 4.539500 | 2.713287 | 0.325781 | 1.257313 | 2.407968 | 2.421946 | 2.713619 | 2.714254 | 0.000635 | 0 | 0 | 0.047978 | NO | NO | NO |
| Dense GRU h116 | 0.703710 | 8.107992 | 5.560264 | 0.507932 | 1.214837 | 5.277694 | 5.322918 | 5.561715 | 5.560189 | -0.001526 | 0 | 0 | 0.024448 | NO | NO | NO |

Operational rules: late drift is flagged when the absolute difference between Q4 and Q2 mean hidden-state norm exceeds twice the mean within-trace norm SD; this excludes the expected transient from zero initialization. Explosion is flagged by non-finite values or a maximum norm over 20 times the mean norm; strong saturation is flagged when at least 90% of recurrent activations have absolute value >=0.99. These are numerical diagnostics of the learned hidden-state representation, not physical-state identification.

## 12. Reviewer 2.2 answer values

- Trajectory length: 15,999 samples.
- 256-sample fraction: 0.01600100 (1.600100%).
- Window table status: complete.
- Evaluated samples per trajectory: 8,064 (window 128), 8,192 (window 256), and 8,704 (window 512).
- KAN-GRU optimizer steps at fixed 100 epochs: 15,300 / 7,700 / 3,900 for windows 128 / 256 / 512.
- Interpretation: sensitivity of the complete fixed-epoch protocol to sequence length; optimization budgets are not update-matched.
- Figure: `revision_experiments\figures\figure_02_window_length_sensitivity.png`.
- Suggested factual conclusion: broadly stable but magnitude varies.

| Window | Model | R2 | RMSE |
|---|---|---|---|
| 128 | Bounded KAN-GRU h32 | 0.995437 | 0.059322 |
| 256 | Bounded KAN-GRU h32 | 0.994476 | 0.064944 |
| 512 | Bounded KAN-GRU h32 | 0.988630 | 0.091899 |
| 128 | Dense GRU h116 | 0.990827 | 0.084092 |
| 256 | Dense GRU h116 | 0.983067 | 0.112285 |
| 512 | Dense GRU h116 | 0.957201 | 0.179449 |

| Window | Mean paired delta R2 | 95% CI low | 95% CI high |
|---|---|---|---|
| 128 | 0.011691 | 0.003696 | 0.022312 |
| 256 | 0.024714 | 0.013414 | 0.036671 |
| 512 | 0.054363 | 0.024261 | 0.088971 |

## 13. Reviewer 2.8 answer values

| Model | Reset R2 | Continuous R2 | Delta R2 | Reset RMSE | Continuous RMSE | Delta RMSE |
|---|---|---|---|---|---|---|
| Bounded KAN-GRU h32 | 0.981846 | 0.994185 | 0.012339 | 0.064588 | 0.038465 | -0.026123 |
| Dense GRU h116 | 0.957850 | 0.986491 | 0.028641 | 0.106137 | 0.053120 | -0.053016 |

| Model | Q1 RMSE | Q2 RMSE | Q3 RMSE | Q4 RMSE |
|---|---|---|---|---|
| Bounded KAN-GRU h32 | 0.039964 | 0.037998 | 0.037378 | 0.038106 |
| Dense GRU h116 | 0.056446 | 0.052099 | 0.051197 | 0.051609 |

| Model | Minimum norm | Maximum norm | Start norm | Middle norm | End norm | NaN | Inf |
|---|---|---|---|---|---|---|---|
| Bounded KAN-GRU h32 | 1.095225 | 4.539500 | 1.257313 | 2.407968 | 2.421946 | 0 | 0 |
| Dense GRU h116 | 0.703710 | 8.107992 | 1.214837 | 5.277694 | 5.322918 | 0 | 0 |

- Representative figure: `reviewer_experiments\figures\figure_full_trajectory_continuous_inference.png`.
- Same-trajectory hidden-state PCA figure: `reviewer_experiments\figures\figure_same_trajectory_prediction_hidden_pca.png`.
- Suggested factual conclusion: use the reset-to-continuous deltas, Q1-Q4 profile, and finite-state diagnostics above; do not infer a physical device state from recurrent hidden units.

## 14. Generated figures

| Filename | Full path | What it shows | Reviewer point |
|---|---|---|---|
| figure_02_window_length_sensitivity.png | revision_experiments\figures\figure_02_window_length_sensitivity.png | R2 and RMSE versus 128/256/512 window length | R2.2 |
| figure_full_trajectory_continuous_inference.png | reviewer_experiments\figures\figure_full_trajectory_continuous_inference.png | complete input, prediction, residual, and hidden-state norm | R2.8 |
| figure_same_trajectory_prediction_hidden_pca.png | reviewer_experiments\figures\figure_same_trajectory_prediction_hidden_pca.png | same-trajectory I-V prediction, residual, hidden PCA | R2.8 |

## 15. Generated CSV files

- `revision_experiments\window_sensitivity_per_trajectory.csv`
- `revision_experiments\window_sensitivity_per_seed.csv`
- `revision_experiments\window_sensitivity_summary.csv`
- `revision_experiments\window_sensitivity_paired.csv`
- `reviewer_experiments\full_trajectory_per_trajectory.csv`
- `reviewer_experiments\full_trajectory_summary.csv`
- `reviewer_experiments\sequence_position_error.csv`
- `reviewer_experiments\sequence_position_summary.csv`
- `reviewer_experiments\hidden_state_summary.csv`

## 16. Problems or deviations

- The full-trajectory comparison was inference-only on existing final-epoch 256-sample checkpoints; no long-sequence retraining was performed.
- Full-trajectory inference ran on CPU, whereas archived and sensitivity training used Kaggle T4 GPUs. Metrics are hardware-independent; runtime is not compared across those devices.
- Reset-mode full-trajectory inference uses contiguous non-overlapping chunks to cover every sample once. The archived headline evaluator uses 256-sample windows at evaluation stride 512, so the reset-mode aggregate here is a dedicated paired control rather than the submitted headline number.
- Window length necessarily changes windows per trajectory, batches per epoch, and optimizer steps. These quantities are recorded; epochs, optimizer, features, split, seeds, and model definitions remain fixed.
- Evaluation stride was scaled as twice the window length (256/512/1024), producing 8,064/8,192/8,704 evaluated samples per trajectory. The sensitivity result therefore should not be described as a perfectly update- and sample-matched isolation of context length.
- Window result files: all present.
- Full-trajectory result files: all 54 model-seed-trajectory pairs present.
- No missing/non-finite hidden-state values were concealed; exact NaN and Inf counts are tabulated.

## 17. Final manuscript-ready summary

### Reviewer 2.2 - manuscript-ready values

The original SDC trajectories contain 15,999 samples, so the reference 256-sample window represents 1.600100% of a trajectory. The complete 128/256/512 metrics and paired trajectory analysis are reported in Sections 3-6 above. Evaluation covered 8,064/8,192/8,704 samples per trajectory and KAN-GRU used 15,300/7,700/3,900 optimizer steps, respectively, because strides scaled with window length under a fixed 100-epoch protocol. The objective classification is: broadly stable but magnitude varies.

### Reviewer 2.8 - manuscript-ready values

All nine held-out trajectories and all three original seeds were evaluated in contiguous reset-window and single-reset continuous modes. Exact R2/RMSE deltas, quarter-wise errors, reset counts, and hidden-state finite/norm ranges are reported in Sections 7-11 above.
