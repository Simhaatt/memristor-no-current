# Memristor no-current benchmark: research artifacts

Source repository: https://github.com/Simhaatt/memristor-no-current

Research code and archived results associated with *No-current memory inference for memristor dynamics: a cross-regime evaluation of recurrent and KAN-gated models against compact device models*. Paper DOI supplied by the authors: https://doi.org/10.1007/s10825-026-02674-6.

## Start here

- `docs/REVIEWER_EXPERIMENT_RESULTS.md`: complete final 128/256/512 and no-reset results.
- `results/final_tables/`: main SDC, four-dopant, RRAM 100-epoch, distillation and timing CSVs.
- `revision_experiments/window_sensitivity_*.csv`: 18 model/window/seed runs and 162 trajectory records.
- `reviewer_experiments/`: continuous-state metrics, quarter-wise error and hidden-state summaries.
- `checkpoints/INVENTORY.csv`: available weights with relative locations, byte sizes and SHA-256 hashes. Locations preserve the historical script defaults.
- `data/splits/`: exact archived partition metadata; no raw measurement files are distributed.
- `REPRODUCTION.md`: commands and script-to-artifact mapping.
- `PUBLICATION_CHECKLIST.md`: remaining author/repository metadata and DOI instructions.

## Installation

Use Python 3.11 and a fresh virtual environment:

```bash
python -m venv .venv
# Activate .venv using your operating system's standard command.
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python analysis/verify_release.py
```

`requirements.txt` provides a reference environment; it is not a recovered historical lockfile. Archived Kaggle runs recorded PyTorch 2.10.0+cu128 and Tesla T4. The revision CPU inference ran with Python 3.14.3 / PyTorch 2.12.0+cpu / NumPy 2.4.2. Full per-run training budgets are preserved in JSON logs. Seeds were set for Python, NumPy and Torch; deterministic algorithms were not enforced. Checkpoint evaluation should agree to approximately 1e-5 in R2 on a comparable environment; this is an evaluation check tolerance, not a guarantee about retraining. Seed SDs in the result CSVs describe retraining variability. The tolerance has not been validated across all GPU architectures.

## Data and physical units

Four regimes are included: original SDC state characterisation, four-dopant SDC sine, the public RRAM-LSTM benchmark, and Southampton D0132. Run `python data/download.py --all` to retrieve original sources, then `python data/preprocess.py --regime all`. Dataset licenses remain those of the upstream providers. Processed numeric partitions are rebuilt locally from the source data and archived split definitions; they are not bundled pending source-license clearance.

The archived original SDC loader uses `V = Volt.1 - Volt` and target `Volt`. This target is a sense-voltage proxy. With the paper's 100 kOhm resistor, physical current is `I_A = Volt / 100000`; a physical-unit RMSE requires the same conversion. The release preserves the training target and saved checkpoints. R2 is invariant under a common affine target rescaling; RMSE is not. Standardized target RMSE must be labeled as standardized target units. Four-dopant and D0132 have their own source unit conventions; they must not inherit the original SDC resistor conversion.

The original `no_current` input selects `[V_t, V_(t-1), dV_t, sweep_direction]` and excludes the explicit previous-current input. `V_t` itself uses the sense channel in its construction as described above; the original protocol must not be described as using an independently measured source-voltage signal alone.

## Scope of reproducibility

The CSVs can be inspected without training. Checkpoint evaluation requires the regenerated caches. Re-training requires the original data, the archived settings, and compute time. The KAN implementation is custom fixed-grid cubic B-splines with SiLU base functions. Fused and hybrid implementations are in `src/memristorkan/models/recurrent.py`.

The final typeset manuscript is not included in this workspace. Figure scripts are mapped by artifact name, not assumed journal figure numbers. Existing figures are included for reference; `analysis/make_figures.py` regenerates the window comparison directly from the released CSVs with explicit standardized RMSE labeling. Consult `REPRODUCTION.md` for other original generators and their data requirements.

The code is MIT-licensed; upstream data are excluded from that grant. Cite the paper and, after the authors create it, the version-specific software DOI from Zenodo. This local archive does not yet have a software DOI.
