"""Rebuild the reviewer sensitivity plot from the released CSV, without training."""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from finalize_revision_window_sensitivity import save_plot


def main():
    with (ROOT / "revision_experiments/window_sensitivity_summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("window_length", "test_r2_mean", "test_r2_std", "test_rmse_mean", "test_rmse_std"):
            row[key] = float(row[key])
    print(save_plot(rows, ROOT / "regenerated_figures"))


if __name__ == "__main__":
    main()
