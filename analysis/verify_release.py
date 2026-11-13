"""Check archive integrity, syntax and released result counts without training."""
import ast
import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"checksum mismatch: {relative}")
    for path in ROOT.rglob("*.py"):
        if ".venv" not in path.parts:
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    for filename, expected in (("window_sensitivity_summary.csv", 6), ("window_sensitivity_per_seed.csv", 18), ("window_sensitivity_per_trajectory.csv", 162)):
        with (ROOT / "revision_experiments" / filename).open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != expected:
            raise ValueError(f"unexpected row count: {filename}")
    with (ROOT / "checkpoints/INVENTORY.csv").open(newline="") as handle:
        checkpoints = list(csv.DictReader(handle))
    print(f"Integrity, Python syntax and result matrix verified; {len(checkpoints)} archived checkpoint paths.")


if __name__ == "__main__":
    main()
