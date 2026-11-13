"""Rebuild the original caches and check their partition IDs."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", choices=["all", "sdc", "four_dopant"], default="all")
    args = parser.parse_args()
    scripts = []
    if args.regime in {"all", "sdc"}:
        scripts.append("prepare_main_sdc.py")
    if args.regime in {"all", "four_dopant"}:
        scripts.append("prepare_sdc4dopant_sine.py")
    for script in scripts:
        subprocess.run([sys.executable, str(ROOT / "scripts" / script)], cwd=ROOT, check=True)
    for archived in (ROOT / "data/splits").glob("*metadata.json"):
        generated = ROOT / "data/processed" / archived.name
        if not generated.exists():
            continue
        expected = json.loads(archived.read_text())
        actual = json.loads(generated.read_text())
        for key in ("split_curve_indices", "split_indices"):
            if key in expected and expected[key] != actual.get(key):
                raise ValueError(f"partition mismatch: {archived.name} {key}")
    print("SDC cache regeneration finished; RRAM and D0132 loaders read their source files directly.")


if __name__ == "__main__":
    main()
