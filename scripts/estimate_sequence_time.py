"""Estimate longer sequence-run time from a calibration JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("calibration_json")
    parser.add_argument("--target-epochs", type=int, default=5)
    parser.add_argument("--target-seeds", type=int, default=1)
    args = parser.parse_args()

    payload = json.loads(Path(args.calibration_json).read_text(encoding="utf-8"))
    summary = {}
    total_seconds = 0.0
    for variant, row in payload["summary"].items():
        seconds_per_epoch = row["wall_clock_seconds_mean"]
        estimate = seconds_per_epoch * args.target_epochs * args.target_seeds
        summary[variant] = {
            "estimated_seconds": estimate,
            "estimated_hours": estimate / 3600.0,
        }
        total_seconds += estimate
    summary["total"] = {
        "estimated_seconds": total_seconds,
        "estimated_hours": total_seconds / 3600.0,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
