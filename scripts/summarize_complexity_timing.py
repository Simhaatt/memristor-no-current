"""Summarize complexity and timing from SDC Track A result JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from memristorkan.complexity import model_complexity
from memristorkan.train import ModelConfig, build_model


MODEL_FOR_VARIANT = {
    "mlp": "mlp",
    "kan": "kan",
    "kan_standard_weight_decay": "kan",
    "gauge_kan": "gauge_kan",
    "gauge_kan_reset_moments": "gauge_kan",
    "gauge_kan_standard_weight_decay": "gauge_kan",
    "gauge_kan_amplitude_decay": "gauge_kan",
}


def load_results(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    return payload.get("results", [])


def complexity_for_row(row: dict[str, object]) -> dict[str, int]:
    existing = row.get("complexity")
    if isinstance(existing, dict):
        return existing
    variant = str(row["variant"])
    model_name = MODEL_FOR_VARIANT.get(variant, variant)
    model = build_model(
        ModelConfig(
            name=model_name,
            input_dim=5,
            output_dim=1,
            hidden_dims=(int(row.get("hidden_dim", 16)),),
            hidden_dim=int(row.get("hidden_dim", 16)),
            grid_size=int(row.get("grid_size", 8)),
            spline_order=int(row.get("spline_order", 3)),
        )
    )
    return model_complexity(model)


def timing_for_row(row: dict[str, object]) -> dict[str, float]:
    timing = row.get("timing")
    if isinstance(timing, dict):
        return {key: float(value) for key, value in timing.items()}
    wall = float(row.get("wall_clock_seconds", 0.0))
    steps = max(1, int(row.get("steps", 1)))
    return {
        "wall_clock_seconds": wall,
        "seconds_per_step": wall / steps,
        "projection_wall_clock_seconds": 0.0,
        "projection_seconds_per_event": 0.0,
        "projection_wall_clock_fraction": 0.0,
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["variant"]), []).append(row)

    summary = {}
    for variant, variant_rows in grouped.items():
        timings = [timing_for_row(row) for row in variant_rows]
        complexities = [complexity_for_row(row) for row in variant_rows]
        wall = np.asarray([timing["wall_clock_seconds"] for timing in timings], dtype=float)
        sec_per_step = np.asarray([timing["seconds_per_step"] for timing in timings], dtype=float)
        projection_fraction = np.asarray([timing["projection_wall_clock_fraction"] for timing in timings], dtype=float)
        projection_per_event = np.asarray([timing["projection_seconds_per_event"] for timing in timings], dtype=float)
        summary[variant] = {
            "runs": len(variant_rows),
            "wall_clock_seconds_mean": float(wall.mean()),
            "wall_clock_seconds_std": float(wall.std(ddof=0)),
            "seconds_per_step_mean": float(sec_per_step.mean()),
            "seconds_per_step_std": float(sec_per_step.std(ddof=0)),
            "projection_seconds_per_event_mean": float(projection_per_event.mean()),
            "projection_wall_clock_fraction_mean": float(projection_fraction.mean()),
            "complexity": complexities[0],
        }
    return summary


def markdown_table(summary: dict[str, object]) -> str:
    lines = [
        "| Variant | Params | KAN edges | Spline coeffs | Mean wall sec | Mean sec/step | Projection/event sec | Projection fraction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, row in summary.items():
        complexity = row["complexity"]
        lines.append(
            "| {variant} | {params} | {edges} | {coeffs} | {wall:.2f} | {step:.4f} | {proj_event:.6f} | {proj_frac:.4%} |".format(
                variant=variant,
                params=complexity["trainable_parameters"],
                edges=complexity["kan_edges"],
                coeffs=complexity["spline_coefficients"],
                wall=row["wall_clock_seconds_mean"],
                step=row["seconds_per_step_mean"],
                proj_event=row["projection_seconds_per_event_mean"],
                proj_frac=row["projection_wall_clock_fraction_mean"],
            )
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/sdc_track_a_cpu_1000x3.json")
    parser.add_argument("--out-json", default="results/sdc_complexity_timing_summary.json")
    parser.add_argument("--out-md", default="results/sdc_complexity_timing_summary.md")
    args = parser.parse_args()

    rows = load_results(Path(args.input))
    summary = summarize(rows)
    Path(args.out_json).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    md = "# SDC Track A Complexity And Timing\n\n"
    md += f"Source results: `{args.input}`\n\n"
    md += markdown_table(summary)
    md += "\n\nNote: older result files include total wall-clock time but not isolated projection-overhead timing. New runs include projection overhead fields automatically.\n"
    Path(args.out_md).write_text(md, encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
