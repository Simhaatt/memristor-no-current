"""Audit logged training budgets across memristor benchmark result files."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "training_budget_audit"


def iter_dicts(value: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if "epochs" in value and ("variant" in value or "model" in value or "model_name" in value):
            found.append(value)
        for child in value.values():
            found.extend(iter_dicts(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(iter_dicts(child))
    return found


def regime_from_path(path: Path, row: dict[str, Any]) -> str:
    text = path.as_posix().lower()
    if "rram" in text:
        task = row.get("task")
        return f"rram_{task}" if task else "rram"
    if "sdc4dopant" in text or "holdout_" in text or "pooled" in text:
        return "sdc4dopant"
    if "d0132" in text:
        return "d0132"
    if "sdc_distillation" in text:
        return "sdc_distillation"
    if "sdc" in text:
        return "original_sdc"
    return "other"


def compact_source(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def collect_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    skip_parts = {"manuscript_ready", "__pycache__"}
    for path in sorted((ROOT / "results").rglob("*.json")):
        if any(part in skip_parts for part in path.parts):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for item in iter_dicts(payload):
            epochs = item.get("epochs")
            try:
                epochs_i = int(epochs)
            except Exception:
                continue
            variant = item.get("variant") or item.get("model") or item.get("model_name") or "unknown"
            timing = item.get("timing", {}) if isinstance(item.get("timing"), dict) else {}
            epoch_summaries = item.get("epoch_summaries", [])
            if not isinstance(epoch_summaries, list):
                epoch_summaries = []
            first_loss = None
            last_loss = None
            tail_delta = None
            if epoch_summaries:
                losses = [
                    float(e["train_loss_mean"])
                    for e in epoch_summaries
                    if isinstance(e, dict) and "train_loss_mean" in e
                ]
                if losses:
                    first_loss = losses[0]
                    last_loss = losses[-1]
                    if len(losses) >= 10:
                        prev = mean(losses[-10:-5])
                        tail = mean(losses[-5:])
                        tail_delta = tail - prev
            rows.append(
                {
                    "source": compact_source(path),
                    "regime": regime_from_path(path, item),
                    "task": item.get("task", ""),
                    "fold": item.get("fold", item.get("feature_mode", "")),
                    "variant": str(variant),
                    "seed": item.get("seed", ""),
                    "epochs": epochs_i,
                    "global_steps": item.get("global_steps", ""),
                    "train_rows": item.get("train_rows", item.get("train_windows", "")),
                    "seconds_per_epoch": timing.get("seconds_per_epoch", item.get("seconds_per_epoch", "")),
                    "first_train_loss": first_loss if first_loss is not None else "",
                    "last_train_loss": last_loss if last_loss is not None else "",
                    "tail_loss_delta_last5_minus_prev5": tail_delta if tail_delta is not None else "",
                }
            )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["regime"], row["task"], row["variant"])].append(row)
    summary: list[dict[str, Any]] = []
    for (regime, task, variant), group in sorted(grouped.items()):
        epochs = [int(r["epochs"]) for r in group]
        steps = [int(r["global_steps"]) for r in group if str(r["global_steps"]).strip()]
        seconds = [float(r["seconds_per_epoch"]) for r in group if str(r["seconds_per_epoch"]).strip()]
        tail = [float(r["tail_loss_delta_last5_minus_prev5"]) for r in group if str(r["tail_loss_delta_last5_minus_prev5"]).strip()]
        summary.append(
            {
                "regime": regime,
                "task": task,
                "variant": variant,
                "runs": len(group),
                "epochs_min": min(epochs),
                "epochs_max": max(epochs),
                "global_steps_mean": mean(steps) if steps else "",
                "seconds_per_epoch_mean": mean(seconds) if seconds else "",
                "tail_loss_delta_mean": mean(tail) if tail else "",
                "sources": "; ".join(sorted({r["source"] for r in group})[:4]),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: list[dict[str, Any]]) -> None:
    interesting = [
        row
        for row in summary
        if row["regime"] in {"rram_random_walk", "rram_sine", "rram_anneal", "sdc4dopant", "d0132", "original_sdc", "sdc_distillation"}
    ]
    lines = [
        "# Training Budget Audit",
        "",
        "This audit scans result JSON files for logged epoch counts, optimizer steps, timing, and available tail-loss changes. It is intended to keep the manuscript's budget claims consistent after the RRAM 100-epoch follow-up.",
        "",
        "## Main Findings",
        "",
        "- Original SDC locked sequence runs and four-dopant SDC sine runs are logged as 100-epoch training jobs in the saved result files/checkpoint exports.",
        "- D0132 ML and hybrid-residual checks are logged as 100-epoch CPU jobs; compact-only rows have no training budget because they are supplied model references.",
        "- The old RRAM random-walk and sine tables were short-budget external checks. Both tasks now have 100-epoch, three-seed external-validation reruns and should supersede the old short-budget table.",
        "- Anneal is available in the RRAM runner, but it is much larger than sine/random-walk. The existing local result is a 1-epoch smoke, so full anneal KAN rows should be treated as optional unless GPU time is available.",
        "",
        "## Budget Summary",
        "",
        "| Regime | Task | Variant | Runs | Epochs | Mean steps | Mean sec/epoch | Tail-loss delta |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in interesting:
        sec = row["seconds_per_epoch_mean"]
        steps = row["global_steps_mean"]
        tail = row["tail_loss_delta_mean"]
        lines.append(
            f"| `{row['regime']}` | `{row['task']}` | `{row['variant']}` | {row['runs']} | "
            f"{row['epochs_min']}-{row['epochs_max']} | "
            f"{float(steps):.1f}" if steps != "" else f"| `{row['regime']}` | `{row['task']}` | `{row['variant']}` | {row['runs']} | {row['epochs_min']}-{row['epochs_max']} | "
        )
        if steps == "":
            lines[-1] += " | "
        else:
            lines[-1] += " | "
        lines[-1] += f"{float(sec):.3f}" if sec != "" else ""
        lines[-1] += " | "
        lines[-1] += f"{float(tail):.3e}" if tail != "" else ""
        lines[-1] += " |"
    lines.extend(
        [
            "",
            "## Action Items",
            "",
            "1. Use the 100-epoch RRAM random-walk and sine reruns as the final external-validation tables.",
            "2. Treat RRAM anneal as optional or split it into a separate compute note. Its train split is about 200k rows, so KAN rows are not comparable to the quick sine/random-walk runs.",
            "3. Keep four-dopant as a 100-epoch benchmark/control unless new convergence plots show a late separation; current test rows show static/recurrent/KAN models remain near-tied after full training.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = collect_rows()
    summary = summarize(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "training_budget_runs.csv", rows)
    write_csv(OUT_DIR / "training_budget_summary.csv", summary)
    write_markdown(OUT_DIR / "training_budget_audit.md", summary)
    print(json.dumps({"runs": len(rows), "summary_rows": len(summary), "out": str(OUT_DIR)}, indent=2))


if __name__ == "__main__":
    main()
