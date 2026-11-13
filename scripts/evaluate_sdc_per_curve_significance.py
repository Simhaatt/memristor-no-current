"""Per-curve paired significance for saved SDC sequence checkpoints.

This is evaluation-only. It compares two model families on full held-out SDC
curves, pairing by seed and curve index. For the locked test split this gives
9 curves x 3 seeds = 27 paired observations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_sdc_sequence import (
    SequenceVariant,
    WindowedCurveDataset,
    build_feature_array,
    build_sequence_model,
    feature_names_for_mode,
)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def variant_from_checkpoint(payload: dict[str, Any]) -> SequenceVariant:
    variant = payload.get("variant")
    if not isinstance(variant, dict):
        raise ValueError(f"checkpoint has no variant dictionary: {payload.keys()}")
    allowed = {"name", "model_name", "hidden_dim", "grid_size", "gauge_every_steps", "moment_mode"}
    return SequenceVariant(**{key: value for key, value in variant.items() if key in allowed})


def seed_from_path(path: Path) -> int | None:
    match = re.search(r"seed(\d+)", path.name)
    return int(match.group(1)) if match else None


def checkpoint_map(paths: list[Path], device: torch.device) -> dict[int, tuple[Path, dict[str, Any]]]:
    mapped: dict[int, tuple[Path, dict[str, Any]]] = {}
    for path in paths:
        payload = load_checkpoint(path, device)
        seed = payload.get("seed")
        if seed is None:
            seed = seed_from_path(path)
        if seed is None:
            raise ValueError(f"could not infer seed for {path}")
        mapped[int(seed)] = (path, payload)
    return mapped


def regression_row(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred = pred.reshape(-1).float()
    target = target.reshape(-1).float()
    error = pred - target
    denom = (target - target.mean()).pow(2).sum().clamp_min(1e-12)
    return {
        "r2": float((1.0 - error.pow(2).sum() / denom).item()),
        "rmse": float(torch.sqrt(error.pow(2).mean()).item()),
        "mae": float(error.abs().mean().item()),
    }


@torch.no_grad()
def evaluate_checkpoint_curves(
    checkpoint_path: Path,
    payload: dict[str, Any],
    *,
    cache: Path,
    split: str,
    device: torch.device,
) -> list[dict[str, Any]]:
    variant = variant_from_checkpoint(payload)
    feature_mode = str(payload.get("feature_mode", "no_current"))
    data = np.load(cache)
    raw_x = data[f"{split}_curve_x"].astype(np.float32)
    raw_y = data[f"{split}_curve_y"].astype(np.float32)
    lengths = data[f"{split}_curve_lengths"].astype(np.int32)
    x = build_feature_array(raw_x, lengths, feature_mode)

    model = build_sequence_model(variant, input_dim=x.shape[-1]).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    rows: list[dict[str, Any]] = []
    for curve_idx, length_value in enumerate(lengths):
        length = int(length_value)
        if length <= 1:
            continue
        curve_x = torch.from_numpy(x[curve_idx : curve_idx + 1, :length]).to(device)
        curve_y = torch.from_numpy(raw_y[curve_idx : curve_idx + 1, :length]).to(device)
        pred = model(curve_x)
        metrics = regression_row(pred.cpu(), curve_y.cpu())
        rows.append(
            {
                "variant": variant.name,
                "seed": int(payload.get("seed", seed_from_path(checkpoint_path) or -1)),
                "curve_index": curve_idx,
                "length": length,
                "feature_mode": feature_mode,
                "checkpoint_path": str(checkpoint_path),
                **metrics,
            }
        )
    return rows


@torch.no_grad()
def evaluate_checkpoint_windowed_curves(
    checkpoint_path: Path,
    payload: dict[str, Any],
    *,
    cache: Path,
    split: str,
    window: int,
    stride: int,
    batch_size: int,
    device: torch.device,
    num_workers: int,
) -> list[dict[str, Any]]:
    """Evaluate using the same windowed test protocol, then aggregate by curve."""

    variant = variant_from_checkpoint(payload)
    feature_mode = str(payload.get("feature_mode", "no_current"))
    dataset = WindowedCurveDataset(cache, split, window=window, stride=stride, feature_mode=feature_mode)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model = build_sequence_model(variant, input_dim=dataset.feature_dim).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    grouped: dict[int, dict[str, list[torch.Tensor] | int]] = {}
    offset = 0
    for x, y in loader:
        batch_windows = dataset.windows[offset : offset + x.shape[0]]
        offset += x.shape[0]
        pred = model(x.to(device)).cpu()
        for item_idx, (curve_idx, _start) in enumerate(batch_windows):
            bucket = grouped.setdefault(curve_idx, {"pred": [], "target": [], "length": 0})
            bucket["pred"].append(pred[item_idx])  # type: ignore[index,union-attr]
            bucket["target"].append(y[item_idx])  # type: ignore[index,union-attr]
            bucket["length"] = int(bucket["length"]) + int(y.shape[1])

    rows: list[dict[str, Any]] = []
    for curve_idx, bucket in sorted(grouped.items()):
        pred_curve = torch.cat(bucket["pred"], dim=0)  # type: ignore[arg-type]
        target_curve = torch.cat(bucket["target"], dim=0)  # type: ignore[arg-type]
        metrics = regression_row(pred_curve, target_curve)
        rows.append(
            {
                "variant": variant.name,
                "seed": int(payload.get("seed", seed_from_path(checkpoint_path) or -1)),
                "curve_index": curve_idx,
                "length": int(bucket["length"]),
                "feature_mode": feature_mode,
                "checkpoint_path": str(checkpoint_path),
                **metrics,
            }
        )
    return rows


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def sign_test_p_value(positive: int, n: int) -> float:
    """Two-sided exact sign test under p=0.5."""

    if n <= 0:
        return float("nan")
    k = min(positive, n - positive)
    prob = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return float(min(1.0, 2.0 * prob))


def paired_tests(diffs: list[float]) -> dict[str, float | str]:
    clean = [float(x) for x in diffs if math.isfinite(float(x))]
    n = len(clean)
    if n == 0:
        return {
            "n": 0,
            "mean_diff": float("nan"),
            "std_diff": float("nan"),
            "paired_t_p": float("nan"),
            "wilcoxon_p": float("nan"),
            "sign_test_p": float("nan"),
            "stats_backend": "none",
        }
    mean_diff = mean(clean)
    std_diff = pstdev(clean)
    positive = sum(1 for x in clean if x > 0)
    sign_p = sign_test_p_value(positive, sum(1 for x in clean if x != 0))
    backend = "normal_fallback"
    paired_t_p = float("nan")
    wilcoxon_p = float("nan")
    if n > 1 and std_diff > 0:
        t_stat = mean_diff / (std_diff / math.sqrt(n))
        paired_t_p = 2.0 * (1.0 - normal_cdf(abs(t_stat)))
    try:
        from scipy import stats  # type: ignore

        if n > 1:
            paired_t_p = float(stats.ttest_1samp(clean, 0.0).pvalue)
        nonzero = [x for x in clean if x != 0]
        if nonzero:
            wilcoxon_p = float(stats.wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided").pvalue)
        backend = "scipy"
    except Exception:
        pass
    return {
        "n": n,
        "mean_diff": float(mean_diff),
        "std_diff": float(std_diff),
        "paired_t_p": float(paired_t_p),
        "wilcoxon_p": float(wilcoxon_p),
        "sign_test_p": float(sign_p),
        "wins": positive,
        "losses": sum(1 for x in clean if x < 0),
        "ties": sum(1 for x in clean if x == 0),
        "stats_backend": backend,
    }


def summarize_pair(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_a = {(row["seed"], row["curve_index"]): row for row in rows_a}
    by_b = {(row["seed"], row["curve_index"]): row for row in rows_b}
    keys = sorted(set(by_a) & set(by_b))
    paired = []
    for seed, curve_idx in keys:
        a = by_a[(seed, curve_idx)]
        b = by_b[(seed, curve_idx)]
        paired.append(
            {
                "seed": seed,
                "curve_index": curve_idx,
                "length": a["length"],
                "model_a": a["variant"],
                "model_b": b["variant"],
                "a_r2": a["r2"],
                "b_r2": b["r2"],
                "r2_diff_a_minus_b": a["r2"] - b["r2"],
                "a_rmse": a["rmse"],
                "b_rmse": b["rmse"],
                "rmse_diff_a_minus_b": a["rmse"] - b["rmse"],
                "a_mae": a["mae"],
                "b_mae": b["mae"],
                "mae_diff_a_minus_b": a["mae"] - b["mae"],
            }
        )
    r2_tests = paired_tests([row["r2_diff_a_minus_b"] for row in paired])
    rmse_tests = paired_tests([-row["rmse_diff_a_minus_b"] for row in paired])
    summary = {
        "model_a": rows_a[0]["variant"] if rows_a else "",
        "model_b": rows_b[0]["variant"] if rows_b else "",
        "paired_observations": len(paired),
        "curves_per_seed": len({row["curve_index"] for row in paired}),
        "seeds": sorted({row["seed"] for row in paired}),
        "r2_model_a_mean": mean([row["a_r2"] for row in paired]) if paired else float("nan"),
        "r2_model_b_mean": mean([row["b_r2"] for row in paired]) if paired else float("nan"),
        "r2_diff_tests": r2_tests,
        "rmse_model_a_mean": mean([row["a_rmse"] for row in paired]) if paired else float("nan"),
        "rmse_model_b_mean": mean([row["b_rmse"] for row in paired]) if paired else float("nan"),
        "rmse_advantage_tests": rmse_tests,
    }
    return paired, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    r2 = summary["r2_diff_tests"]
    rmse = summary["rmse_advantage_tests"]
    lines = [
        "# SDC Per-Curve Paired Significance",
        "",
        f"Comparison: `{summary['model_a']}` minus `{summary['model_b']}`.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Paired observations | {summary['paired_observations']} |",
        f"| Curves per seed | {summary['curves_per_seed']} |",
        f"| Seeds | {', '.join(str(seed) for seed in summary['seeds'])} |",
        f"| Mean R2 model A | {summary['r2_model_a_mean']:.6f} |",
        f"| Mean R2 model B | {summary['r2_model_b_mean']:.6f} |",
        f"| Mean R2 diff A-B | {r2['mean_diff']:.6f} |",
        f"| R2 wins/losses/ties | {r2['wins']}/{r2['losses']}/{r2['ties']} |",
        f"| Paired t-test p, R2 diff | {r2['paired_t_p']:.6g} |",
        f"| Wilcoxon p, R2 diff | {r2['wilcoxon_p']:.6g} |",
        f"| Sign-test p, R2 diff | {r2['sign_test_p']:.6g} |",
        f"| Mean RMSE model A | {summary['rmse_model_a_mean']:.6f} |",
        f"| Mean RMSE model B | {summary['rmse_model_b_mean']:.6f} |",
        f"| RMSE advantage wins/losses/ties | {rmse['wins']}/{rmse['losses']}/{rmse['ties']} |",
        "",
        "Suggested table line:",
        "",
        (
            f"`{summary['model_a']}` outperformed `{summary['model_b']}` on "
            f"{r2['wins']}/{summary['paired_observations']} seed-curve R2 pairs "
            f"(mean paired R2 difference {r2['mean_diff']:.4f}; "
            f"paired t-test p={r2['paired_t_p']:.3g}, Wilcoxon p={r2['wilcoxon_p']:.3g})."
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-a-checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--model-b-checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("data/processed/sdc_state_splits.npz"))
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--evaluation-mode", default="windows", choices=["windows", "full_curve"])
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results/test_evaluation/sdc_per_curve_significance.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    a_map = checkpoint_map(args.model_a_checkpoints, device)
    b_map = checkpoint_map(args.model_b_checkpoints, device)
    shared_seeds = sorted(set(a_map) & set(b_map))
    if not shared_seeds:
        raise ValueError("no shared seeds between model A and model B checkpoints")

    rows_a: list[dict[str, Any]] = []
    rows_b: list[dict[str, Any]] = []
    for seed in shared_seeds:
        a_path, a_payload = a_map[seed]
        b_path, b_payload = b_map[seed]
        print(f"Evaluating seed={seed}: {a_path.name} vs {b_path.name}", flush=True)
        if args.evaluation_mode == "full_curve":
            rows_a.extend(evaluate_checkpoint_curves(a_path, a_payload, cache=args.cache, split=args.split, device=device))
            rows_b.extend(evaluate_checkpoint_curves(b_path, b_payload, cache=args.cache, split=args.split, device=device))
        else:
            rows_a.extend(
                evaluate_checkpoint_windowed_curves(
                    a_path,
                    a_payload,
                    cache=args.cache,
                    split=args.split,
                    window=args.window,
                    stride=args.stride,
                    batch_size=args.batch_size,
                    device=device,
                    num_workers=args.num_workers,
                )
            )
            rows_b.extend(
                evaluate_checkpoint_windowed_curves(
                    b_path,
                    b_payload,
                    cache=args.cache,
                    split=args.split,
                    window=args.window,
                    stride=args.stride,
                    batch_size=args.batch_size,
                    device=device,
                    num_workers=args.num_workers,
                )
            )

    paired, summary = summarize_pair(rows_a, rows_b)
    payload = {
        "split": args.split,
        "model_a_curve_rows": rows_a,
        "model_b_curve_rows": rows_b,
        "paired_rows": paired,
        "summary": summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(args.out.with_suffix(".paired.csv"), paired)
    write_markdown(args.out.with_suffix(".md"), summary)
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
