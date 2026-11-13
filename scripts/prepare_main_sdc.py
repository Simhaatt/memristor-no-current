"""Prepare the main SDC V-I dataset as normalized curve-wise splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from memristorkan.data import load_sdc_state_curves


FEATURE_NAMES = ["V_t", "V_t_minus_1", "I_t_minus_1", "dV_t", "sweep_direction"]
TARGET_NAMES = ["I_t"]


def split_indices(n: int, seed: int, train_fraction: float, val_fraction: float) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    train_end = int(round(n * train_fraction))
    val_end = train_end + int(round(n * val_fraction))
    return {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end:],
    }


def concat_curves(curves: list[dict[str, object]], indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = []
    ys = []
    curve_ids = []
    for out_id, curve_idx in enumerate(indices):
        curve = curves[int(curve_idx)]
        x = curve["x"]
        y = curve["y"]
        xs.append(x)
        ys.append(y)
        curve_ids.append(np.full((x.shape[0],), int(curve_idx), dtype=np.int32))
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), np.concatenate(curve_ids, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/raw/sdc/memristor-state-characterisation")
    parser.add_argument("--out", default="data/processed/sdc_state_splits.npz")
    parser.add_argument("--metadata-out", default="data/processed/sdc_state_metadata.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    args = parser.parse_args()

    curves, summary = load_sdc_state_curves(args.root)
    splits = split_indices(len(curves), args.seed, args.train_fraction, args.val_fraction)
    raw = {name: concat_curves(curves, indices) for name, indices in splits.items()}
    train_x, train_y, _ = raw["train"]
    x_mean = train_x.mean(axis=0, keepdims=True).astype(np.float32)
    x_std = train_x.std(axis=0, keepdims=True).clip(min=1e-8).astype(np.float32)
    y_mean = train_y.mean(axis=0, keepdims=True).astype(np.float32)
    y_std = train_y.std(axis=0, keepdims=True).clip(min=1e-8).astype(np.float32)

    arrays = {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }
    metadata = {
        "summary": summary.__dict__,
        "feature_names": FEATURE_NAMES,
        "target_names": TARGET_NAMES,
        "seed": args.seed,
        "split_curve_indices": {name: indices.astype(int).tolist() for name, indices in splits.items()},
        "split_curve_counts": {name: int(len(indices)) for name, indices in splits.items()},
        "normalization": "Feature and target z-score statistics are computed from train curves only.",
        "source_convention": "voltage = Volt.1 - Volt; current = Volt",
    }
    for split_name, (x, y, curve_ids) in raw.items():
        arrays[f"{split_name}_x"] = ((x - x_mean) / x_std).astype(np.float32)
        arrays[f"{split_name}_y"] = ((y - y_mean) / y_std).astype(np.float32)
        arrays[f"{split_name}_x_raw"] = x.astype(np.float32)
        arrays[f"{split_name}_y_raw"] = y.astype(np.float32)
        arrays[f"{split_name}_curve_id"] = curve_ids.astype(np.int32)
        metadata[f"{split_name}_rows"] = int(x.shape[0])

        curve_x = []
        curve_y = []
        curve_x_raw = []
        curve_y_raw = []
        curve_lengths = []
        for curve_idx in splits[split_name]:
            curve = curves[int(curve_idx)]
            x_curve = curve["x"].astype(np.float32)
            y_curve = curve["y"].astype(np.float32)
            curve_x.append(((x_curve - x_mean) / x_std).astype(np.float32))
            curve_y.append(((y_curve - y_mean) / y_std).astype(np.float32))
            curve_x_raw.append(x_curve)
            curve_y_raw.append(y_curve)
            curve_lengths.append(x_curve.shape[0])
        max_len = max(curve_lengths)
        feature_dim = curve_x[0].shape[-1]
        target_dim = curve_y[0].shape[-1]
        padded_x = np.zeros((len(curve_x), max_len, feature_dim), dtype=np.float32)
        padded_y = np.zeros((len(curve_y), max_len, target_dim), dtype=np.float32)
        padded_x_raw = np.zeros((len(curve_x), max_len, feature_dim), dtype=np.float32)
        padded_y_raw = np.zeros((len(curve_y), max_len, target_dim), dtype=np.float32)
        mask = np.zeros((len(curve_x), max_len), dtype=bool)
        for i, (x_curve, y_curve, x_raw_curve, y_raw_curve) in enumerate(
            zip(curve_x, curve_y, curve_x_raw, curve_y_raw, strict=True)
        ):
            n = x_curve.shape[0]
            padded_x[i, :n] = x_curve
            padded_y[i, :n] = y_curve
            padded_x_raw[i, :n] = x_raw_curve
            padded_y_raw[i, :n] = y_raw_curve
            mask[i, :n] = True
        arrays[f"{split_name}_curve_x"] = padded_x
        arrays[f"{split_name}_curve_y"] = padded_y
        arrays[f"{split_name}_curve_x_raw"] = padded_x_raw
        arrays[f"{split_name}_curve_y_raw"] = padded_y_raw
        arrays[f"{split_name}_curve_mask"] = mask
        arrays[f"{split_name}_curve_lengths"] = np.asarray(curve_lengths, dtype=np.int32)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    metadata_path = Path(args.metadata_out)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"out": str(out), "metadata": str(metadata_path), **metadata}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
