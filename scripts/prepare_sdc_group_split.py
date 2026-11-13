"""Prepare SDC-style train/val/test caches using path-derived groups.

Use this for dopant-split or device-family split experiments once a raw SDC-like
folder contains group labels in file paths. The output NPZ has the same schema
as `prepare_main_sdc.py`, so `scripts/run_sdc_sequence.py` can consume it
unchanged.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from memristorkan.data import load_sdc_state_curves


FEATURE_NAMES = ["V_t", "V_t_minus_1", "I_t_minus_1", "dV_t", "sweep_direction"]
TARGET_NAMES = ["I_t"]


def relative_path(curve: dict[str, object], root: Path) -> str:
    path = Path(str(curve["path"]))
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def infer_group(rel_path: str, *, group_regex: str | None, group_level: int | None) -> str:
    if group_regex:
        match = re.search(group_regex, rel_path)
        if not match:
            raise ValueError(f"group regex did not match path: {rel_path}")
        if match.groupdict():
            return next(iter(match.groupdict().values()))
        if match.groups():
            return match.group(1)
        return match.group(0)
    parts = rel_path.replace("\\", "/").split("/")
    if "state_data" in parts:
        parts = parts[parts.index("state_data") + 1 :]
    if group_level is None:
        group_level = 0
    if group_level >= len(parts):
        raise ValueError(f"group level {group_level} out of range for path: {rel_path}")
    return parts[group_level]


def concat_curves(curves: list[dict[str, object]], indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = []
    ys = []
    curve_ids = []
    for curve_idx in indices:
        curve = curves[int(curve_idx)]
        x = curve["x"]
        y = curve["y"]
        xs.append(x)
        ys.append(y)
        curve_ids.append(np.full((x.shape[0],), int(curve_idx), dtype=np.int32))
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), np.concatenate(curve_ids, axis=0)


def choose_splits(
    groups: list[str],
    *,
    train_groups: list[str] | None,
    val_groups: list[str] | None,
    test_groups: list[str],
    val_fraction: float,
    seed: int,
) -> dict[str, np.ndarray]:
    all_groups = sorted(set(groups))
    test_set = set(test_groups)
    if not test_set:
        raise ValueError("at least one --test-groups value is required")
    if train_groups:
        train_set = set(train_groups)
    else:
        train_set = set(all_groups) - test_set
    val_set = set(val_groups or [])
    overlap = (train_set & test_set) | (train_set & val_set) | (val_set & test_set)
    if overlap:
        raise ValueError(f"group overlap across splits: {sorted(overlap)}")
    unknown = (train_set | val_set | test_set) - set(all_groups)
    if unknown:
        raise ValueError(f"requested groups not present: {sorted(unknown)}; available={all_groups}")

    indices_by_group: dict[str, list[int]] = {group: [] for group in all_groups}
    for idx, group in enumerate(groups):
        indices_by_group[group].append(idx)

    train_indices = [idx for group in sorted(train_set) for idx in indices_by_group[group]]
    test_indices = [idx for group in sorted(test_set) for idx in indices_by_group[group]]
    if val_set:
        val_indices = [idx for group in sorted(val_set) for idx in indices_by_group[group]]
    else:
        rng = np.random.default_rng(seed)
        shuffled = np.asarray(train_indices, dtype=np.int32)
        rng.shuffle(shuffled)
        val_count = max(1, int(round(len(shuffled) * val_fraction))) if len(shuffled) > 1 else 0
        val_indices = shuffled[:val_count].astype(int).tolist()
        train_indices = shuffled[val_count:].astype(int).tolist()

    if not train_indices or not test_indices:
        raise ValueError("empty train or test split after grouping")
    if not val_indices:
        raise ValueError("empty validation split after grouping")
    return {
        "train": np.asarray(train_indices, dtype=np.int32),
        "val": np.asarray(val_indices, dtype=np.int32),
        "test": np.asarray(test_indices, dtype=np.int32),
    }


def add_curve_arrays(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, object],
    split_name: str,
    curves: list[dict[str, object]],
    indices: np.ndarray,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
) -> None:
    x, y, curve_ids = concat_curves(curves, indices)
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
    for curve_idx in indices:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/sdc/memristor-state-characterisation"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metadata-out", type=Path, required=True)
    parser.add_argument("--group-regex", default=None, help="Regex applied to relative path; first capture/groupdict value is group.")
    parser.add_argument("--group-level", type=int, default=0, help="Path part after state_data to use when --group-regex is omitted.")
    parser.add_argument("--train-groups", nargs="+", default=None)
    parser.add_argument("--val-groups", nargs="+", default=None)
    parser.add_argument("--test-groups", nargs="+", required=True)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    curves, summary = load_sdc_state_curves(args.root)
    rel_paths = [relative_path(curve, args.root) for curve in curves]
    groups = [
        infer_group(path, group_regex=args.group_regex, group_level=args.group_level)
        for path in rel_paths
    ]
    splits = choose_splits(
        groups,
        train_groups=args.train_groups,
        val_groups=args.val_groups,
        test_groups=args.test_groups,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    train_x, train_y, _ = concat_curves(curves, splits["train"])
    x_mean = train_x.mean(axis=0, keepdims=True).astype(np.float32)
    x_std = train_x.std(axis=0, keepdims=True).clip(min=1e-8).astype(np.float32)
    y_mean = train_y.mean(axis=0, keepdims=True).astype(np.float32)
    y_std = train_y.std(axis=0, keepdims=True).clip(min=1e-8).astype(np.float32)
    arrays = {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}

    group_counts = {group: groups.count(group) for group in sorted(set(groups))}
    metadata: dict[str, object] = {
        "summary": summary.__dict__,
        "feature_names": FEATURE_NAMES,
        "target_names": TARGET_NAMES,
        "seed": args.seed,
        "group_regex": args.group_regex,
        "group_level": args.group_level,
        "group_counts": group_counts,
        "curve_groups": groups,
        "curve_paths": rel_paths,
        "split_curve_indices": {name: indices.astype(int).tolist() for name, indices in splits.items()},
        "split_curve_groups": {name: [groups[int(i)] for i in indices] for name, indices in splits.items()},
        "split_curve_counts": {name: int(len(indices)) for name, indices in splits.items()},
        "normalization": "Feature and target z-score statistics are computed from train curves only.",
        "source_convention": "voltage = Volt.1 - Volt; current = Volt",
    }
    for split_name, indices in splits.items():
        add_curve_arrays(arrays, metadata, split_name, curves, indices, x_mean, x_std, y_mean, y_std)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_out.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "metadata": str(args.metadata_out), **metadata}, indent=2), flush=True)


if __name__ == "__main__":
    main()
