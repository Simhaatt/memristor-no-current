"""Prepare four-dopant SDC sine-wave measurements as SDC-style sequence caches."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


FEATURE_NAMES = ["V_t", "V_t_minus_1", "I_t_minus_1", "dV_t", "sweep_direction"]
TARGET_NAMES = ["I_t"]

DOPANTS = {
    "1": {"short": "w", "name": "Tungsten", "symbol": "W", "series_resistance_kohm": 5.11},
    "2": {"short": "sn", "name": "Tin", "symbol": "Sn", "series_resistance_kohm": 5.11},
    "3": {"short": "cr", "name": "Chromium", "symbol": "Cr", "series_resistance_kohm": 5.11},
    "4": {"short": "c", "name": "Carbon", "symbol": "C", "series_resistance_kohm": 47.5},
}

FILENAME_RE = re.compile(
    r"^mem(?P<device>[1-4])_sine_(?P<amplitude>\d+(?:,\d+)?)V_\s*(?P<frequency>\d+)Hz\.txt$"
)


@dataclass(frozen=True)
class CurveRecord:
    curve_id: int
    path: str
    filename: str
    device: str
    dopant: str
    dopant_symbol: str
    dopant_short: str
    amplitude_v: float
    frequency_hz: int
    raw_rows: int
    decimated_rows: int
    feature_rows: int
    decimation_factor: float
    duration_seconds: float
    estimated_cycles: float
    raw_samples_per_cycle: float
    decimated_samples_per_cycle: float
    history_reset_ok: bool
    x: np.ndarray
    y: np.ndarray
    t_raw_start: float
    t_raw_end: float


def parse_filename(path: Path) -> dict[str, object]:
    match = FILENAME_RE.match(path.name)
    if not match:
        raise ValueError(f"unexpected four-dopant sine filename: {path.name}")
    device = match.group("device")
    dopant = DOPANTS[device]
    amplitude = float(match.group("amplitude").replace(",", "."))
    frequency = int(match.group("frequency"))
    return {
        "device": f"mem{device}",
        "dopant": str(dopant["name"]),
        "dopant_symbol": str(dopant["symbol"]),
        "dopant_short": str(dopant["short"]),
        "series_resistance_kohm": float(dopant["series_resistance_kohm"]),
        "amplitude_v": amplitude,
        "frequency_hz": frequency,
    }


def read_measurement(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["voltage", "current", "timestamp"],
        decimal=",",
        engine="c",
    )
    numeric = df.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
    if numeric.shape[0] < 2:
        raise ValueError(f"not enough numeric rows in {path}")
    voltage = numeric["voltage"].to_numpy(dtype=np.float32)
    current = numeric["current"].to_numpy(dtype=np.float32)
    timestamp = numeric["timestamp"].to_numpy(dtype=np.float64)
    return voltage, current, timestamp


def decimation_indices(length: int, max_points: int) -> np.ndarray:
    if max_points <= 0 or length <= max_points:
        return np.arange(length, dtype=np.int64)
    return np.linspace(0, length - 1, max_points, dtype=np.int64)


def make_features(voltage: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d_v = np.diff(voltage, prepend=voltage[0]).astype(np.float32)
    sweep_direction = np.sign(d_v).astype(np.float32)
    x = np.stack([voltage[1:], voltage[:-1], current[:-1], d_v[1:], sweep_direction[1:]], axis=-1)
    y = current[1:, None]
    return x.astype(np.float32), y.astype(np.float32)


def history_reset_ok(x: np.ndarray, voltage: np.ndarray, current: np.ndarray) -> bool:
    if x.shape[0] == 0:
        return False
    return bool(
        np.isclose(x[0, 1], voltage[0])
        and np.isclose(x[0, 2], current[0])
        and np.isclose(x[0, 3], voltage[1] - voltage[0])
    )


def load_curves(root: Path, max_points_per_curve: int) -> list[CurveRecord]:
    files = sorted(root.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"no .txt sine measurements found under {root}")

    curves: list[CurveRecord] = []
    for curve_id, path in enumerate(files):
        parsed = parse_filename(path)
        voltage, current, timestamp = read_measurement(path)
        raw_rows = int(len(voltage))
        keep = decimation_indices(raw_rows, max_points_per_curve)
        voltage_d = voltage[keep]
        current_d = current[keep]
        timestamp_d = timestamp[keep]
        x, y = make_features(voltage_d, current_d)
        duration = float(max(0.0, timestamp_d[-1] - timestamp_d[0]))
        estimated_cycles = float(max(0.0, int(parsed["frequency_hz"]) * duration))
        raw_spc = float(raw_rows / estimated_cycles) if estimated_cycles > 0 else float("inf")
        decimated_spc = float(len(keep) / estimated_cycles) if estimated_cycles > 0 else float("inf")
        curves.append(
            CurveRecord(
                curve_id=curve_id,
                path=str(path),
                filename=path.name,
                device=str(parsed["device"]),
                dopant=str(parsed["dopant"]),
                dopant_symbol=str(parsed["dopant_symbol"]),
                dopant_short=str(parsed["dopant_short"]),
                amplitude_v=float(parsed["amplitude_v"]),
                frequency_hz=int(parsed["frequency_hz"]),
                raw_rows=raw_rows,
                decimated_rows=int(len(keep)),
                feature_rows=int(x.shape[0]),
                decimation_factor=float(raw_rows / max(1, len(keep))),
                duration_seconds=duration,
                estimated_cycles=estimated_cycles,
                raw_samples_per_cycle=raw_spc,
                decimated_samples_per_cycle=decimated_spc,
                history_reset_ok=history_reset_ok(x, voltage_d, current_d),
                x=x,
                y=y,
                t_raw_start=float(timestamp_d[0]),
                t_raw_end=float(timestamp_d[-1]),
            )
        )
    return curves


def condition_key(curve: CurveRecord) -> tuple[float, int]:
    return (curve.amplitude_v, curve.frequency_hz)


def choose_validation_ids(curves: list[CurveRecord], holdout: str, val_per_dopant: int) -> set[int]:
    val_ids: set[int] = set()
    for dopant_short in sorted({curve.dopant_short for curve in curves if curve.dopant_short != holdout}):
        candidates = sorted(
            [curve for curve in curves if curve.dopant_short == dopant_short],
            key=condition_key,
        )
        if not candidates:
            continue
        if val_per_dopant <= 0:
            chosen: list[CurveRecord] = []
        elif val_per_dopant >= len(candidates):
            chosen = candidates
        else:
            indices = np.linspace(0, len(candidates) - 1, val_per_dopant, dtype=int)
            chosen = [candidates[int(i)] for i in indices]
        val_ids.update(curve.curve_id for curve in chosen)
    return val_ids


def choose_balanced_ids(candidates: list[CurveRecord], count: int) -> list[int]:
    if count <= 0:
        return []
    if count >= len(candidates):
        return [curve.curve_id for curve in candidates]
    indices = np.linspace(0, len(candidates) - 1, count, dtype=int)
    return [candidates[int(i)].curve_id for i in indices]


def split_curve_ids(curves: list[CurveRecord], holdout: str, val_per_dopant: int) -> dict[str, list[int]]:
    valid = {str(info["short"]) for info in DOPANTS.values()}
    if holdout not in valid:
        raise ValueError(f"holdout must be one of {sorted(valid)}, got {holdout!r}")

    test = [curve.curve_id for curve in curves if curve.dopant_short == holdout]
    val_ids = choose_validation_ids(curves, holdout, val_per_dopant)
    train = [
        curve.curve_id
        for curve in curves
        if curve.dopant_short != holdout and curve.curve_id not in val_ids
    ]
    val = sorted(val_ids)
    if not test or not train or not val:
        raise ValueError(f"bad split for holdout={holdout}: train={len(train)} val={len(val)} test={len(test)}")
    return {"train": train, "val": val, "test": test}


def pooled_split_curve_ids(curves: list[CurveRecord], val_per_dopant: int, test_per_dopant: int) -> dict[str, list[int]]:
    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    for dopant_short in sorted({curve.dopant_short for curve in curves}):
        candidates = sorted(
            [curve for curve in curves if curve.dopant_short == dopant_short],
            key=condition_key,
        )
        val_ids = set(choose_balanced_ids(candidates, val_per_dopant))
        remaining = [curve for curve in candidates if curve.curve_id not in val_ids]
        test_ids = set(choose_balanced_ids(remaining, test_per_dopant))
        val.extend(sorted(val_ids))
        test.extend(sorted(test_ids))
        blocked_ids = val_ids | test_ids
        train.extend(curve.curve_id for curve in candidates if curve.curve_id not in blocked_ids)
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def concat_curves(curves: list[CurveRecord], ids: Iterable[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_id = {curve.curve_id: curve for curve in curves}
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    curve_ids: list[np.ndarray] = []
    for curve_id in ids:
        curve = by_id[int(curve_id)]
        xs.append(curve.x)
        ys.append(curve.y)
        curve_ids.append(np.full((curve.x.shape[0],), int(curve_id), dtype=np.int32))
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), np.concatenate(curve_ids, axis=0)


def add_curve_arrays(
    arrays: dict[str, np.ndarray],
    split_name: str,
    curves: list[CurveRecord],
    ids: list[int],
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
) -> None:
    by_id = {curve.curve_id: curve for curve in curves}
    curve_x = []
    curve_y = []
    curve_x_raw = []
    curve_y_raw = []
    curve_lengths = []
    for curve_id in ids:
        curve = by_id[int(curve_id)]
        curve_x.append(((curve.x - x_mean) / x_std).astype(np.float32))
        curve_y.append(((curve.y - y_mean) / y_std).astype(np.float32))
        curve_x_raw.append(curve.x.astype(np.float32))
        curve_y_raw.append(curve.y.astype(np.float32))
        curve_lengths.append(curve.x.shape[0])

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


def build_cache(
    curves: list[CurveRecord],
    *,
    split_name: str,
    splits: dict[str, list[int]],
    out: Path,
    metadata_out: Path,
    source_root: Path,
    max_points_per_curve: int,
    val_per_dopant: int,
    test_per_dopant: int | None = None,
    holdout: str | None = None,
) -> dict[str, object]:
    raw = {name: concat_curves(curves, ids) for name, ids in splits.items()}
    train_x, train_y, _ = raw["train"]
    x_mean = train_x.mean(axis=0, keepdims=True).astype(np.float32)
    x_std = train_x.std(axis=0, keepdims=True).clip(min=1e-8).astype(np.float32)
    y_mean = train_y.mean(axis=0, keepdims=True).astype(np.float32)
    y_std = train_y.std(axis=0, keepdims=True).clip(min=1e-8).astype(np.float32)

    arrays: dict[str, np.ndarray] = {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }
    for cache_split, (x, y, curve_ids) in raw.items():
        arrays[f"{cache_split}_x"] = ((x - x_mean) / x_std).astype(np.float32)
        arrays[f"{cache_split}_y"] = ((y - y_mean) / y_std).astype(np.float32)
        arrays[f"{cache_split}_x_raw"] = x.astype(np.float32)
        arrays[f"{cache_split}_y_raw"] = y.astype(np.float32)
        arrays[f"{cache_split}_curve_id"] = curve_ids.astype(np.int32)
        add_curve_arrays(arrays, cache_split, curves, splits[cache_split], x_mean, x_std, y_mean, y_std)

    curve_records = [
        {
            key: value
            for key, value in asdict(curve).items()
            if key not in {"x", "y"}
        }
        for curve in curves
    ]
    split_curve_records = {
        split: [curve_records[int(curve_id)] for curve_id in ids]
        for split, ids in splits.items()
    }
    metadata: dict[str, object] = {
        "dataset": "four-dopant SDC sine-wave measurements",
        "split_name": split_name,
        "source_root": str(source_root),
        "feature_names": FEATURE_NAMES,
        "target_names": TARGET_NAMES,
        "base_features": "[V_t, V_t_minus_1, I_t_minus_1, dV_t, sweep_direction] -> I_t",
        "recommended_feature_mode": "no_current",
        "dopant_mapping": DOPANTS,
        "holdout_dopant": holdout,
        "holdout_dopant_name": next(
            (str(curve.dopant) for curve in curves if curve.dopant_short == holdout),
            None,
        ),
        "max_points_per_curve": max_points_per_curve,
        "val_per_dopant": val_per_dopant,
        "test_per_dopant": test_per_dopant,
        "split_curve_indices": splits,
        "split_curve_counts": {name: len(ids) for name, ids in splits.items()},
        "split_rows": {name: int(raw[name][0].shape[0]) for name in raw},
        "split_dopants": {
            name: sorted({curves[int(curve_id)].dopant_short for curve_id in ids})
            for name, ids in splits.items()
        },
        "curve_records": curve_records,
        "split_curve_records": split_curve_records,
        "normalization": "Feature and target z-score statistics are computed from train curves only.",
        "decimation": "Uniform linspace decimation is applied per raw trace when raw rows exceed max_points_per_curve.",
        "history_reset": (
            "V_t_minus_1, I_t_minus_1, and dV_t are computed independently per curve; "
            "windowing uses padded per-curve arrays and does not span curve boundaries."
        ),
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "split_name": split_name,
        "holdout": holdout,
        "out": str(out),
        "metadata_out": str(metadata_out),
        "split_curve_counts": metadata["split_curve_counts"],
        "split_rows": metadata["split_rows"],
        "split_dopants": metadata["split_dopants"],
    }


def build_fold(
    curves: list[CurveRecord],
    *,
    holdout: str,
    val_per_dopant: int,
    out: Path,
    metadata_out: Path,
    source_root: Path,
    max_points_per_curve: int,
) -> dict[str, object]:
    splits = split_curve_ids(curves, holdout, val_per_dopant)
    return build_cache(
        curves,
        split_name=f"leave_one_dopant_out_{holdout}",
        splits=splits,
        holdout=holdout,
        val_per_dopant=val_per_dopant,
        out=out,
        metadata_out=metadata_out,
        source_root=source_root,
        max_points_per_curve=max_points_per_curve,
    )


def validate_curves(curves: list[CurveRecord], min_samples_per_cycle_warning: float) -> dict[str, object]:
    by_dopant: dict[str, list[CurveRecord]] = {}
    for curve in curves:
        by_dopant.setdefault(curve.dopant_short, []).append(curve)
    conditions_by_dopant = {
        dopant: sorted([condition_key(curve) for curve in items])
        for dopant, items in sorted(by_dopant.items())
    }
    reference = next(iter(conditions_by_dopant.values()))
    low_spc = [
        {
            "filename": curve.filename,
            "dopant": curve.dopant_short,
            "frequency_hz": curve.frequency_hz,
            "duration_seconds": curve.duration_seconds,
            "estimated_cycles": curve.estimated_cycles,
            "decimated_samples_per_cycle": curve.decimated_samples_per_cycle,
        }
        for curve in curves
        if curve.decimated_samples_per_cycle < min_samples_per_cycle_warning
    ]
    return {
        "files": len(curves),
        "dopant_counts": {dopant: len(items) for dopant, items in sorted(by_dopant.items())},
        "conditions_match": all(conditions == reference for conditions in conditions_by_dopant.values()),
        "conditions_by_dopant": conditions_by_dopant,
        "raw_rows_total": int(sum(curve.raw_rows for curve in curves)),
        "feature_rows_total_after_decimation": int(sum(curve.feature_rows for curve in curves)),
        "min_raw_rows": int(min(curve.raw_rows for curve in curves)),
        "max_raw_rows": int(max(curve.raw_rows for curve in curves)),
        "min_decimated_samples_per_cycle": float(min(curve.decimated_samples_per_cycle for curve in curves)),
        "max_decimated_samples_per_cycle": float(max(curve.decimated_samples_per_cycle for curve in curves)),
        "min_samples_per_cycle_warning": min_samples_per_cycle_warning,
        "low_samples_per_cycle_files": low_spc,
        "history_reset_all_ok": bool(all(curve.history_reset_ok for curve in curves)),
    }


def write_summary(path: Path, payload: dict[str, object]) -> None:
    lines = [
        "# Four-Dopant SDC Sine Preparation Summary",
        "",
        f"Source root: `{payload['source_root']}`",
        "",
        "## Dataset Checks",
        "",
        "| Check | Value |",
        "|---|---:|",
        f"| Files parsed | {payload['validation']['files']} |",
        f"| Raw rows total | {payload['validation']['raw_rows_total']} |",
        f"| Feature rows after decimation | {payload['validation']['feature_rows_total_after_decimation']} |",
        f"| Conditions match across dopants | {payload['validation']['conditions_match']} |",
        f"| History reset check | {payload['validation']['history_reset_all_ok']} |",
        f"| Min decimated samples/cycle | {payload['validation']['min_decimated_samples_per_cycle']:.2f} |",
        f"| Files below samples/cycle warning | {len(payload['validation']['low_samples_per_cycle_files'])} |",
        "",
        "## Dopant Counts",
        "",
        "| Dopant | Files |",
        "|---|---:|",
    ]
    for dopant, count in payload["validation"]["dopant_counts"].items():
        lines.append(f"| `{dopant}` | {count} |")
    lines.extend(["", "## Generated Splits", "", "| Split | Holdout | Train curves | Val curves | Test curves | Output |", "|---|---|---:|---:|---:|---|"])
    for fold in payload["folds"]:
        counts = fold["split_curve_counts"]
        lines.append(
            f"| `{fold['split_name']}` | `{fold['holdout']}` | {counts['train']} | {counts['val']} | {counts['test']} | `{fold['out']}` |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/raw/candidate_memristor_programming_study/SineWave_measure_results"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--metadata-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--summary-out", type=Path, default=Path("results/sdc4dopant_preparation_summary.md"))
    parser.add_argument("--holdouts", nargs="+", default=["w", "sn", "cr", "c"])
    parser.add_argument("--max-points-per-curve", type=int, default=16384)
    parser.add_argument("--val-per-dopant", type=int, default=3)
    parser.add_argument("--pooled-test-per-dopant", type=int, default=3)
    parser.add_argument("--skip-pooled", action="store_true")
    parser.add_argument("--min-samples-per-cycle-warning", type=float, default=20.0)
    args = parser.parse_args()

    curves = load_curves(args.root, args.max_points_per_curve)
    validation = validate_curves(curves, args.min_samples_per_cycle_warning)
    if validation["files"] != 72:
        raise ValueError(f"expected 72 files, got {validation['files']}")
    if set(validation["dopant_counts"].values()) != {18}:
        raise ValueError(f"expected 18 files per dopant, got {validation['dopant_counts']}")
    if not validation["conditions_match"]:
        raise ValueError("amplitude/frequency conditions do not match across dopants")
    if not validation["history_reset_all_ok"]:
        raise ValueError("history reset validation failed for at least one curve")

    folds = []
    for holdout in args.holdouts:
        holdout = holdout.lower()
        out = args.out_dir / f"sdc4dopant_holdout_{holdout}.npz"
        metadata_out = args.metadata_dir / f"sdc4dopant_holdout_{holdout}_metadata.json"
        folds.append(
            build_fold(
                curves,
                holdout=holdout,
                val_per_dopant=args.val_per_dopant,
                out=out,
                metadata_out=metadata_out,
                source_root=args.root,
                max_points_per_curve=args.max_points_per_curve,
            )
        )
    if not args.skip_pooled:
        pooled_splits = pooled_split_curve_ids(curves, args.val_per_dopant, args.pooled_test_per_dopant)
        folds.append(
            build_cache(
                curves,
                split_name="pooled_in_distribution",
                splits=pooled_splits,
                out=args.out_dir / "sdc4dopant_pooled.npz",
                metadata_out=args.metadata_dir / "sdc4dopant_pooled_metadata.json",
                source_root=args.root,
                max_points_per_curve=args.max_points_per_curve,
                val_per_dopant=args.val_per_dopant,
                test_per_dopant=args.pooled_test_per_dopant,
            )
        )

    payload = {
        "source_root": str(args.root),
        "max_points_per_curve": args.max_points_per_curve,
        "val_per_dopant": args.val_per_dopant,
        "pooled_test_per_dopant": args.pooled_test_per_dopant,
        "validation": validation,
        "folds": folds,
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_summary(args.summary_out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
