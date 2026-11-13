"""Loaders for the public SDC memristor state-characterisation repository."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset


class SDCDataError(RuntimeError):
    """Raised when SDC CSV columns cannot be inferred safely."""


@dataclass(frozen=True)
class DatasetSummary:
    name: str
    sequences: int
    rows: int
    columns: tuple[str, ...]
    split_type: str
    notes: str


def _resolve_subdir(root: str | Path, preferred: str) -> Path:
    path = Path(root)
    if path.name == preferred:
        return path
    candidate = path / "data" / preferred
    if candidate.exists():
        return candidate
    candidate = path / preferred
    if candidate.exists():
        return candidate
    return path


def _csv_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.csv") if path.is_file())


def _read_numeric_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=1)
    numeric = df.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(axis=1, how="all").dropna(axis=0, how="all")
    if numeric.empty:
        raise SDCDataError(f"no numeric data found in {path}")
    return numeric.reset_index(drop=True)


def _infer_voltage_current(df: pd.DataFrame, path: Path) -> tuple[str, str]:
    if "Volt" in df.columns and "Volt.1" in df.columns:
        return "Volt.1-Volt", "Volt"
    lower = {str(col).lower(): col for col in df.columns}
    voltage_candidates = [
        original
        for key, original in lower.items()
        if key in {"v", "voltage"} or "voltage" in key or key.endswith("_v")
    ]
    current_candidates = [
        original
        for key, original in lower.items()
        if key in {"i", "current"} or "current" in key or key.endswith("_i")
    ]
    if not voltage_candidates or not current_candidates:
        non_time_cols = [str(col) for col in df.columns if str(col).lower() not in {"second", "time", "x-axis"}]
        if len(non_time_cols) >= 2:
            return non_time_cols[1], non_time_cols[0]
        if df.shape[1] >= 3:
            return str(df.columns[2]), str(df.columns[1])
        raise SDCDataError(f"could not infer voltage/current columns in {path}")
    return str(voltage_candidates[0]), str(current_candidates[0])


def _voltage_current_arrays(df: pd.DataFrame, path: Path) -> tuple[np.ndarray, np.ndarray]:
    v_col, i_col = _infer_voltage_current(df, path)
    if v_col == "Volt.1-Volt":
        voltage = (df["Volt.1"] - df["Volt"]).to_numpy(dtype=np.float32)
    else:
        voltage = df[v_col].to_numpy(dtype=np.float32)
    current = df[i_col].to_numpy(dtype=np.float32)
    return voltage, current


def load_sdc_state_table(root: str | Path) -> tuple[TensorDataset, DatasetSummary]:
    """Load SDC triangle-wave V-I data as Track A history features.

    Features are ``[V_t, V_{t-1}, I_{t-1}, dV_t, sweep_direction]`` and targets
    are ``I_t``.
    """

    data_root = _resolve_subdir(root, "state_data")
    files = _csv_files(data_root)
    if not files:
        raise SDCDataError(f"no CSV files found under {data_root}")

    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    columns: set[str] = set()
    for path in files:
        df = _read_numeric_csv(path)
        v, i = _voltage_current_arrays(df, path)
        if len(v) < 2:
            continue
        d_v = np.diff(v, prepend=v[0])
        sweep_direction = np.sign(d_v).astype(np.float32)
        x = np.stack([v[1:], v[:-1], i[:-1], d_v[1:], sweep_direction[1:]], axis=-1)
        y = i[1:, None]
        features.append(x)
        targets.append(y)
        columns.update(map(str, df.columns))

    if not features:
        raise SDCDataError("state CSV files did not contain enough rows")

    x_tensor = torch.from_numpy(np.concatenate(features, axis=0))
    y_tensor = torch.from_numpy(np.concatenate(targets, axis=0))
    summary = DatasetSummary(
        name="SDC state data",
        sequences=len(files),
        rows=int(x_tensor.shape[0]),
        columns=tuple(sorted(columns)),
        split_type="curve-wise",
        notes="Triangle-wave V-I features with measured one-step history.",
    )
    return TensorDataset(x_tensor, y_tensor), summary


def load_sdc_state_curves(root: str | Path) -> tuple[list[dict[str, object]], DatasetSummary]:
    """Load SDC V-I curves separately for curve-wise splitting."""

    data_root = _resolve_subdir(root, "state_data")
    files = _csv_files(data_root)
    if not files:
        raise SDCDataError(f"no CSV files found under {data_root}")

    curves: list[dict[str, object]] = []
    columns: set[str] = set()
    for path in files:
        df = _read_numeric_csv(path)
        v, i = _voltage_current_arrays(df, path)
        if len(v) < 2:
            continue
        d_v = np.diff(v, prepend=v[0])
        sweep_direction = np.sign(d_v).astype(np.float32)
        x = np.stack([v[1:], v[:-1], i[:-1], d_v[1:], sweep_direction[1:]], axis=-1)
        y = i[1:, None]
        curves.append({"path": str(path), "x": x.astype(np.float32), "y": y.astype(np.float32)})
        columns.update(map(str, df.columns))

    rows = int(sum(curve["x"].shape[0] for curve in curves))
    summary = DatasetSummary(
        name="SDC state data",
        sequences=len(curves),
        rows=rows,
        columns=tuple(sorted(columns)),
        split_type="curve-wise",
        notes="Voltage is Volt.1 - Volt and current is Volt, matching the source repository preprocessing.",
    )
    return curves, summary


def _resistance_from_read(df: pd.DataFrame, path: Path) -> float:
    voltage, current = _voltage_current_arrays(df, path)
    voltage = voltage.astype(np.float64)
    current = current.astype(np.float64)
    valid = np.abs(current) > 1e-15
    if not valid.any():
        raise SDCDataError(f"read has no non-zero current values: {path}")
    resistance = np.abs(voltage[valid] / current[valid])
    resistance = resistance[np.isfinite(resistance)]
    if resistance.size == 0:
        raise SDCDataError(f"read has no finite resistance values: {path}")
    return float(np.median(resistance))


def load_sdc_retention_sequences(
    root: str | Path,
    *,
    crop_first_half: bool = True,
) -> tuple[TensorDataset, DatasetSummary]:
    """Load SDC retention reads as sequence samples.

    The public README states that some second-half read samples may be missing
    or corrupted; ``crop_first_half=True`` is therefore the default and should
    stay enabled for the planned experiments.
    """

    data_root = _resolve_subdir(root, "retention_data")
    if not data_root.exists():
        raise SDCDataError(f"retention data root not found: {data_root}")

    experiment_dirs = sorted(path for path in data_root.iterdir() if path.is_dir())
    if not experiment_dirs:
        experiment_dirs = [data_root]

    x_sequences: list[np.ndarray] = []
    y_sequences: list[np.ndarray] = []
    seen_columns: set[str] = set()
    for exp_dir in experiment_dirs:
        files = _csv_files(exp_dir)
        if len(files) < 2:
            continue
        resistances: list[float] = []
        for path in files[1:]:
            df = _read_numeric_csv(path)
            seen_columns.update(map(str, df.columns))
            if crop_first_half:
                df = df.iloc[: max(1, len(df) // 2)].reset_index(drop=True)
            resistances.append(_resistance_from_read(df, path))
        if len(resistances) < 2:
            continue
        r = np.asarray(resistances, dtype=np.float32)
        log_r = np.log(r)
        read_index = np.arange(len(r), dtype=np.float32)
        t_seconds = read_index * 60.0
        dt = np.diff(t_seconds, prepend=t_seconds[0])
        previous = np.concatenate([[log_r[0]], log_r[:-1]]).astype(np.float32)
        r0 = np.full_like(previous, log_r[0])
        features = np.stack([t_seconds, r0, previous, dt, read_index], axis=-1).astype(np.float32)
        x_sequences.append(features)
        y_sequences.append(log_r[:, None].astype(np.float32))

    if not x_sequences:
        raise SDCDataError("no usable retention sequences found")

    min_len = min(seq.shape[0] for seq in x_sequences)
    x = torch.from_numpy(np.stack([seq[:min_len] for seq in x_sequences], axis=0))
    y = torch.from_numpy(np.stack([seq[:min_len] for seq in y_sequences], axis=0))
    summary = DatasetSummary(
        name="SDC retention data",
        sequences=len(x_sequences),
        rows=int(x.numel() // x.shape[-1]),
        columns=tuple(sorted(seen_columns)),
        split_type="sequence-wise only",
        notes="Retention proof-of-concept; read CSVs cropped to first half by default.",
    )
    return TensorDataset(x, y), summary
