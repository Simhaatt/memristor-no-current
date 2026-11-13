"""Loaders for Zenodo record 18619371 external V-I checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


class ZenodoDataError(RuntimeError):
    """Raised when the Zenodo external data cannot be read safely."""


@dataclass(frozen=True)
class ZenodoTrace:
    name: str
    source_path: str
    voltage: np.ndarray
    current: np.ndarray
    time: np.ndarray | None
    notes: str


@dataclass(frozen=True)
class ZenodoSummary:
    traces: int
    rows: int
    notes: str


def _dataset_root(root: str | Path) -> Path:
    path = Path(root)
    if (path / "Figure 3").exists():
        return path
    candidate = path / "extracted" / "Dataset"
    if (candidate / "Figure 3").exists():
        return candidate
    candidate = path / "Dataset"
    if (candidate / "Figure 3").exists():
        return candidate
    raise ZenodoDataError(f"could not locate extracted Zenodo Dataset under {path}")


def _read_three_column_iv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path, sep="\t")
    df.columns = [str(col).replace("\ufeff", "") for col in df.columns]
    if df.shape[1] < 3:
        raise ZenodoDataError(f"expected at least 3 columns in {path}")
    time = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=np.float32)
    voltage = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(dtype=np.float32)
    current = pd.to_numeric(df.iloc[:, 2], errors="coerce").to_numpy(dtype=np.float32)
    valid = np.isfinite(time) & np.isfinite(voltage) & np.isfinite(current)
    if valid.sum() < 3:
        raise ZenodoDataError(f"not enough finite V-I rows in {path}")
    return time[valid], voltage[valid], current[valid]


def _read_one_column(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size < 3:
        raise ZenodoDataError(f"not enough finite rows in {path}")
    return values


def load_zenodo_vi_traces(
    root: str | Path,
    *,
    include_phase_traces: bool = True,
) -> tuple[list[ZenodoTrace], ZenodoSummary]:
    """Load Zenodo 18619371 traces as voltage/current sequences.

    ``iv-sol-mem-2v`` is the clean standalone memristor I-V check. Figure 2/4
    phase traces are larger coupled-system checks and are cropped to the common
    voltage/current length, matching the provided MATLAB plotting scripts.
    """

    base = _dataset_root(root)
    traces: list[ZenodoTrace] = []

    mem_path = base / "Figure 3" / "iv-sol-mem-2v"
    if mem_path.exists():
        time, voltage, current = _read_three_column_iv(mem_path)
        traces.append(
            ZenodoTrace(
                name="fig3_memristor_iv_2v",
                source_path=str(mem_path),
                voltage=voltage,
                current=current,
                time=time,
                notes="Standalone memristor I-V file; small qualitative external check.",
            )
        )

    if include_phase_traces:
        pair_specs = [
            ("Figure 2", "pt5", "0p5hz"),
            ("Figure 2", "1", "1hz"),
            ("Figure 2", "2", "2hz"),
            ("Figure 4", "pt125", "0p125hz"),
            ("Figure 4", "pt5", "0p5hz"),
        ]
        for fig_dir, token, label in pair_specs:
            folder = base / fig_dir
            if not folder.exists():
                continue
            current_candidates = sorted(folder.glob(f"output_phi-vs-i-{token}hz.txt"))
            if not current_candidates and token.startswith("pt"):
                current_candidates = sorted(folder.glob(f"output_phi-vs-i-{token}Hz.txt"))
            voltage_candidates = sorted(folder.glob(f"output_V_phi-vs-V-{token}Hz.txt"))
            if not voltage_candidates and token in {"1", "2"}:
                voltage_candidates = sorted(folder.glob(f"output_V_phi-vs-V-{token}hz.txt"))
            if not current_candidates or not voltage_candidates:
                continue
            current = _read_one_column(current_candidates[0])
            voltage = _read_one_column(voltage_candidates[0])
            length = min(len(voltage), len(current))
            traces.append(
                ZenodoTrace(
                    name=f"{fig_dir.lower().replace(' ', '')}_phase_{label}",
                    source_path=f"{voltage_candidates[0]} | {current_candidates[0]}",
                    voltage=voltage[:length],
                    current=current[:length],
                    time=None,
                    notes="Coupled solar-cell/memristor phase trace; domain-shift check.",
                )
            )

    if not traces:
        raise ZenodoDataError(f"no usable V-I traces found under {base}")
    summary = ZenodoSummary(
        traces=len(traces),
        rows=int(sum(len(trace.current) for trace in traces)),
        notes="Zenodo 18619371 external qualitative V-I/domain-shift traces.",
    )
    return traces, summary


def trace_to_sdc_base_features(trace: ZenodoTrace) -> tuple[np.ndarray, np.ndarray]:
    """Convert one trace to raw SDC base features and target.

    Base feature order is ``[V_t, V_t_minus_1, I_t_minus_1, dV_t,
    sweep_direction]`` and target is ``I_t``.
    """

    voltage = np.asarray(trace.voltage, dtype=np.float32).reshape(-1)
    current = np.asarray(trace.current, dtype=np.float32).reshape(-1)
    length = min(len(voltage), len(current))
    voltage = voltage[:length]
    current = current[:length]
    if length < 2:
        raise ZenodoDataError(f"trace {trace.name} has fewer than 2 V-I rows")
    d_v = np.diff(voltage, prepend=voltage[0]).astype(np.float32)
    sweep_direction = np.sign(d_v).astype(np.float32)
    x = np.stack(
        [voltage[1:], voltage[:-1], current[:-1], d_v[1:], sweep_direction[1:]],
        axis=-1,
    ).astype(np.float32)
    y = current[1:, None].astype(np.float32)
    return x, y
