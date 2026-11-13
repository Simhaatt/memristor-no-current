"""Evaluate Southampton D0132 compact-model/VTEAM baseline tables.

The D0132 archive already contains measured data alongside outputs from the
published compact model and VTEAM fits. This script treats those supplied model
outputs as reference baselines and computes comparable error metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return {
            "n": 0,
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
            "mape_percent": float("nan"),
            "log10_rmse": float("nan"),
            "nrmse_range": float("nan"),
        }
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - np.sum(err**2) / denom) if denom > 1e-30 else float("nan")
    pct_mask = np.abs(y_true) > 1e-30
    mape = float(np.mean(np.abs(err[pct_mask] / y_true[pct_mask])) * 100.0) if np.any(pct_mask) else float("nan")
    log_mask = (y_true > 0) & (y_pred > 0)
    log_rmse = (
        float(np.sqrt(np.mean((np.log10(y_pred[log_mask]) - np.log10(y_true[log_mask])) ** 2)))
        if np.any(log_mask)
        else float("nan")
    )
    value_range = float(np.max(y_true) - np.min(y_true))
    nrmse_range = rmse / value_range if value_range > 1e-30 else float("nan")
    return {
        "n": int(y_true.size),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape_percent": mape,
        "log10_rmse": log_rmse,
        "nrmse_range": nrmse_range,
    }


def find_column(columns: list[str], prefix: str) -> str:
    matches = [col for col in columns if col.lower().startswith(prefix.lower())]
    if not matches:
        raise ValueError(f"could not find column starting with {prefix!r} in {columns}")
    return matches[0]


def read_simple_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", encoding="latin-1", engine="python")


def read_multi_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", encoding="latin-1", engine="python", header=[0, 1])


def add_row(
    rows: list[dict[str, Any]],
    *,
    dataset: str,
    task: str,
    device: str,
    condition: str,
    target: str,
    baseline: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    source_file: Path,
) -> None:
    row = {
        "dataset": dataset,
        "task": task,
        "device": device,
        "condition": condition,
        "target": target,
        "baseline": baseline,
        "source_file": str(source_file),
    }
    row.update(metrics(y_true, y_pred))
    rows.append(row)


def evaluate_parameter_extraction(root: Path) -> list[dict[str, Any]]:
    specs = [
        ("DUT_1", root / "DUT_1_Characterization__Parameter Extraction.txt"),
        ("DUT_2", root / "DUT_2_Characterization__Parameter_Extraction.txt"),
    ]
    rows: list[dict[str, Any]] = []
    for device, path in specs:
        frame = read_simple_table(path)
        columns = [str(col) for col in frame.columns]
        y_true = numeric(frame[find_column(columns, "Measurements")])
        for baseline, prefix in [("published_model", "Model"), ("VTEAM", "VTEAM")]:
            y_pred = numeric(frame[find_column(columns, prefix)])
            add_row(
                rows,
                dataset="D0132",
                task="pulse_parameter_extraction",
                device=device,
                condition="all_pulses",
                target="resistance_ohm",
                baseline=baseline,
                y_true=y_true,
                y_pred=y_pred,
                source_file=path,
            )
    return rows


def multi_time_column(frame: pd.DataFrame) -> tuple[Any, ...]:
    return frame.columns[1] if str(frame.columns[0][0]).lower().startswith("pulse") else frame.columns[0]


def resistance_columns(frame: pd.DataFrame) -> list[tuple[Any, ...]]:
    return [col for col in frame.columns if str(col[0]).lower().startswith("resistance")]


def evaluate_time_response(root: Path) -> list[dict[str, Any]]:
    specs = [
        (
            "DUT_3",
            "SET",
            root / "DUT_3__SET_Characterization_Data.txt",
            root / "DUT_3__SET_Model_Fit.txt",
            root / "DUT_3__SET_VTEAM_Fit.txt",
        ),
        (
            "DUT_3",
            "RESET",
            root / "DUT_3__RESET_Characterization_Data.txt",
            root / "DUT_3__RESET_Model_Fit.txt",
            root / "DUT_3__RESET_VTEAM_Fit.txt",
        ),
        (
            "DUT_4",
            "SET",
            root / "DUT_4__SET_Characterization_Data.txt",
            root / "DUT_4__SET_Model_Fit.txt",
            root / "DUT_4__SET_VTEAM_Fit.txt",
        ),
        (
            "DUT_4",
            "RESET",
            root / "DUT_4__RESET_Characterization_Data.txt",
            root / "DUT_4__RESET_Model_Fit.txt",
            root / "DUT_4__RESET_VTEAM_Fit.txt",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for device, transition, meas_path, model_path, vteam_path in specs:
        measured = read_multi_table(meas_path)
        model = read_multi_table(model_path)
        vteam = read_multi_table(vteam_path)
        t_meas = numeric(measured[multi_time_column(measured)])
        t_model = numeric(model[multi_time_column(model)])
        t_vteam = numeric(vteam[multi_time_column(vteam)])
        for col in resistance_columns(measured):
            condition = str(col[1]).strip()
            if col not in model.columns or col not in vteam.columns:
                continue
            y_true = numeric(measured[col])
            y_model_dense = numeric(model[col])
            y_vteam_dense = numeric(vteam[col])
            valid_model = np.isfinite(t_model) & np.isfinite(y_model_dense)
            valid_vteam = np.isfinite(t_vteam) & np.isfinite(y_vteam_dense)
            y_model = np.interp(t_meas, t_model[valid_model], y_model_dense[valid_model])
            y_vteam = np.interp(t_meas, t_vteam[valid_vteam], y_vteam_dense[valid_vteam])
            for baseline, y_pred, source in [
                ("published_model", y_model, model_path),
                ("VTEAM", y_vteam, vteam_path),
            ]:
                add_row(
                    rows,
                    dataset="D0132",
                    task=f"{transition.lower()}_time_response",
                    device=device,
                    condition=condition,
                    target="resistance_ohm",
                    baseline=baseline,
                    y_true=y_true,
                    y_pred=y_pred,
                    source_file=source,
                )
    return rows


def paired_iv_columns(frame: pd.DataFrame) -> list[tuple[int, int, str]]:
    columns = list(frame.columns)
    pairs: list[tuple[int, int, str]] = []
    if len(columns) >= 2 and str(columns[0]).lower().startswith("voltage"):
        if all(str(columns[idx]).lower().startswith("current") for idx in range(1, len(columns))):
            for idx in range(1, len(columns)):
                pairs.append((0, idx, str(columns[idx])))
        else:
            for idx in range(0, len(columns) - 1, 2):
                if str(columns[idx]).lower().startswith("voltage") and str(columns[idx + 1]).lower().startswith("current"):
                    pairs.append((idx, idx + 1, f"curve_{len(pairs) + 1}"))
    return pairs


def evaluate_iv_curves(root: Path) -> list[dict[str, Any]]:
    specs = [
        (
            "TaOx",
            root / "IV_measurements__TaOx.txt",
            root / "IV_model__TaOx.txt",
            "current_mA",
        ),
        (
            "TiOx",
            root / "IV_measurements__TiOx.txt",
            root / "IV_model__TiOx.txt",
            "current_A",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for device, meas_path, model_path, target in specs:
        measured = read_simple_table(meas_path)
        model = read_simple_table(model_path)
        meas_pairs = paired_iv_columns(measured)
        model_pairs = paired_iv_columns(model)
        for curve_idx, (v_col_idx, i_col_idx, condition) in enumerate(meas_pairs):
            if curve_idx >= len(model_pairs):
                continue
            mv_idx, mi_idx, _ = model_pairs[curve_idx]
            v_meas = numeric(measured.iloc[:, v_col_idx])
            i_true = numeric(measured.iloc[:, i_col_idx])
            v_model = numeric(model.iloc[:, mv_idx])
            i_model_dense = numeric(model.iloc[:, mi_idx])
            valid = np.isfinite(v_model) & np.isfinite(i_model_dense)
            order = np.argsort(v_model[valid])
            x = v_model[valid][order]
            y = i_model_dense[valid][order]
            unique_x, unique_indices = np.unique(x, return_index=True)
            unique_y = y[unique_indices]
            mask = np.isfinite(v_meas) & np.isfinite(i_true)
            if unique_x.size < 2 or not np.any(mask):
                continue
            i_model = np.interp(v_meas[mask], unique_x, unique_y)
            add_row(
                rows,
                dataset="D0132",
                task="iv_curve",
                device=device,
                condition=condition,
                target=target,
                baseline="published_model",
                y_true=i_true[mask],
                y_pred=i_model,
                source_file=model_path,
            )
    return rows


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["task"], row["target"], row["baseline"]), []).append(row)
    out: list[dict[str, Any]] = []
    for (task, target, baseline), items in sorted(grouped.items()):
        total_n = sum(int(item["n"]) for item in items)
        # Weight pointwise metrics by n; R2 is summarized as simple mean because
        # rows may represent different devices/conditions.
        def weighted(field: str) -> float:
            vals = [(float(item[field]), int(item["n"])) for item in items if np.isfinite(float(item[field]))]
            denom = sum(n for _value, n in vals)
            return sum(value * n for value, n in vals) / denom if denom else float("nan")

        out.append(
            {
                "task": task,
                "target": target,
                "baseline": baseline,
                "rows": len(items),
                "n": total_n,
                "rmse_weighted": weighted("rmse"),
                "mae_weighted": weighted("mae"),
                "mape_percent_weighted": weighted("mape_percent"),
                "log10_rmse_weighted": weighted("log10_rmse"),
                "nrmse_range_weighted": weighted("nrmse_range"),
                "r2_mean": float(np.nanmean([float(item["r2"]) for item in items])),
                "r2_min": float(np.nanmin([float(item["r2"]) for item in items])),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 4) -> str:
    try:
        numeric_value = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(numeric_value):
        return "nan"
    if abs(numeric_value) < 1e-3 and numeric_value != 0:
        return f"{numeric_value:.3e}"
    return f"{numeric_value:.{digits}f}"


def write_markdown(path: Path, rows: list[dict[str, Any]], summary: list[dict[str, Any]]) -> None:
    lines = [
        "# D0132 Compact-Model Baseline",
        "",
        "Dataset: Southampton D0132, `data_sets.zip`, associated with the practical Verilog-A ReRAM model.",
        "",
        "This baseline uses the model outputs already supplied in the D0132 archive:",
        "",
        "- `Model` / published Verilog-A model columns",
        "- `VTEAM` fit columns",
        "- measured resistance/current columns",
        "",
        "No ML training is performed here. The goal is to contextualize ML results against a compact-model reference for device-reviewer credibility.",
        "",
        "## Aggregate Summary",
        "",
        "| Task | Target | Baseline | Rows | N | RMSE | MAE | MAPE % | Log10 RMSE | NRMSE range | Mean R2 | Min R2 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| `{row['task']}` | `{row['target']}` | `{row['baseline']}` | "
            f"{row['rows']} | {row['n']} | {fmt(row['rmse_weighted'])} | "
            f"{fmt(row['mae_weighted'])} | {fmt(row['mape_percent_weighted'])} | "
            f"{fmt(row['log10_rmse_weighted'])} | {fmt(row['nrmse_range_weighted'])} | "
            f"{fmt(row['r2_mean'])} | {fmt(row['r2_min'])} |"
        )

    lines.extend(
        [
            "",
            "## Per-Condition Results",
            "",
            "| Task | Device | Condition | Target | Baseline | N | RMSE | MAE | MAPE % | Log10 RMSE | R2 |",
            "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(rows, key=lambda item: (item["task"], item["device"], item["condition"], item["baseline"])):
        lines.append(
            f"| `{row['task']}` | `{row['device']}` | `{row['condition']}` | `{row['target']}` | "
            f"`{row['baseline']}` | {row['n']} | {fmt(row['rmse'])} | {fmt(row['mae'])} | "
            f"{fmt(row['mape_percent'])} | {fmt(row['log10_rmse'])} | {fmt(row['r2'])} |"
        )

    lines.extend(
        [
            "",
            "## Paper Use",
            "",
            "Use this as a compact-model reference/baseline section, not as a direct competitor to the SDC ML models because the dataset and target are different.",
            "",
            "Safe manuscript wording:",
            "",
            "```text",
            "To contextualize the ML sequence models against established device modelling practice, we also evaluated the published Southampton D0132 compact-model and VTEAM fit outputs against their paired measured resistance/current data. This comparison is not an apples-to-apples SDC benchmark, but it anchors the ML results against a recognized compact-model reference.",
            "```",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def make_plot(path: Path, summary: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    labels = [f"{row['task']}\n{row['baseline']}" for row in summary]
    values = [float(row["nrmse_range_weighted"]) for row in summary]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.3), 4.6))
    ax.bar(range(len(values)), values, color="#4C78A8")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Weighted NRMSE / measured range")
    ax.set_title("D0132 compact-model/VTEAM baseline errors")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/southampton/D0132/data_sets_extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/compact_model_baselines/d0132"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    rows.extend(evaluate_parameter_extraction(args.root))
    rows.extend(evaluate_time_response(args.root))
    rows.extend(evaluate_iv_curves(args.root))
    summary = aggregate(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "d0132_compact_baseline_per_condition.csv", rows)
    write_csv(args.out_dir / "d0132_compact_baseline_summary.csv", summary)
    write_markdown(args.out_dir / "d0132_compact_model_baseline.md", rows, summary)
    make_plot(args.out_dir / "d0132_compact_baseline_nrmse.png", summary)
    (args.out_dir / "d0132_compact_baseline.json").write_text(
        json.dumps({"rows": rows, "summary": summary}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"out_dir": str(args.out_dir), "rows": len(rows), "summary_rows": len(summary)}, indent=2))


if __name__ == "__main__":
    main()
