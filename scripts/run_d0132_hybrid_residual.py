"""Hybrid compact-model + ML residual correction on D0132.

The model does not replace the compact baseline. It learns a residual:

    log10(R_measured) = log10(R_compact) + residual_ML

Inputs:
    [V_t, V_t_minus_1, dV_t, sweep_direction, log10(R_compact)]

Target:
    log10(R_measured) - log10(R_compact)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from memristorkan.complexity import model_complexity
from memristorkan.train import ModelConfig, build_model


BASE_FEATURE_NAMES = ["V_t", "V_t_minus_1", "dV_t", "sweep_direction"]
FEATURE_NAMES = [*BASE_FEATURE_NAMES, "log10_R_compact"]
CompactKind = Literal["published_model", "VTEAM"]


@dataclass(frozen=True)
class Variant:
    name: str
    model_name: str
    hidden_dim: int
    grid_size: int = 4


VARIANTS = {
    "static_mlp_h200": Variant("static_mlp_h200", "static_mlp_sequence", 200, 4),
    "gru_h116": Variant("gru_h116", "gru", 116, 4),
    "clipped_kan_gru_h32": Variant("clipped_kan_gru_h32", "clipped_kan_gru", 32, 4),
}


@dataclass
class Block:
    device: str
    block_index: int
    voltage: float
    measured_r: np.ndarray
    model_r: np.ndarray
    vteam_r: np.ndarray

    @property
    def label(self) -> str:
        return f"{self.device}_block{self.block_index}_V{self.voltage:g}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_parameter_table(path: Path, device: str) -> list[Block]:
    frame = pd.read_csv(path, sep="\t", encoding="latin-1")
    frame.columns = [str(col).strip() for col in frame.columns]
    voltage = pd.to_numeric(frame["Voltage (V)"], errors="coerce").to_numpy(dtype=float)
    measured = pd.to_numeric(frame[[col for col in frame.columns if col.startswith("Measurements")][0]], errors="coerce").to_numpy(dtype=float)
    model = pd.to_numeric(frame[[col for col in frame.columns if col.startswith("Model")][0]], errors="coerce").to_numpy(dtype=float)
    vteam = pd.to_numeric(frame[[col for col in frame.columns if col.startswith("VTEAM")][0]], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(voltage) & np.isfinite(measured) & np.isfinite(model) & np.isfinite(vteam)
    voltage, measured, model, vteam = voltage[valid], measured[valid], model[valid], vteam[valid]

    blocks: list[Block] = []
    start = 0
    block_index = 0
    for idx in range(1, len(voltage) + 1):
        if idx == len(voltage) or voltage[idx] != voltage[start]:
            end = idx
            blocks.append(
                Block(
                    device=device,
                    block_index=block_index,
                    voltage=float(voltage[start]),
                    measured_r=measured[start:end],
                    model_r=model[start:end],
                    vteam_r=vteam[start:end],
                )
            )
            block_index += 1
            start = idx
    return blocks


def compact_values(block: Block, compact: CompactKind) -> np.ndarray:
    return block.model_r if compact == "published_model" else block.vteam_r


def block_features(block: Block, compact: CompactKind) -> np.ndarray:
    v = np.full_like(block.measured_r, block.voltage, dtype=np.float32)
    v_prev = np.empty_like(v)
    v_prev[0] = v[0]
    v_prev[1:] = v[:-1]
    d_v = v - v_prev
    sweep = np.sign(d_v).astype(np.float32)
    log_compact = np.log10(np.clip(compact_values(block, compact), 1e-12, None)).astype(np.float32)
    return np.stack([v, v_prev, d_v, sweep, log_compact], axis=-1).astype(np.float32)


def residual_target(block: Block, compact: CompactKind) -> np.ndarray:
    measured_log = np.log10(np.clip(block.measured_r, 1e-12, None)).astype(np.float32)
    compact_log = np.log10(np.clip(compact_values(block, compact), 1e-12, None)).astype(np.float32)
    return measured_log - compact_log


def window_starts(length: int, window: int, stride: int) -> list[int]:
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, stride))
    last = length - window
    if starts[-1] != last:
        starts.append(last)
    return starts


class ResidualWindowDataset(Dataset):
    def __init__(
        self,
        blocks: list[Block],
        *,
        compact: CompactKind,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        y_mean: float,
        y_std: float,
        window: int,
        stride: int,
    ) -> None:
        self.blocks = blocks
        self.compact = compact
        self.x_mean = x_mean.astype(np.float32)
        self.x_std = x_std.astype(np.float32)
        self.y_mean = float(y_mean)
        self.y_std = float(y_std)
        self.window = int(window)
        self.items: list[tuple[int, int]] = []
        for block_idx, block in enumerate(blocks):
            for start in window_starts(len(block.measured_r), window, stride):
                self.items.append((block_idx, start))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        block_idx, start = self.items[idx]
        block = self.blocks[block_idx]
        end = start + self.window
        x = (block_features(block, self.compact)[start:end] - self.x_mean) / self.x_std
        y = (residual_target(block, self.compact)[start:end] - self.y_mean) / self.y_std
        return torch.from_numpy(x), torch.from_numpy(y[:, None])


def normalization(blocks: list[Block], compact: CompactKind) -> tuple[np.ndarray, np.ndarray, float, float]:
    x = np.concatenate([block_features(block, compact) for block in blocks], axis=0)
    y = np.concatenate([residual_target(block, compact) for block in blocks], axis=0)
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std[x_std < 1e-6] = 1.0
    y_mean = float(y.mean())
    y_std = float(y.std())
    if y_std < 1e-6:
        y_std = 1.0
    return x_mean, x_std, y_mean, y_std


def build_variant(variant: Variant, input_dim: int) -> nn.Module:
    return build_model(
        ModelConfig(
            name=variant.model_name,  # type: ignore[arg-type]
            input_dim=input_dim,
            output_dim=1,
            hidden_dim=variant.hidden_dim,
            hidden_dims=(variant.hidden_dim,),
            grid_size=variant.grid_size,
            spline_order=3,
        )
    )


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - np.sum(err**2) / denom) if denom > 1e-30 else float("nan")
    log_mask = (y_true > 0) & (y_pred > 0)
    log10_rmse = (
        float(np.sqrt(np.mean((np.log10(y_pred[log_mask]) - np.log10(y_true[log_mask])) ** 2)))
        if np.any(log_mask)
        else float("nan")
    )
    pct = np.abs(y_true) > 1e-30
    mape = float(np.mean(np.abs(err[pct] / y_true[pct])) * 100.0) if np.any(pct) else float("nan")
    return {
        "n": int(y_true.size),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "log10_rmse": log10_rmse,
        "mape_percent": mape,
    }


def reference_metrics(blocks: list[Block], compact: CompactKind, *, window: int, stride: int) -> dict[str, float]:
    truth = []
    pred = []
    for block in blocks:
        compact_r = compact_values(block, compact)
        for start in window_starts(len(block.measured_r), window, stride):
            end = start + window
            truth.append(block.measured_r[start:end])
            pred.append(compact_r[start:end])
    return metrics(np.concatenate(truth), np.concatenate(pred))


def evaluate_residual_model(
    model: nn.Module,
    blocks: list[Block],
    *,
    compact: CompactKind,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: float,
    y_std: float,
    window: int,
    stride: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    dataset = ResidualWindowDataset(
        blocks,
        compact=compact,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        window=window,
        stride=stride,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds = []
    targets = []
    compact_logs = []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred_res_norm = model(x).squeeze(-1).cpu().numpy()
            true_res_norm = y.squeeze(-1).numpy()
            x_raw = x.cpu().numpy() * x_std + x_mean
            log_compact = x_raw[..., 4]
            pred_res = pred_res_norm * y_std + y_mean
            true_res = true_res_norm * y_std + y_mean
            preds.append(np.power(10.0, log_compact + pred_res).reshape(-1))
            targets.append(np.power(10.0, log_compact + true_res).reshape(-1))
            compact_logs.append(log_compact.reshape(-1))
    return metrics(np.concatenate(targets), np.concatenate(preds))


def split_protocol(blocks_dut1: list[Block], blocks_dut2: list[Block], protocol: str) -> tuple[list[Block], list[Block], list[Block]]:
    if protocol == "device_transfer":
        return blocks_dut1[:6], blocks_dut1[6:], blocks_dut2
    if protocol == "within_d0132":
        return blocks_dut1 + blocks_dut2[:4], [blocks_dut2[4]], [blocks_dut2[5]]
    raise ValueError(f"unknown protocol: {protocol}")


def train_one(
    *,
    variant: Variant,
    compact: CompactKind,
    seed: int,
    train_blocks: list[Block],
    val_blocks: list[Block],
    test_blocks: list[Block],
    epochs: int,
    lr: float,
    batch_size: int,
    window: int,
    train_stride: int,
    eval_stride: int,
    device: torch.device,
) -> dict[str, Any]:
    set_seed(seed)
    x_mean, x_std, y_mean, y_std = normalization(train_blocks, compact)
    train_ds = ResidualWindowDataset(
        train_blocks,
        compact=compact,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        window=window,
        stride=train_stride,
    )
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    model = build_variant(variant, input_dim=len(FEATURE_NAMES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    losses: list[float] = []
    start = time.perf_counter()
    for _epoch in range(epochs):
        model.train()
        epoch_losses = []
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        losses.append(float(mean(epoch_losses)))
    elapsed = time.perf_counter() - start
    val_metrics = evaluate_residual_model(
        model,
        val_blocks,
        compact=compact,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        window=window,
        stride=eval_stride,
        batch_size=batch_size,
        device=device,
    )
    test_metrics = evaluate_residual_model(
        model,
        test_blocks,
        compact=compact,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        window=window,
        stride=eval_stride,
        batch_size=batch_size,
        device=device,
    )
    return {
        "variant": variant.name,
        "compact": compact,
        "seed": seed,
        "epochs": epochs,
        "train_loss_last": losses[-1],
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "complexity": model_complexity(model),
        "timing": {
            "wall_clock_seconds": elapsed,
            "seconds_per_epoch": elapsed / max(1, epochs),
        },
        "train_windows": len(train_ds),
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["protocol"]), str(row["compact"]), str(row["model"])), []).append(row)
    summary = []
    for (protocol, compact, model), items in sorted(grouped.items()):
        def values(field: str) -> list[float]:
            return [float(item[field]) for item in items]

        summary.append(
            {
                "protocol": protocol,
                "compact": compact,
                "model": model,
                "runs": len(items),
                "n": int(items[0]["n"]),
                "r2_mean": mean(values("r2")),
                "r2_std": pstdev(values("r2")) if len(items) > 1 else 0.0,
                "rmse_mean": mean(values("rmse")),
                "mae_mean": mean(values("mae")),
                "log10_rmse_mean": mean(values("log10_rmse")),
                "mape_percent_mean": mean(values("mape_percent")),
                "seconds_per_epoch_mean": mean(values("seconds_per_epoch")) if "seconds_per_epoch" in items[0] else 0.0,
                "params": int(items[0].get("params", 0)),
            }
        )
    summary.sort(key=lambda row: (row["protocol"], row["compact"], -float(row["r2_mean"])))
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    numeric = float(value)
    if not np.isfinite(numeric):
        return "nan"
    if abs(numeric) < 1e-3 and numeric != 0:
        return f"{numeric:.3e}"
    return f"{numeric:.4f}"


def write_markdown(path: Path, summary: list[dict[str, Any]]) -> None:
    lines = [
        "# D0132 Hybrid Compact-ML Residual Results",
        "",
        "Hybrid equation:",
        "",
        "```text",
        "log10(R_pred) = log10(R_compact) + ML_residual(V_t, V_t_minus_1, dV_t, sweep_direction, log10(R_compact))",
        "```",
        "",
        "## Summary",
        "",
        "| Protocol | Compact prior | Model | Runs | N | R2 | RMSE ohm | MAE ohm | Log10 RMSE | MAPE % | sec/epoch | Params |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| `{row['protocol']}` | `{row['compact']}` | `{row['model']}` | {row['runs']} | {row['n']} | "
            f"{fmt(row['r2_mean'])} +/- {fmt(row['r2_std'])} | {fmt(row['rmse_mean'])} | "
            f"{fmt(row['mae_mean'])} | {fmt(row['log10_rmse_mean'])} | {fmt(row['mape_percent_mean'])} | "
            f"{fmt(row['seconds_per_epoch_mean'])} | {row['params']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This tests whether ML can improve an already calibrated compact/VTEAM prediction by learning the residual. A positive row must beat the corresponding `compact_only` row under the same protocol and compact prior.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def make_plot(out_dir: Path, summary: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    for protocol in sorted({row["protocol"] for row in summary}):
        sub = [row for row in summary if row["protocol"] == protocol]
        labels = [f"{row['compact']}\n{row['model']}" for row in sub]
        values = [float(row["r2_mean"]) for row in sub]
        errors = [float(row["r2_std"]) for row in sub]
        fig, ax = plt.subplots(figsize=(max(9, 0.8 * len(labels)), 4.8))
        ax.bar(range(len(labels)), values, yerr=errors, color="#4C78A8")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=40, ha="right")
        ax.set_ylabel("Test R2")
        ax.set_title(f"D0132 compact+ML residual: {protocol}")
        fig.tight_layout()
        fig.savefig(out_dir / f"d0132_hybrid_residual_{protocol}_r2.png", dpi=180)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/southampton/D0132/data_sets_extracted"))
    parser.add_argument("--variants", nargs="+", default=["static_mlp_h200", "gru_h116", "clipped_kan_gru_h32"], choices=list(VARIANTS))
    parser.add_argument("--compact-priors", nargs="+", default=["published_model", "VTEAM"], choices=["published_model", "VTEAM"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--train-stride", type=int, default=128)
    parser.add_argument("--eval-stride", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", type=Path, default=Path("results/d0132_hybrid_residual"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    blocks_dut1 = read_parameter_table(args.root / "DUT_1_Characterization__Parameter Extraction.txt", "DUT_1")
    blocks_dut2 = read_parameter_table(args.root / "DUT_2_Characterization__Parameter_Extraction.txt", "DUT_2")
    flat_rows: list[dict[str, Any]] = []
    run_payloads: list[dict[str, Any]] = []

    for protocol in ["device_transfer", "within_d0132"]:
        train_blocks, val_blocks, test_blocks = split_protocol(blocks_dut1, blocks_dut2, protocol)
        for compact in args.compact_priors:
            compact_kind: CompactKind = compact  # type: ignore[assignment]
            compact_metrics = reference_metrics(test_blocks, compact_kind, window=args.window, stride=args.eval_stride)
            flat_rows.append(
                {
                    "protocol": protocol,
                    "compact": compact,
                    "model": "compact_only",
                    "seed": -1,
                    "params": 0,
                    "seconds_per_epoch": 0.0,
                    **compact_metrics,
                }
            )
            for variant_name in args.variants:
                variant = VARIANTS[variant_name]
                for seed in args.seeds:
                    result = train_one(
                        variant=variant,
                        compact=compact_kind,
                        seed=seed,
                        train_blocks=train_blocks,
                        val_blocks=val_blocks,
                        test_blocks=test_blocks,
                        epochs=args.epochs,
                        lr=args.lr,
                        batch_size=args.batch_size,
                        window=args.window,
                        train_stride=args.train_stride,
                        eval_stride=args.eval_stride,
                        device=device,
                    )
                    run_payloads.append({"protocol": protocol, **result})
                    flat_rows.append(
                        {
                            "protocol": protocol,
                            "compact": compact,
                            "model": variant.name,
                            "seed": seed,
                            "params": result["complexity"]["total_parameters"],
                            "seconds_per_epoch": result["timing"]["seconds_per_epoch"],
                            **result["test_metrics"],
                        }
                    )
                    print(
                        json.dumps(
                            {
                                "protocol": protocol,
                                "compact": compact,
                                "variant": variant.name,
                                "seed": seed,
                                "test": result["test_metrics"],
                            }
                        )
                    )

    summary = summarize(flat_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "d0132_hybrid_residual_per_run.csv", flat_rows)
    write_csv(args.out_dir / "d0132_hybrid_residual_summary.csv", summary)
    write_markdown(args.out_dir / "d0132_hybrid_residual_results.md", summary)
    make_plot(args.out_dir, summary)
    (args.out_dir / "d0132_hybrid_residual_results.json").write_text(
        json.dumps(
            {
                "feature_names": FEATURE_NAMES,
                "target": "log10_measured_resistance_minus_log10_compact_resistance",
                "window": args.window,
                "train_stride": args.train_stride,
                "eval_stride": args.eval_stride,
                "flat_rows": flat_rows,
                "summary": summary,
                "runs": run_payloads,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"out_dir": str(args.out_dir), "summary_rows": len(summary), "runs": len(flat_rows)}, indent=2))


if __name__ == "__main__":
    main()
