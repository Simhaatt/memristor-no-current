"""Run no-current ML models on D0132 pulse resistance sequences.

Task:
    [V_t, V_t_minus_1, dV_t, sweep_direction] -> R_t

The target is log10 resistance during training, then predictions are converted
back to ohms for reporting. Two protocols are reported:

* ``device_transfer``: train on DUT1 pulse blocks, test on all DUT2 blocks.
* ``within_d0132``: train on DUT1 plus early DUT2 blocks, test on held-out
  DUT2 voltage blocks.

Both are compared against the supplied D0132 published-model and VTEAM columns.
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
from typing import Any

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


FEATURE_NAMES = ["V_t", "V_t_minus_1", "dV_t", "sweep_direction"]


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
    pulse_number: np.ndarray
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
    pulse = pd.to_numeric(frame["Pulse Number"], errors="coerce").to_numpy(dtype=float)
    voltage = pd.to_numeric(frame["Voltage (V)"], errors="coerce").to_numpy(dtype=float)
    measured = pd.to_numeric(frame[[col for col in frame.columns if col.startswith("Measurements")][0]], errors="coerce").to_numpy(dtype=float)
    model = pd.to_numeric(frame[[col for col in frame.columns if col.startswith("Model")][0]], errors="coerce").to_numpy(dtype=float)
    vteam = pd.to_numeric(frame[[col for col in frame.columns if col.startswith("VTEAM")][0]], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(voltage) & np.isfinite(measured) & np.isfinite(model) & np.isfinite(vteam)
    pulse, voltage, measured, model, vteam = pulse[valid], voltage[valid], measured[valid], model[valid], vteam[valid]

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
                    pulse_number=pulse[start:end],
                    measured_r=measured[start:end],
                    model_r=model[start:end],
                    vteam_r=vteam[start:end],
                )
            )
            block_index += 1
            start = idx
    return blocks


def block_features(block: Block) -> np.ndarray:
    v = np.full_like(block.measured_r, block.voltage, dtype=np.float32)
    v_prev = np.empty_like(v)
    v_prev[0] = v[0]
    v_prev[1:] = v[:-1]
    d_v = v - v_prev
    sweep = np.sign(d_v).astype(np.float32)
    return np.stack([v, v_prev, d_v, sweep], axis=-1).astype(np.float32)


class BlockWindowDataset(Dataset):
    def __init__(
        self,
        blocks: list[Block],
        *,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        y_mean: float,
        y_std: float,
        window: int,
        stride: int,
    ) -> None:
        self.blocks = blocks
        self.x_mean = x_mean.astype(np.float32)
        self.x_std = x_std.astype(np.float32)
        self.y_mean = float(y_mean)
        self.y_std = float(y_std)
        self.window = int(window)
        self.items: list[tuple[int, int]] = []
        for block_idx, block in enumerate(blocks):
            length = len(block.measured_r)
            if length <= self.window:
                self.items.append((block_idx, 0))
                continue
            for start in range(0, length - self.window + 1, stride):
                self.items.append((block_idx, start))
            last = length - self.window
            if self.items[-1] != (block_idx, last):
                self.items.append((block_idx, last))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        block_idx, start = self.items[idx]
        block = self.blocks[block_idx]
        end = start + self.window
        x = (block_features(block)[start:end] - self.x_mean) / self.x_std
        y = (np.log10(block.measured_r[start:end]).astype(np.float32) - self.y_mean) / self.y_std
        return torch.from_numpy(x), torch.from_numpy(y[:, None])


def normalization(blocks: list[Block]) -> tuple[np.ndarray, np.ndarray, float, float]:
    x = np.concatenate([block_features(block) for block in blocks], axis=0)
    y = np.concatenate([np.log10(block.measured_r).astype(np.float32) for block in blocks], axis=0)
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
    log_rmse = (
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
        "log10_rmse": log_rmse,
        "mape_percent": mape,
    }


def evaluate_ml(
    model: nn.Module,
    blocks: list[Block],
    *,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: float,
    y_std: float,
    window: int,
    stride: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    dataset = BlockWindowDataset(
        blocks,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        window=window,
        stride=stride,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred_norm = model(x).squeeze(-1).cpu().numpy()
            true_norm = y.squeeze(-1).numpy()
            preds.append(np.power(10.0, pred_norm * y_std + y_mean).reshape(-1))
            targets.append(np.power(10.0, true_norm * y_std + y_mean).reshape(-1))
    return metrics(np.concatenate(targets), np.concatenate(preds))


def window_starts(length: int, window: int, stride: int) -> list[int]:
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, stride))
    last = length - window
    if starts[-1] != last:
        starts.append(last)
    return starts


def evaluate_reference(blocks: list[Block], attr: str, *, window: int, stride: int) -> tuple[dict[str, float], list[dict[str, Any]]]:
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    per_block: list[dict[str, Any]] = []
    for block in blocks:
        true_parts = []
        pred_parts = []
        for start in window_starts(len(block.measured_r), window, stride):
            end = start + window
            true_parts.append(block.measured_r[start:end])
            pred_parts.append(getattr(block, attr)[start:end])
        true_r = np.concatenate(true_parts)
        pred_r = np.concatenate(pred_parts)
        row = {
            "block": block.label,
            "device": block.device,
            "voltage": block.voltage,
        }
        row.update(metrics(true_r, pred_r))
        per_block.append(row)
        all_true.append(true_r)
        all_pred.append(pred_r)
    return metrics(np.concatenate(all_true), np.concatenate(all_pred)), per_block


def split_protocol(blocks_dut1: list[Block], blocks_dut2: list[Block], protocol: str) -> tuple[list[Block], list[Block], list[Block]]:
    if protocol == "device_transfer":
        train = blocks_dut1[:6]
        val = blocks_dut1[6:]
        test = blocks_dut2
        return train, val, test
    if protocol == "within_d0132":
        train = blocks_dut1 + blocks_dut2[:4]
        val = [blocks_dut2[4]]
        test = [blocks_dut2[5]]
        return train, val, test
    raise ValueError(f"unknown protocol: {protocol}")


def train_one(
    *,
    variant: Variant,
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
    x_mean, x_std, y_mean, y_std = normalization(train_blocks)
    train_ds = BlockWindowDataset(
        train_blocks,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        window=window,
        stride=train_stride,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    model = build_variant(variant, input_dim=len(FEATURE_NAMES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    start = time.perf_counter()
    epoch_losses: list[float] = []
    for _epoch in range(epochs):
        model.train()
        losses = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        epoch_losses.append(float(mean(losses)))
    elapsed = time.perf_counter() - start
    val_metrics = evaluate_ml(
        model,
        val_blocks,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        window=window,
        stride=eval_stride,
        batch_size=batch_size,
        device=device,
    )
    test_metrics = evaluate_ml(
        model,
        test_blocks,
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
        "seed": seed,
        "epochs": epochs,
        "train_loss_last": epoch_losses[-1],
        "train_loss_mean": float(mean(epoch_losses)),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "complexity": model_complexity(model),
        "timing": {
            "wall_clock_seconds": elapsed,
            "seconds_per_epoch": elapsed / max(1, epochs),
        },
        "normalization": {
            "x_mean": x_mean.tolist(),
            "x_std": x_std.tolist(),
            "y_mean_log10": y_mean,
            "y_std_log10": y_std,
        },
        "train_windows": len(train_ds),
    }


def summarize(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in results:
        grouped.setdefault((str(row["protocol"]), str(row["model"])), []).append(row)
    out: list[dict[str, Any]] = []
    for (protocol, model), rows in sorted(grouped.items()):
        def values(field: str) -> list[float]:
            return [float(row[field]) for row in rows]

        out.append(
            {
                "protocol": protocol,
                "model": model,
                "runs": len(rows),
                "n": int(rows[0]["n"]),
                "r2_mean": mean(values("r2")),
                "r2_std": pstdev(values("r2")) if len(rows) > 1 else 0.0,
                "rmse_mean": mean(values("rmse")),
                "mae_mean": mean(values("mae")),
                "log10_rmse_mean": mean(values("log10_rmse")),
                "mape_percent_mean": mean(values("mape_percent")),
                "seconds_per_epoch_mean": mean(values("seconds_per_epoch")) if "seconds_per_epoch" in rows[0] else 0.0,
                "params": int(rows[0].get("params", 0)),
            }
        )
    out.sort(key=lambda row: (row["protocol"], -float(row["r2_mean"])))
    return out


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


def write_markdown(path: Path, summary: list[dict[str, Any]], flat: list[dict[str, Any]]) -> None:
    lines = [
        "# D0132 Resistance-Sequence ML Benchmark",
        "",
        "Task:",
        "",
        "```text",
        "[V_t, V_t_minus_1, dV_t, sweep_direction] -> R_t",
        "```",
        "",
        "Training target is normalized `log10(R_t)`. Reported metrics are computed after converting predictions back to ohms.",
        "",
        "## Protocols",
        "",
        "| Protocol | Train | Validation | Test | Meaning |",
        "|---|---|---|---|---|",
        "| `device_transfer` | DUT1 blocks 0-5 | DUT1 blocks 6-7 | all DUT2 blocks | strict transfer to a different DUT/resistance range |",
        "| `within_d0132` | all DUT1 + DUT2 blocks 0-3 | DUT2 block 4 | DUT2 block 5 | in-dataset held-out voltage-block check |",
        "",
        "All ML models are trained/evaluated on fixed windows from the pulse blocks, matching the windowed sequence style used in the SDC experiments.",
        "",
        "## Summary",
        "",
        "| Protocol | Model | Runs | N | R2 | RMSE ohm | MAE ohm | Log10 RMSE | MAPE % | sec/epoch | Params |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| `{row['protocol']}` | `{row['model']}` | {row['runs']} | {row['n']} | "
            f"{fmt(row['r2_mean'])} +/- {fmt(row['r2_std'])} | {fmt(row['rmse_mean'])} | "
            f"{fmt(row['mae_mean'])} | {fmt(row['log10_rmse_mean'])} | "
            f"{fmt(row['mape_percent_mean'])} | {fmt(row['seconds_per_epoch_mean'])} | {row['params']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The `device_transfer` protocol is intentionally harsh because DUT2 has a substantially different resistance range from DUT1 and the no-current feature set does not include an initial resistance/device identifier. Poor ML transfer here should be framed as a device-transfer boundary, not a failure of the main SDC current-prediction result.",
            "",
            "The `within_d0132` protocol is the fairer ML sanity check: the model sees the D0132 resistance range during training and is tested on a held-out DUT2 voltage block.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/southampton/D0132/data_sets_extracted"))
    parser.add_argument("--variants", nargs="+", default=["static_mlp_h200", "gru_h116", "clipped_kan_gru_h32"], choices=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--train-stride", type=int, default=32)
    parser.add_argument("--eval-stride", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", type=Path, default=Path("results/d0132_resistance_sequence"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    blocks_dut1 = read_parameter_table(args.root / "DUT_1_Characterization__Parameter Extraction.txt", "DUT_1")
    blocks_dut2 = read_parameter_table(args.root / "DUT_2_Characterization__Parameter_Extraction.txt", "DUT_2")
    all_results: list[dict[str, Any]] = []
    flat_rows: list[dict[str, Any]] = []

    for protocol in ["device_transfer", "within_d0132"]:
        train_blocks, val_blocks, test_blocks = split_protocol(blocks_dut1, blocks_dut2, protocol)
        for ref_name, attr in [("published_model", "model_r"), ("VTEAM", "vteam_r")]:
            ref_metrics, _ref_per_block = evaluate_reference(
                test_blocks,
                attr,
                window=args.window,
                stride=args.eval_stride,
            )
            row = {
                "protocol": protocol,
                "model": ref_name,
                "seed": -1,
                "params": 0,
                "seconds_per_epoch": 0.0,
                **ref_metrics,
            }
            flat_rows.append(row)

        for variant_name in args.variants:
            variant = VARIANTS[variant_name]
            for seed in args.seeds:
                result = train_one(
                    variant=variant,
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
                all_results.append({"protocol": protocol, **result})
                test = result["test_metrics"]
                flat_rows.append(
                    {
                        "protocol": protocol,
                        "model": variant.name,
                        "seed": seed,
                        "params": result["complexity"]["total_parameters"],
                        "seconds_per_epoch": result["timing"]["seconds_per_epoch"],
                        **test,
                    }
                )
                print(json.dumps({"protocol": protocol, "variant": variant.name, "seed": seed, "test": test}))

    summary = summarize(flat_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "d0132_resistance_sequence_per_run.csv", flat_rows)
    write_csv(args.out_dir / "d0132_resistance_sequence_summary.csv", summary)
    write_markdown(args.out_dir / "d0132_resistance_sequence_results.md", summary, flat_rows)
    (args.out_dir / "d0132_resistance_sequence_results.json").write_text(
        json.dumps(
            {
                "feature_names": FEATURE_NAMES,
                "target": "log10_resistance_for_training_raw_ohm_for_metrics",
                "dut1_blocks": [block.label for block in blocks_dut1],
                "dut2_blocks": [block.label for block in blocks_dut2],
                "window": args.window,
                "train_stride": args.train_stride,
                "eval_stride": args.eval_stride,
                "runs": all_results,
                "flat_rows": flat_rows,
                "summary": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"out_dir": str(args.out_dir), "summary_rows": len(summary), "runs": len(flat_rows)}, indent=2))


if __name__ == "__main__":
    main()
