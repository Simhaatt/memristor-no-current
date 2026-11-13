"""Export manuscript plotting CSVs from saved memristor checkpoints.

Outputs are inference-oriented trace files for:

* SDC full-trajectory I-V loops with measured and predicted current,
* time-series overlays/residual-ready columns,
* clipped KAN-GRU hidden-state PCA trajectories,
* D0132 DUT2 compact/residual resistance traces,
* pooled voltage/error points for the SDC no-current task.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
SCRIPT_DIR = ROOT / "scripts"
for path in (SRC_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from memristorkan.train import ModelConfig, build_model
from run_d0132_hybrid_residual import (  # type: ignore
    VARIANTS as D0132_VARIANTS,
    ResidualWindowDataset,
    compact_values,
    normalization as d0132_normalization,
    read_parameter_table,
    residual_target,
    split_protocol,
)
from run_sdc_sequence import build_feature_array, build_sequence_model  # type: ignore
from evaluate_sequence_checkpoints import load_checkpoint, variant_from_checkpoint
from memristorkan.models.recurrent import ClippedKANGRURegressor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def predict_sequence_checkpoint(
    checkpoint: Path,
    x: np.ndarray,
    *,
    device: torch.device,
) -> tuple[np.ndarray, nn.Module, dict[str, Any]]:
    payload = load_checkpoint(checkpoint, device)
    variant = variant_from_checkpoint(payload)
    model = build_sequence_model(variant, input_dim=x.shape[-1]).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(x[None, :, :]).to(device)).squeeze(0).squeeze(-1).cpu().numpy()
    return pred, model, payload


def clipped_kangru_hidden_states(model: ClippedKANGRURegressor, x: np.ndarray, device: torch.device) -> np.ndarray:
    xt = torch.from_numpy(x[None, :, :]).to(device)
    states: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        h = xt.new_zeros(1, model.hidden_dim)
        for t in range(xt.shape[1]):
            x_t = xt[:, t, :]
            gate_input = torch.cat([x_t, h], dim=-1)
            z_raw, r_raw = model.cell.gates(gate_input).chunk(2, dim=-1)
            z = ((z_raw + 1.0) * 0.5).clamp(0.0, 1.0)
            r = ((r_raw + 1.0) * 0.5).clamp(0.0, 1.0)
            candidate_input = torch.cat([x_t, r * h], dim=-1)
            n = model.cell.candidate(candidate_input).clamp(-1.0, 1.0)
            h = (1.0 - z) * n + z * h
            states.append(h.squeeze(0).cpu().numpy())
    return np.stack(states, axis=0)


def pca2(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def export_sdc_traces(
    *,
    cache: Path,
    kan_checkpoint: Path,
    gru_checkpoint: Path,
    mlp_checkpoint: Path,
    out_dir: Path,
    curve_indices: list[int],
    device: torch.device,
) -> dict[str, Any]:
    data = np.load(cache)
    lengths = data["test_curve_lengths"].astype(np.int32)
    x_curve = data["test_curve_x"].astype(np.float32)
    y_curve = data["test_curve_y"].astype(np.float32)
    x_raw = data["test_curve_x_raw"].astype(np.float32)
    y_raw = data["test_curve_y_raw"].astype(np.float32)
    x_no_current = build_feature_array(x_curve, lengths, "no_current")
    y_mean = float(data["y_mean"][0, 0])
    y_std = float(data["y_std"][0, 0])

    trace_dir = out_dir / "sdc_traces"
    hidden_dir = out_dir / "hidden_state"
    pooled_error: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    for curve_idx in curve_indices:
        length = int(lengths[curve_idx])
        x = x_no_current[curve_idx, :length]
        voltage = x_raw[curve_idx, :length, 0]
        measured_i = y_raw[curve_idx, :length, 0]

        pred_kan_norm, kan_model, _kan_payload = predict_sequence_checkpoint(kan_checkpoint, x, device=device)
        pred_gru_norm, _gru_model, _gru_payload = predict_sequence_checkpoint(gru_checkpoint, x, device=device)
        pred_mlp_norm, _mlp_model, _mlp_payload = predict_sequence_checkpoint(mlp_checkpoint, x, device=device)

        pred_kan = pred_kan_norm * y_std + y_mean
        pred_gru = pred_gru_norm * y_std + y_mean
        pred_mlp = pred_mlp_norm * y_std + y_mean

        rows = []
        for sample in range(length):
            rows.append(
                {
                    "sample": sample,
                    "V": float(voltage[sample]),
                    "I_measured": float(measured_i[sample]),
                    "I_predicted": float(pred_kan[sample]),
                    "I_pred_gru": float(pred_gru[sample]),
                    "I_pred_mlp": float(pred_mlp[sample]),
                    "residual_kan": float(pred_kan[sample] - measured_i[sample]),
                    "residual_gru": float(pred_gru[sample] - measured_i[sample]),
                    "residual_mlp": float(pred_mlp[sample] - measured_i[sample]),
                }
            )
            pooled_error.append(
                {
                    "curve_idx": curve_idx,
                    "sample": sample,
                    "V": float(voltage[sample]),
                    "abs_error": float(abs(pred_kan[sample] - measured_i[sample])),
                    "abs_error_gru": float(abs(pred_gru[sample] - measured_i[sample])),
                    "abs_error_mlp": float(abs(pred_mlp[sample] - measured_i[sample])),
                }
            )
        write_csv(trace_dir / f"sdc_test_curve_{curve_idx:02d}_iv_timeseries_predictions.csv", rows)

        if not isinstance(kan_model, ClippedKANGRURegressor):
            raise TypeError("expected clipped KAN-GRU for hidden-state export")
        hidden = clipped_kangru_hidden_states(kan_model, x, device)
        pcs = pca2(hidden)
        hidden_rows = [
            {
                "sample": sample,
                "V": float(voltage[sample]),
                "I": float(measured_i[sample]),
                "h_pc1": float(pcs[sample, 0]),
                "h_pc2": float(pcs[sample, 1]),
            }
            for sample in range(length)
        ]
        write_csv(hidden_dir / f"sdc_test_curve_{curve_idx:02d}_hidden_state_pc.csv", hidden_rows)
        np.save(hidden_dir / f"sdc_test_curve_{curve_idx:02d}_hidden_state_32d.npy", hidden.astype(np.float32))
        manifest_rows.append(
            {
                "curve_idx": curve_idx,
                "length": length,
                "trace_csv": f"sdc_traces/sdc_test_curve_{curve_idx:02d}_iv_timeseries_predictions.csv",
                "hidden_pc_csv": f"hidden_state/sdc_test_curve_{curve_idx:02d}_hidden_state_pc.csv",
                "hidden_32d_npy": f"hidden_state/sdc_test_curve_{curve_idx:02d}_hidden_state_32d.npy",
            }
        )

    write_csv(out_dir / "sdc_voltage_abs_error_pooled.csv", pooled_error)
    return {
        "selected_test_curves": manifest_rows,
        "target_inverse_transform": {"y_raw": "y_norm * y_std + y_mean", "y_mean": y_mean, "y_std": y_std},
    }


def build_d0132_model(variant_name: str, input_dim: int) -> nn.Module:
    variant = D0132_VARIANTS[variant_name]
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


def train_d0132_residual_model(
    *,
    model: nn.Module,
    train_blocks: list[Any],
    compact: str,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: float,
    y_std: float,
    epochs: int,
    batch_size: int,
    window: int,
    stride: int,
    lr: float,
    device: torch.device,
) -> None:
    dataset = ResidualWindowDataset(
        train_blocks,
        compact=compact,  # type: ignore[arg-type]
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        window=window,
        stride=stride,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for _epoch in range(epochs):
        model.train()
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


def predict_d0132_residual_full_block(
    *,
    model: nn.Module,
    block: Any,
    compact: str,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: float,
    y_std: float,
    device: torch.device,
) -> np.ndarray:
    from run_d0132_hybrid_residual import block_features  # type: ignore

    x = block_features(block, compact)  # type: ignore[arg-type]
    x_norm = (x - x_mean) / x_std
    model.eval()
    with torch.no_grad():
        pred_res_norm = model(torch.from_numpy(x_norm[None, :, :]).to(device)).squeeze(0).squeeze(-1).cpu().numpy()
    pred_res = pred_res_norm * y_std + y_mean
    compact_log = np.log10(np.clip(compact_values(block, compact), 1e-12, None))  # type: ignore[arg-type]
    return np.power(10.0, compact_log + pred_res)


def export_d0132_trace(
    *,
    root: Path,
    out_dir: Path,
    compact: str,
    variant: str,
    seed: int,
    epochs: int,
    device: torch.device,
) -> dict[str, Any]:
    set_seed(seed)
    blocks_dut1 = read_parameter_table(root / "DUT_1_Characterization__Parameter Extraction.txt", "DUT_1")
    blocks_dut2 = read_parameter_table(root / "DUT_2_Characterization__Parameter_Extraction.txt", "DUT_2")
    train_blocks, _val_blocks, test_blocks = split_protocol(blocks_dut1, blocks_dut2, "device_transfer")
    x_mean, x_std, y_mean, y_std = d0132_normalization(train_blocks, compact)  # type: ignore[arg-type]
    model = build_d0132_model(variant, input_dim=5)
    start = time.perf_counter()
    train_d0132_residual_model(
        model=model,
        train_blocks=train_blocks,
        compact=compact,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        epochs=epochs,
        batch_size=8,
        window=128,
        stride=128,
        lr=1e-3,
        device=device,
    )
    seconds = time.perf_counter() - start

    rows_all: list[dict[str, Any]] = []
    trace_dir = out_dir / "d0132"
    for block in test_blocks:
        corrected = predict_d0132_residual_full_block(
            model=model,
            block=block,
            compact=compact,
            x_mean=x_mean,
            x_std=x_std,
            y_mean=y_mean,
            y_std=y_std,
            device=device,
        )
        compact_r = compact_values(block, compact)  # type: ignore[arg-type]
        rows = []
        for sample in range(len(block.measured_r)):
            row = {
                "block": block.label,
                "sample": sample,
                "pulse_number": float(sample),
                "voltage": float(block.voltage),
                "R_measured": float(block.measured_r[sample]),
                "R_compact": float(compact_r[sample]),
                "R_residual_corrected": float(corrected[sample]),
            }
            rows.append(row)
            rows_all.append(row)
        write_csv(trace_dir / f"d0132_{block.label}_compact_residual_trace.csv", rows)
    write_csv(out_dir / "d0132_dut2_all_blocks_compact_residual_trace.csv", rows_all)
    return {
        "protocol": "device_transfer",
        "compact": compact,
        "residual_variant": variant,
        "seed": seed,
        "epochs_regenerated_for_trace": epochs,
        "seconds": seconds,
        "note": "D0132 pointwise residual traces are regenerated because earlier aggregate D0132 result files did not store per-sample predictions.",
    }


def make_zip(out_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    files = sorted(path for path in out_dir.rglob("*") if path.is_file())
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, path.relative_to(out_dir).as_posix())
    with zipfile.ZipFile(zip_path) as zf:
        bad = [name for name in zf.namelist() if "\\" in name]
        if bad:
            raise RuntimeError(f"bad archive paths: {bad[:3]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("data/processed/sdc_state_splits.npz"))
    parser.add_argument("--kan-checkpoint", type=Path, default=Path("results/kaggle_imported/kangru_sdc_100epoch_seed0/checkpoints/sequence/clipped_kan_gru_h32_no_current_seed0.pt"))
    parser.add_argument("--gru-checkpoint", type=Path, default=Path("checkpoints/kaggle_imported/no_current_baselines_100epoch_3seed/sequence/gru_h116_no_current_seed0.pt"))
    parser.add_argument("--mlp-checkpoint", type=Path, default=Path("checkpoints/sequence/static_mlp_h200_no_current_seed0.pt"))
    parser.add_argument("--curve-indices", type=int, nargs="+", default=[0, 4, 8])
    parser.add_argument("--d0132-root", type=Path, default=Path("data/raw/southampton/D0132/data_sets_extracted"))
    parser.add_argument("--d0132-epochs", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", type=Path, default=Path("results/manuscript_trace_csvs"))
    parser.add_argument("--zip", type=Path, default=Path("results/manuscript_trace_csvs.zip"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    sdc_manifest = export_sdc_traces(
        cache=args.cache,
        kan_checkpoint=args.kan_checkpoint,
        gru_checkpoint=args.gru_checkpoint,
        mlp_checkpoint=args.mlp_checkpoint,
        out_dir=args.out_dir,
        curve_indices=args.curve_indices,
        device=device,
    )
    d0132_manifest = export_d0132_trace(
        root=args.d0132_root,
        out_dir=args.out_dir,
        compact="VTEAM",
        variant="clipped_kan_gru_h32",
        seed=0,
        epochs=args.d0132_epochs,
        device=device,
    )
    manifest = {
        "sdc": sdc_manifest,
        "d0132": d0132_manifest,
        "files": [str(path.relative_to(args.out_dir).as_posix()) for path in sorted(args.out_dir.rglob("*")) if path.is_file()],
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    make_zip(args.out_dir, args.zip)
    print(json.dumps({"out_dir": str(args.out_dir), "zip": str(args.zip), "files": len(manifest["files"])}, indent=2))


if __name__ == "__main__":
    main()
