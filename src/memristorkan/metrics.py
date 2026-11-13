"""Metrics used across point, sequence, and rollout experiments."""

from __future__ import annotations

import torch


def regression_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred = pred.detach().reshape(-1).float()
    target = target.detach().reshape(-1).float()
    error = pred - target
    mse = error.pow(2).mean()
    mae = error.abs().mean()
    denom = (target - target.mean()).pow(2).sum().clamp_min(1e-12)
    r2 = 1.0 - error.pow(2).sum() / denom
    return {
        "rmse": float(torch.sqrt(mse).item()),
        "mae": float(mae.item()),
        "r2": float(r2.item()),
    }


def sequence_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    metrics = regression_metrics(pred, target)
    final_error = (pred[:, -1] - target[:, -1]).abs().mean()
    metrics["final_state_mae"] = float(final_error.item())
    metrics["normalized_horizon_rmse"] = metrics["rmse"]
    return metrics


def drift_slope_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean absolute linear-slope error over normalized sequence horizons."""

    if pred.ndim != 3 or target.ndim != 3:
        raise ValueError("drift_slope_error expects tensors shaped (batch, seq, features)")
    seq_len = pred.shape[1]
    tau = torch.linspace(0.0, 1.0, seq_len, device=pred.device, dtype=pred.dtype)
    tau_centered = tau - tau.mean()
    denom = tau_centered.pow(2).sum().clamp_min(1e-12)
    pred_centered = pred[..., 0] - pred[..., 0].mean(dim=1, keepdim=True)
    target_centered = target[..., 0] - target[..., 0].mean(dim=1, keepdim=True)
    pred_slope = (pred_centered * tau_centered).sum(dim=1) / denom
    target_slope = (target_centered * tau_centered).sum(dim=1) / denom
    return float((pred_slope - target_slope).abs().mean().item())
