"""Rollout evaluation utilities for history-feature memristor models."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from memristorkan.metrics import regression_metrics


@torch.no_grad()
def rollout_sdc_state(
    model: nn.Module,
    curve_x_raw: np.ndarray,
    curve_y_raw: np.ndarray,
    curve_mask: np.ndarray,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    *,
    device: str = "cpu",
) -> dict[str, float]:
    """Evaluate SDC V-I rollout using predicted previous current.

    Feature order is ``[V_t, V_{t-1}, I_{t-1}, dV_t, sweep_direction]``.
    The first step of each curve is seeded with the measured ``I_{t-1}``
    already present in the cached raw feature array. Subsequent steps feed back
    the previous model prediction.
    """

    model.eval()
    model.to(device)
    x_mean_t = torch.as_tensor(x_mean, dtype=torch.float32, device=device)
    x_std_t = torch.as_tensor(x_std, dtype=torch.float32, device=device)
    y_mean_t = torch.as_tensor(y_mean, dtype=torch.float32, device=device)
    y_std_t = torch.as_tensor(y_std, dtype=torch.float32, device=device)

    x_raw_t = torch.as_tensor(curve_x_raw, dtype=torch.float32, device=device)
    y_raw_t = torch.as_tensor(curve_y_raw, dtype=torch.float32, device=device)
    mask_t = torch.as_tensor(curve_mask, dtype=torch.bool, device=device)
    predictions = torch.zeros_like(y_raw_t)
    prev_pred = x_raw_t[:, 0, 2].clone()

    for t in range(x_raw_t.shape[1]):
        active = mask_t[:, t]
        if not active.any():
            continue
        x_step = x_raw_t[active, t, :].clone()
        if t > 0:
            x_step[:, 2] = prev_pred[active]
        x_norm = (x_step - x_mean_t) / x_std_t
        y_norm = model(x_norm)
        y_pred = y_norm * y_std_t + y_mean_t
        predictions[active, t, :] = y_pred
        prev_pred[active] = y_pred[:, 0]

    pred = predictions[mask_t].detach().cpu()
    target = y_raw_t[mask_t].detach().cpu()
    metrics = regression_metrics(pred, target)
    lengths = curve_mask.sum(axis=1)
    final_errors = []
    predictions_cpu = predictions.detach().cpu().numpy()
    for curve_idx, length in enumerate(lengths):
        if length > 0:
            final_errors.append(abs(float(predictions_cpu[curve_idx, length - 1, 0] - curve_y_raw[curve_idx, length - 1, 0])))
    metrics["final_state_mae"] = float(np.mean(final_errors)) if final_errors else 0.0
    return {f"rollout_{key}": value for key, value in metrics.items()}
