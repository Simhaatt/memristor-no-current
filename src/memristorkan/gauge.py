"""Gauge projection and optimizer-state handling for spline KAN layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch
from torch import nn
from torch.optim import Optimizer

from memristorkan.models.kan import KANLayer


@dataclass(frozen=True)
class GaugeProjectionOptions:
    """Options controlling function-preserving KAN canonicalization."""

    renormalize: bool = True
    recenter: bool = False
    eps: float = 1e-8
    moment_mode: str = "unchanged"
    zero_norm_strategy: str = "leave"

    def __post_init__(self) -> None:
        if self.moment_mode not in {"unchanged", "reset"}:
            raise ValueError("moment_mode must be 'unchanged' or 'reset'")
        if self.zero_norm_strategy not in {"leave", "unit_coeff_zero_scale"}:
            raise ValueError("zero_norm_strategy must be 'leave' or 'unit_coeff_zero_scale'")


@dataclass
class GaugeProjectionStats:
    projected_edges: int = 0
    skipped_edges: int = 0
    recentered_edges: int = 0
    reset_state_tensors: int = 0
    layer_stats: list[dict[str, float]] = field(default_factory=list)


@dataclass
class GaugeLayerSnapshot:
    scale_sp: torch.Tensor
    coef: torch.Tensor


def iter_kan_layers(model: nn.Module) -> Iterable[KANLayer]:
    for module in model.modules():
        if isinstance(module, KANLayer):
            yield module


def snapshot_gauge_state(model: nn.Module) -> list[GaugeLayerSnapshot]:
    """Capture spline gauge coordinates before an optimizer step."""

    return [
        GaugeLayerSnapshot(layer.scale_sp.detach().clone(), layer.coef.detach().clone())
        for layer in iter_kan_layers(model)
    ]


@torch.no_grad()
def gauge_project_model(
    model: nn.Module,
    optimizer: Optimizer | None = None,
    options: GaugeProjectionOptions | None = None,
) -> GaugeProjectionStats:
    """Apply Gauge-KAN projection to every KAN layer in ``model``.

    The multiplicative projection is exactly function-preserving for edges with
    ``||c|| > eps``. Near-zero coefficient vectors use the configured epsilon
    strategy.
    """

    options = options or GaugeProjectionOptions()
    stats = GaugeProjectionStats()
    touched_params: list[nn.Parameter] = []

    for layer in iter_kan_layers(model):
        layer_projected = 0
        layer_skipped = 0
        if options.renormalize:
            norms = layer.coef.norm(dim=-1)
            valid = norms > options.eps
            if valid.any():
                layer.scale_sp.data[valid] *= norms[valid]
                layer.coef.data[valid] /= norms[valid].unsqueeze(-1)
                layer_projected = int(valid.sum().item())
                touched_params.extend([layer.scale_sp, layer.coef])
            invalid = ~valid
            if invalid.any():
                layer_skipped = int(invalid.sum().item())
                if options.zero_norm_strategy == "unit_coeff_zero_scale":
                    layer.scale_sp.data[invalid] = 0.0
                    replacement = torch.zeros_like(layer.coef.data[invalid])
                    replacement[:, 0] = 1.0
                    layer.coef.data[invalid] = replacement
                    touched_params.extend([layer.scale_sp, layer.coef])

        recentered = 0
        if options.recenter:
            if layer.output_bias is None:
                raise ValueError("recenter=True requires KANLayer.output_bias to exist")
            dc = layer.coef.mean(dim=-1)
            layer.coef.data -= dc.unsqueeze(-1)
            layer.output_bias.data += (layer.scale_sp.data * dc).sum(dim=0)
            recentered = int(dc.numel())
            touched_params.extend([layer.coef, layer.output_bias])

        stats.projected_edges += layer_projected
        stats.skipped_edges += layer_skipped
        stats.recentered_edges += recentered
        stats.layer_stats.append(
            {
                "projected_edges": float(layer_projected),
                "skipped_edges": float(layer_skipped),
                "recentered_edges": float(recentered),
            }
        )

    if optimizer is not None and options.moment_mode == "reset":
        stats.reset_state_tensors = reset_optimizer_moments(optimizer, touched_params)

    return stats


@torch.no_grad()
def reset_optimizer_moments(optimizer: Optimizer, params: Iterable[nn.Parameter]) -> int:
    """Reset Adam-style moment buffers for projected parameters."""

    unique_params = {id(param): param for param in params}.values()
    reset_count = 0
    for param in unique_params:
        state = optimizer.state.get(param)
        if not state:
            continue
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq", "momentum_buffer"):
            value = state.get(key)
            if torch.is_tensor(value):
                value.zero_()
                reset_count += 1
    return reset_count


@torch.no_grad()
def gauge_update_projection(
    model: nn.Module,
    before: list[GaugeLayerSnapshot],
) -> dict[str, float]:
    """Project actual parameter updates onto the multiplicative gauge tangent.

    For each edge, the infinitesimal gauge direction is ``(w_s, -c)``. The
    returned scalars summarize how much the optimizer update moved along this
    function-preserving direction.
    """

    numerators: list[torch.Tensor] = []
    denominators: list[torch.Tensor] = []
    for layer, snapshot in zip(iter_kan_layers(model), before, strict=True):
        delta_scale = layer.scale_sp.detach() - snapshot.scale_sp
        delta_coef = layer.coef.detach() - snapshot.coef
        tangent_scale = snapshot.scale_sp
        tangent_coef = -snapshot.coef
        numerator = delta_scale * tangent_scale + (delta_coef * tangent_coef).sum(dim=-1)
        denominator = torch.sqrt(tangent_scale.pow(2) + tangent_coef.pow(2).sum(dim=-1)).clamp_min(1e-12)
        numerators.append(numerator)
        denominators.append(denominator)

    if not numerators:
        return {"mean_abs": 0.0, "rms": 0.0, "max_abs": 0.0}
    values = torch.cat([(n / d).reshape(-1) for n, d in zip(numerators, denominators, strict=True)])
    return {
        "mean_abs": float(values.abs().mean().item()),
        "rms": float(torch.sqrt(values.pow(2).mean()).item()),
        "max_abs": float(values.abs().max().item()),
    }


@torch.no_grad()
def adam_moment_gauge_projection(model: nn.Module, optimizer: Optimizer) -> dict[str, float]:
    """Project Adam first moments onto the gauge tangent."""

    values: list[torch.Tensor] = []
    for layer in iter_kan_layers(model):
        scale_state = optimizer.state.get(layer.scale_sp, {})
        coef_state = optimizer.state.get(layer.coef, {})
        scale_moment = scale_state.get("exp_avg")
        coef_moment = coef_state.get("exp_avg")
        if not torch.is_tensor(scale_moment) or not torch.is_tensor(coef_moment):
            continue
        tangent_scale = layer.scale_sp.detach()
        tangent_coef = -layer.coef.detach()
        numerator = scale_moment * tangent_scale + (coef_moment * tangent_coef).sum(dim=-1)
        denominator = torch.sqrt(tangent_scale.pow(2) + tangent_coef.pow(2).sum(dim=-1)).clamp_min(1e-12)
        values.append((numerator / denominator).reshape(-1))

    if not values:
        return {"mean_abs": 0.0, "rms": 0.0, "max_abs": 0.0}
    merged = torch.cat(values)
    return {
        "mean_abs": float(merged.abs().mean().item()),
        "rms": float(torch.sqrt(merged.pow(2).mean()).item()),
        "max_abs": float(merged.abs().max().item()),
    }


def amplitude_penalty(model: nn.Module) -> torch.Tensor:
    """Return the gauge-compatible ``sum w_s^2`` penalty for all KAN layers."""

    penalties = [layer.amplitude_penalty() for layer in iter_kan_layers(model)]
    if not penalties:
        first_param = next(model.parameters(), None)
        if first_param is not None:
            return first_param.new_tensor(0.0)
        return torch.tensor(0.0)
    return torch.stack(penalties).sum()
