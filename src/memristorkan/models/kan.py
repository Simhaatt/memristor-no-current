"""KAN and MLP models used by the memristor experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from memristorkan.models.spline import bspline_basis, make_clamped_uniform_knots


@dataclass(frozen=True)
class KANLayerConfig:
    input_dim: int
    output_dim: int
    grid_size: int = 8
    spline_order: int = 3
    grid_range: tuple[float, float] = (-1.0, 1.0)
    train_base_scale: bool = True
    train_spline_scale: bool = True
    bias: bool = True


class KANLayer(nn.Module):
    """A fixed-grid spline KAN layer with explicit spline scale parameters.

    Each edge function is

    ``phi_ij(x_i) = w_b,ij * SiLU(x_i) + w_s,ij * sum_k c_ijk B_k(x_i)``.

    The pair ``(w_s, c)`` is intentionally represented separately so the
    Gauge-KAN projection can canonicalize their multiplicative redundancy.
    """

    def __init__(self, config: KANLayerConfig) -> None:
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim
        self.grid_size = config.grid_size
        self.spline_order = config.spline_order
        knots = make_clamped_uniform_knots(
            config.grid_size,
            config.spline_order,
            config.grid_range,
        )
        self.register_buffer("knots", knots)
        n_basis = knots.numel() - config.spline_order - 1
        scale = 1.0 / max(1, config.input_dim) ** 0.5
        self.coef = nn.Parameter(torch.randn(config.input_dim, config.output_dim, n_basis) * 0.05)
        self.scale_sp = nn.Parameter(torch.ones(config.input_dim, config.output_dim) * scale)
        self.scale_sp.requires_grad_(config.train_spline_scale)
        self.scale_base = nn.Parameter(torch.empty(config.input_dim, config.output_dim))
        self.scale_base.requires_grad_(config.train_base_scale)
        nn.init.uniform_(self.scale_base, -scale, scale)
        if config.bias:
            self.output_bias = nn.Parameter(torch.zeros(config.output_dim))
        else:
            self.register_parameter("output_bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[-1] != self.input_dim:
            raise ValueError(f"expected input shape (batch, {self.input_dim}), got {tuple(x.shape)}")
        basis = bspline_basis(x, self.knots, self.spline_order)
        spline = torch.einsum("bik,iok->bio", basis, self.coef)
        base = F.silu(x)
        edges = self.scale_base.unsqueeze(0) * base.unsqueeze(-1)
        edges = edges + self.scale_sp.unsqueeze(0) * spline
        y = edges.sum(dim=1)
        if self.output_bias is not None:
            y = y + self.output_bias
        return y

    def amplitude_penalty(self) -> torch.Tensor:
        """Gauge-compatible amplitude penalty for represented spline magnitude."""

        return self.scale_sp.pow(2).sum()


class KANRegressor(nn.Module):
    """Stacked KAN regressor for point/history-feature experiments."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | tuple[int, ...],
        output_dim: int = 1,
        *,
        grid_size: int = 8,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        super().__init__()
        dims = [input_dim, *hidden_dims, output_dim]
        self.layers = nn.ModuleList(
            [
                KANLayer(
                    KANLayerConfig(
                        dims[i],
                        dims[i + 1],
                        grid_size=grid_size,
                        spline_order=spline_order,
                        grid_range=grid_range,
                    )
                )
                for i in range(len(dims) - 1)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def amplitude_penalty(self) -> torch.Tensor:
        penalties = [layer.amplitude_penalty() for layer in self.layers]
        return torch.stack(penalties).sum() if penalties else torch.tensor(0.0)


class MLPRegressor(nn.Module):
    """Plain MLP baseline for Track A."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | tuple[int, ...],
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        dims = [input_dim, *hidden_dims, output_dim]
        modules: list[nn.Module] = []
        for i in range(len(dims) - 1):
            modules.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                modules.append(nn.SiLU())
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
