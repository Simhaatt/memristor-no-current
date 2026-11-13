"""Training loop for Gauge-KAN memristor experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Literal

import torch
from torch import nn
from torch.optim import AdamW, LBFGS, SGD, Optimizer
from torch.utils.data import DataLoader

from memristorkan.gauge import (
    GaugeProjectionOptions,
    adam_moment_gauge_projection,
    amplitude_penalty,
    gauge_project_model,
    gauge_update_projection,
    snapshot_gauge_state,
)
from memristorkan.metrics import regression_metrics, sequence_metrics
from memristorkan.models import (
    BoundedKANLSTMRegressor,
    ClippedKANGRURegressor,
    ClippedKANLSTMRegressor,
    FusedClippedKANGRURegressor,
    GRURegressor,
    HybridClippedKANGRURegressor,
    KANGRURegressor,
    KANLayer,
    KANLSTMRegressor,
    KANRegressor,
    LSTMRegressor,
    MLPRegressor,
    StaticMLPSequenceRegressor,
)


ModelName = Literal[
    "mlp",
    "static_mlp_sequence",
    "kan",
    "gauge_kan",
    "lstm",
    "gru",
    "kan_gru",
    "clipped_kan_gru",
    "fused_clipped_kan_gru",
    "hybrid_clipped_kan_gru",
    "kan_lstm",
    "gauge_kan_lstm",
    "bounded_kan_lstm",
    "gauge_bounded_kan_lstm",
    "clipped_kan_lstm",
    "gauge_clipped_kan_lstm",
]


@dataclass(frozen=True)
class ModelConfig:
    name: ModelName
    input_dim: int
    output_dim: int = 1
    hidden_dims: tuple[int, ...] = (32,)
    hidden_dim: int = 32
    grid_size: int = 8
    spline_order: int = 3


@dataclass(frozen=True)
class TrainingConfig:
    steps: int = 200
    batch_size: int = 128
    lr: float = 1e-3
    optimizer: Literal["adamw", "sgd", "lbfgs"] = "adamw"
    weight_decay: float = 0.0
    amplitude_decay: float = 0.0
    gauge_every_steps: int = 0
    gauge_at_epoch_end: bool = False
    gauge_recenter: bool = False
    gauge_eps: float = 1e-8
    moment_mode: Literal["unchanged", "reset"] = "unchanged"
    wb_decay: Literal["none", "standard_optimizer"] = "standard_optimizer"
    device: str = "cpu"


@dataclass
class TrainingHistory:
    losses: list[float] = field(default_factory=list)
    eval_metrics: dict[str, float] = field(default_factory=dict)
    projection_events: list[dict[str, float]] = field(default_factory=list)
    wall_clock_seconds: float = 0.0
    projection_wall_clock_seconds: float = 0.0


def build_model(config: ModelConfig) -> nn.Module:
    if config.name == "mlp":
        return MLPRegressor(config.input_dim, config.hidden_dims, config.output_dim)
    if config.name == "static_mlp_sequence":
        return StaticMLPSequenceRegressor(config.input_dim, config.hidden_dim, config.output_dim)
    if config.name in {"kan", "gauge_kan"}:
        return KANRegressor(
            config.input_dim,
            config.hidden_dims,
            config.output_dim,
            grid_size=config.grid_size,
            spline_order=config.spline_order,
        )
    if config.name == "lstm":
        return LSTMRegressor(config.input_dim, config.hidden_dim, config.output_dim)
    if config.name == "gru":
        return GRURegressor(config.input_dim, config.hidden_dim, config.output_dim)
    if config.name == "kan_gru":
        return KANGRURegressor(
            config.input_dim,
            config.hidden_dim,
            config.output_dim,
            grid_size=config.grid_size,
            spline_order=config.spline_order,
        )
    if config.name == "clipped_kan_gru":
        return ClippedKANGRURegressor(
            config.input_dim,
            config.hidden_dim,
            config.output_dim,
            grid_size=config.grid_size,
            spline_order=config.spline_order,
        )
    if config.name == "fused_clipped_kan_gru":
        return FusedClippedKANGRURegressor(
            config.input_dim,
            config.hidden_dim,
            config.output_dim,
            grid_size=config.grid_size,
            spline_order=config.spline_order,
        )
    if config.name == "hybrid_clipped_kan_gru":
        return HybridClippedKANGRURegressor(
            config.input_dim,
            config.hidden_dim,
            config.output_dim,
            grid_size=config.grid_size,
            spline_order=config.spline_order,
        )
    if config.name in {"kan_lstm", "gauge_kan_lstm"}:
        return KANLSTMRegressor(
            config.input_dim,
            config.hidden_dim,
            config.output_dim,
            grid_size=config.grid_size,
            spline_order=config.spline_order,
        )
    if config.name in {"bounded_kan_lstm", "gauge_bounded_kan_lstm"}:
        return BoundedKANLSTMRegressor(
            config.input_dim,
            config.hidden_dim,
            config.output_dim,
            grid_size=config.grid_size,
            spline_order=config.spline_order,
        )
    if config.name in {"clipped_kan_lstm", "gauge_clipped_kan_lstm"}:
        return ClippedKANLSTMRegressor(
            config.input_dim,
            config.hidden_dim,
            config.output_dim,
            grid_size=config.grid_size,
            spline_order=config.spline_order,
        )
    raise ValueError(f"unknown model name: {config.name}")


def build_optimizer(model: nn.Module, config: TrainingConfig) -> Optimizer:
    if config.wb_decay == "none" and config.weight_decay > 0:
        no_decay_ids = {
            id(module.scale_base)
            for module in model.modules()
            if isinstance(module, KANLayer) and module.scale_base.requires_grad
        }
        decay_params = [
            param for param in model.parameters() if param.requires_grad and id(param) not in no_decay_ids
        ]
        no_decay_params = [
            param for param in model.parameters() if param.requires_grad and id(param) in no_decay_ids
        ]
        params = [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        optimizer_weight_decay = 0.0
    else:
        params = [param for param in model.parameters() if param.requires_grad]
        optimizer_weight_decay = config.weight_decay
    if config.optimizer == "adamw":
        return AdamW(params, lr=config.lr, weight_decay=optimizer_weight_decay)
    if config.optimizer == "sgd":
        return SGD(params, lr=config.lr, weight_decay=optimizer_weight_decay, momentum=0.9)
    if config.optimizer == "lbfgs":
        return LBFGS(params, lr=config.lr, max_iter=4)
    raise ValueError(f"unknown optimizer: {config.optimizer}")


def _batch_loss(model: nn.Module, batch: tuple[torch.Tensor, torch.Tensor], config: TrainingConfig) -> torch.Tensor:
    x, y = (tensor.to(config.device) for tensor in batch)
    pred = model(x)
    loss = torch.mean((pred - y) ** 2)
    if config.amplitude_decay > 0.0:
        loss = loss + config.amplitude_decay * amplitude_penalty(model).to(loss.device)
    return loss


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    config: TrainingConfig,
    *,
    eval_loader: DataLoader | None = None,
) -> TrainingHistory:
    model.to(config.device)
    optimizer = build_optimizer(model, config)
    history = TrainingHistory()
    start = time.perf_counter()
    model.train()
    gauge_options = GaugeProjectionOptions(
        renormalize=True,
        recenter=config.gauge_recenter,
        eps=config.gauge_eps,
        moment_mode=config.moment_mode,
    )

    loader_iter = iter(train_loader)
    for step in range(1, config.steps + 1):
        try:
            batch = next(loader_iter)
        except StopIteration:
            if config.gauge_at_epoch_end:
                gauge_project_model(model, optimizer, gauge_options)
            loader_iter = iter(train_loader)
            batch = next(loader_iter)

        before = snapshot_gauge_state(model)
        optimizer.zero_grad(set_to_none=True)
        if isinstance(optimizer, LBFGS):
            def closure() -> torch.Tensor:
                optimizer.zero_grad(set_to_none=True)
                closure_loss = _batch_loss(model, batch, config)
                closure_loss.backward()
                return closure_loss

            loss = optimizer.step(closure)
            loss_value = float(loss.item())
        else:
            loss = _batch_loss(model, batch, config)
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().item())
        history.losses.append(loss_value)

        should_project = config.gauge_every_steps > 0 and step % config.gauge_every_steps == 0
        if should_project:
            update_proj = gauge_update_projection(model, before)
            pre_projection_loss = _batch_loss(model, batch, config).detach()
            projection_start = time.perf_counter()
            stats = gauge_project_model(model, optimizer, gauge_options)
            projection_seconds = time.perf_counter() - projection_start
            history.projection_wall_clock_seconds += projection_seconds
            post_projection_loss = _batch_loss(model, batch, config).detach()
            moment_proj = adam_moment_gauge_projection(model, optimizer)
            history.projection_events.append(
                {
                    "step": float(step),
                    "loss_before_step": loss_value,
                    "loss_before_projection": float(pre_projection_loss.item()),
                    "loss_after_projection": float(post_projection_loss.item()),
                    "projected_edges": float(stats.projected_edges),
                    "projection_wall_clock_seconds": projection_seconds,
                    "skipped_edges": float(stats.skipped_edges),
                    "reset_state_tensors": float(stats.reset_state_tensors),
                    "update_gauge_mean_abs": update_proj["mean_abs"],
                    "adam_moment_gauge_mean_abs": moment_proj["mean_abs"],
                }
            )

    history.wall_clock_seconds = time.perf_counter() - start
    if eval_loader is not None:
        history.eval_metrics = evaluate_model(model, eval_loader, device=config.device)
    return history


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, *, device: str = "cpu") -> dict[str, float]:
    model.eval()
    preds = []
    targets = []
    for x, y in loader:
        x = x.to(device)
        preds.append(model(x).cpu())
        targets.append(y.cpu())
    pred = torch.cat(preds, dim=0)
    target = torch.cat(targets, dim=0)
    if pred.ndim == 3:
        return sequence_metrics(pred, target)
    return regression_metrics(pred, target)
