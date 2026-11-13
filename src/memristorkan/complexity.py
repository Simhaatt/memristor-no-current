"""Model complexity accounting for experiment reports."""

from __future__ import annotations

from torch import nn

from memristorkan.models import KANLayer


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {"total_parameters": total, "trainable_parameters": trainable}


def kan_complexity(model: nn.Module) -> dict[str, int]:
    layers = [module for module in model.modules() if isinstance(module, KANLayer)]
    edges = sum(layer.input_dim * layer.output_dim for layer in layers)
    spline_coefficients = sum(layer.coef.numel() for layer in layers)
    spline_scales = sum(layer.scale_sp.numel() for layer in layers)
    base_scales = sum(layer.scale_base.numel() for layer in layers)
    output_biases = sum(layer.output_bias.numel() for layer in layers if layer.output_bias is not None)
    return {
        "kan_layers": len(layers),
        "kan_edges": edges,
        "spline_coefficients": spline_coefficients,
        "spline_scales": spline_scales,
        "base_scales": base_scales,
        "kan_output_biases": output_biases,
    }


def model_complexity(model: nn.Module) -> dict[str, int]:
    return {**count_parameters(model), **kan_complexity(model)}
