"""Fixed-grid B-spline basis utilities for compact KAN layers."""

from __future__ import annotations

import torch


def make_clamped_uniform_knots(
    grid_size: int,
    degree: int,
    grid_range: tuple[float, float] = (-1.0, 1.0),
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return an open uniform knot vector with repeated boundary knots."""

    if grid_size < 1:
        raise ValueError("grid_size must be >= 1")
    if degree < 0:
        raise ValueError("degree must be >= 0")
    left, right = grid_range
    if not left < right:
        raise ValueError("grid_range must be increasing")

    internal = torch.linspace(left, right, grid_size + 1, device=device, dtype=dtype)
    if grid_size > 1:
        middle = internal[1:-1]
    else:
        middle = internal.new_empty(0)
    return torch.cat(
        [
            internal.new_full((degree + 1,), left),
            middle,
            internal.new_full((degree + 1,), right),
        ]
    )


def bspline_basis(x: torch.Tensor, knots: torch.Tensor, degree: int) -> torch.Tensor:
    """Evaluate all clamped B-spline basis functions at ``x``.

    Args:
        x: Tensor of shape ``(...,)``.
        knots: 1D knot vector.
        degree: Polynomial degree.

    Returns:
        Tensor of shape ``(..., n_basis)``.
    """

    if knots.ndim != 1:
        raise ValueError("knots must be a 1D tensor")
    if degree < 0:
        raise ValueError("degree must be >= 0")
    if knots.numel() <= degree + 1:
        raise ValueError("knot vector is too short for the requested degree")

    eps = torch.finfo(x.dtype).eps * 16
    left = knots[0].to(dtype=x.dtype, device=x.device)
    right = knots[-1].to(dtype=x.dtype, device=x.device)
    knots = knots.to(dtype=x.dtype, device=x.device)
    x_eval = x.clamp(min=float(left), max=float(right - eps))
    x_expanded = x_eval.unsqueeze(-1)

    basis = ((x_expanded >= knots[:-1]) & (x_expanded < knots[1:])).to(x.dtype)
    for d in range(1, degree + 1):
        left_den = knots[d:-1] - knots[: -(d + 1)]
        right_den = knots[d + 1 :] - knots[1:-d]
        left_num = x_expanded - knots[: -(d + 1)]
        right_num = knots[d + 1 :] - x_expanded

        left_factor = torch.where(
            left_den.abs() > eps,
            left_num / left_den.clamp_min(eps),
            torch.zeros_like(left_num),
        )
        right_factor = torch.where(
            right_den.abs() > eps,
            right_num / right_den.clamp_min(eps),
            torch.zeros_like(right_num),
        )
        basis = left_factor * basis[..., :-1] + right_factor * basis[..., 1:]

    return basis
