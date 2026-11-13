"""Synthetic resistive-drift data for controlled smoke tests and ablations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import TensorDataset


@dataclass(frozen=True)
class SyntheticDriftConfig:
    num_series: int = 128
    series_length: int = 128
    min_resistance: float = 100.0
    max_resistance: float = 750_000.0
    seed: int = 0
    noise_scale: float = 0.015


@dataclass(frozen=True)
class SyntheticIVConfig:
    num_devices: int = 256
    num_cycles: int = 4
    points_per_cycle: int = 512
    seed: int = 0
    voltage_amplitude: float = 1.0
    dt: float = 1e-3
    g_min: float = 1e-6
    g_max: float = 8e-4
    state_noise: float = 0.002
    current_noise: float = 0.01


def generate_synthetic_drift(config: SyntheticDriftConfig) -> np.ndarray:
    """Generate smooth noisy resistance drift series.

    This is not a scientific replacement for the simulated IEEE DataPort drift
    set; it is a local controlled fixture for implementation smoke tests.
    """

    rng = np.random.default_rng(config.seed)
    initial = rng.uniform(config.min_resistance, config.max_resistance, size=config.num_series)
    equilibrium = rng.uniform(config.min_resistance, config.max_resistance, size=config.num_series)
    tau = rng.uniform(0.15, 0.85, size=config.num_series)
    t = np.linspace(0.0, 1.0, config.series_length)
    series = []
    for r0, req, tau_i in zip(initial, equilibrium, tau, strict=True):
        clean = req + (r0 - req) * np.exp(-t / tau_i)
        noise = rng.normal(0.0, config.noise_scale, size=config.series_length)
        noisy = clean * np.exp(noise)
        series.append(np.clip(noisy, config.min_resistance, config.max_resistance))
    return np.asarray(series, dtype=np.float32)


def make_point_history_dataset(series: np.ndarray) -> TensorDataset:
    """Return Track A point/history features ``[log R0, tau] -> log Rt``."""

    num_series, series_length = series.shape
    tau = np.linspace(0.0, 1.0, series_length, dtype=np.float32)
    log_series = np.log(series).astype(np.float32)
    r0 = np.repeat(log_series[:, [0]], series_length, axis=1)
    features = np.stack([r0, np.broadcast_to(tau, (num_series, series_length))], axis=-1)
    x = torch.from_numpy(features.reshape(-1, 2))
    y = torch.from_numpy(log_series.reshape(-1, 1))
    return TensorDataset(x, y)


def make_sequence_dataset(series: np.ndarray) -> TensorDataset:
    """Return Track B/C sequences ``[log R_{t-1}, tau_t, dt] -> log R_t``."""

    _, series_length = series.shape
    tau = np.linspace(0.0, 1.0, series_length, dtype=np.float32)
    dt = np.diff(tau, prepend=0.0).astype(np.float32)
    log_series = np.log(series).astype(np.float32)
    prev = np.concatenate([log_series[:, [0]], log_series[:, :-1]], axis=1)
    features = np.stack(
        [
            prev,
            np.broadcast_to(tau, log_series.shape),
            np.broadcast_to(dt, log_series.shape),
        ],
        axis=-1,
    )
    return TensorDataset(torch.from_numpy(features), torch.from_numpy(log_series[..., None]))


def _triangle_wave(num_points: int, amplitude: float) -> np.ndarray:
    phase = np.linspace(0.0, 1.0, num_points, endpoint=False)
    values = np.where(
        phase < 0.25,
        4.0 * phase,
        np.where(phase < 0.75, 2.0 - 4.0 * phase, -4.0 + 4.0 * phase),
    )
    return (amplitude * values).astype(np.float32)


def generate_synthetic_iv(config: SyntheticIVConfig) -> dict[str, np.ndarray]:
    """Generate controlled memristor-like hysteretic I-V sweeps.

    The generator is intentionally phenomenological: it creates nonlinear,
    state-dependent hysteresis for controlled algorithm tests, not a claim of a
    physically complete device model.
    """

    rng = np.random.default_rng(config.seed)
    steps = config.num_cycles * config.points_per_cycle
    base_v = np.tile(_triangle_wave(config.points_per_cycle, config.voltage_amplitude), config.num_cycles)
    time = (np.arange(steps, dtype=np.float32) * config.dt).astype(np.float32)

    voltage = np.broadcast_to(base_v, (config.num_devices, steps)).copy()
    current = np.empty_like(voltage)
    state = np.empty_like(voltage)
    device_params = np.empty((config.num_devices, 8), dtype=np.float32)

    for device in range(config.num_devices):
        g_min = config.g_min * rng.lognormal(0.0, 0.25)
        g_max = config.g_max * rng.lognormal(0.0, 0.35)
        if g_max <= g_min:
            g_max = g_min * 10.0
        eta = rng.uniform(0.02, 0.18) * g_max
        set_rate = rng.uniform(0.18, 1.4)
        reset_rate = rng.uniform(0.12, 1.1)
        threshold = rng.uniform(0.08, 0.28)
        leak = rng.uniform(0.001, 0.02)
        x_eq = rng.uniform(0.15, 0.85)
        x = rng.uniform(0.05, 0.95)
        device_params[device] = np.asarray(
            [g_min, g_max, eta, set_rate, reset_rate, threshold, leak, x_eq],
            dtype=np.float32,
        )
        for t, v in enumerate(base_v):
            drive = max(abs(float(v)) - threshold, 0.0)
            if v >= 0.0:
                dx = set_rate * drive * (1.0 - x)
            else:
                dx = -reset_rate * drive * x
            dx -= leak * (x - x_eq)
            dx += rng.normal(0.0, config.state_noise)
            x = float(np.clip(x + config.dt * dx, 0.0, 1.0))
            conductance = g_min + x * (g_max - g_min)
            clean_current = conductance * float(v) + eta * x * float(v) ** 3
            noisy_current = clean_current * (1.0 + rng.normal(0.0, config.current_noise))
            state[device, t] = x
            current[device, t] = noisy_current

    cycle = np.broadcast_to(
        np.repeat(np.arange(config.num_cycles, dtype=np.int32), config.points_per_cycle),
        (config.num_devices, steps),
    )
    sweep_direction = np.sign(np.diff(voltage, axis=1, prepend=voltage[:, :1])).astype(np.float32)
    return {
        "time": np.broadcast_to(time, (config.num_devices, steps)).astype(np.float32),
        "voltage": voltage.astype(np.float32),
        "current": current.astype(np.float32),
        "state": state.astype(np.float32),
        "cycle": cycle.astype(np.int32),
        "sweep_direction": sweep_direction,
        "device_params": device_params,
    }
