"""Sequential baselines and KAN-LSTM hybrid models."""

from __future__ import annotations

import torch
from torch import nn

from memristorkan.models.kan import KANLayer, KANLayerConfig


class LSTMRegressor(nn.Module):
    """Track B LSTM baseline with hidden state carried across timesteps."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = 1,
        *,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.rnn = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.rnn(x)
        return self.head(y)


class GRURegressor(nn.Module):
    """Track B GRU baseline with hidden state carried across timesteps."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = 1,
        *,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.rnn(x)
        return self.head(y)


class StaticMLPSequenceRegressor(nn.Module):
    """Pointwise non-recurrent MLP over sequence features.

    This baseline maps each timestep independently:

    ``[V_t, V_{t-1}, dV_t, sweep_direction] -> I_t``

    for the no-current SDC protocol. It has no hidden state and therefore
    cannot use temporal memory except what is explicitly present in the
    per-step feature vector.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = 1,
        *,
        num_hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be at least 1")
        dims = [input_dim, *([hidden_dim] * num_hidden_layers), output_dim]
        modules: list[nn.Module] = []
        for i in range(len(dims) - 1):
            modules.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                modules.append(nn.SiLU())
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected sequence input shape (batch, seq, features), got {tuple(x.shape)}")
        batch, seq_len, features = x.shape
        y = self.net(x.reshape(batch * seq_len, features))
        return y.reshape(batch, seq_len, -1)


class KANLSTMCell(nn.Module):
    """LSTM cell whose four gate maps are produced by a small KAN layer."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        grid_size: int = 6,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gates = KANLayer(
            KANLayerConfig(
                input_dim + hidden_dim,
                4 * hidden_dim,
                grid_size=grid_size,
                spline_order=spline_order,
                grid_range=grid_range,
            )
        )

    def forward(self, x_t: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        h_prev, c_prev = state
        gate_input = torch.cat([x_t, h_prev], dim=-1)
        f_raw, i_raw, o_raw, g_raw = self.gates(gate_input).chunk(4, dim=-1)
        f_t = torch.sigmoid(f_raw)
        i_t = torch.sigmoid(i_raw)
        o_t = torch.sigmoid(o_raw)
        g_t = torch.tanh(g_raw)
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t


class KANLSTMRegressor(nn.Module):
    """Track C KAN-LSTM/Gauge-KAN-LSTM sequence regressor.

    Gauge behavior is controlled externally by applying Gauge-KAN projection to
    the contained KAN gate layer after optimizer steps.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = 1,
        *,
        grid_size: int = 6,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = KANLSTMCell(
            input_dim,
            hidden_dim,
            grid_size=grid_size,
            spline_order=spline_order,
            grid_range=grid_range,
        )
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected sequence input shape (batch, seq, features), got {tuple(x.shape)}")
        batch, seq_len, _ = x.shape
        h_t = x.new_zeros(batch, self.hidden_dim)
        c_t = x.new_zeros(batch, self.hidden_dim)
        outputs = []
        for t in range(seq_len):
            h_t, c_t = self.cell(x[:, t, :], (h_t, c_t))
            outputs.append(self.head(h_t))
        return torch.stack(outputs, dim=1)


class KANGRUCell(nn.Module):
    """GRU cell whose reset/update/candidate maps are produced by KAN layers."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        grid_size: int = 6,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gates = KANLayer(
            KANLayerConfig(
                input_dim + hidden_dim,
                2 * hidden_dim,
                grid_size=grid_size,
                spline_order=spline_order,
                grid_range=grid_range,
            )
        )
        self.candidate = KANLayer(
            KANLayerConfig(
                input_dim + hidden_dim,
                hidden_dim,
                grid_size=grid_size,
                spline_order=spline_order,
                grid_range=grid_range,
            )
        )

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        gate_input = torch.cat([x_t, h_prev], dim=-1)
        z_raw, r_raw = self.gates(gate_input).chunk(2, dim=-1)
        z_t = torch.sigmoid(z_raw)
        r_t = torch.sigmoid(r_raw)
        candidate_input = torch.cat([x_t, r_t * h_prev], dim=-1)
        n_t = torch.tanh(self.candidate(candidate_input))
        return (1.0 - z_t) * n_t + z_t * h_prev


class KANGRURegressor(nn.Module):
    """Sequence regressor using KAN-gated GRU dynamics."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = 1,
        *,
        grid_size: int = 6,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = KANGRUCell(
            input_dim,
            hidden_dim,
            grid_size=grid_size,
            spline_order=spline_order,
            grid_range=grid_range,
        )
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected sequence input shape (batch, seq, features), got {tuple(x.shape)}")
        batch, seq_len, _ = x.shape
        h_t = x.new_zeros(batch, self.hidden_dim)
        outputs = []
        for t in range(seq_len):
            h_t = self.cell(x[:, t, :], h_t)
            outputs.append(self.head(h_t))
        return torch.stack(outputs, dim=1)


class ClippedKANGRUCell(nn.Module):
    """KAN-GRU cell with directly clipped update/reset/candidate outputs."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        grid_size: int = 6,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gates = KANLayer(
            KANLayerConfig(
                input_dim + hidden_dim,
                2 * hidden_dim,
                grid_size=grid_size,
                spline_order=spline_order,
                grid_range=grid_range,
            )
        )
        self.candidate = KANLayer(
            KANLayerConfig(
                input_dim + hidden_dim,
                hidden_dim,
                grid_size=grid_size,
                spline_order=spline_order,
                grid_range=grid_range,
            )
        )

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        gate_input = torch.cat([x_t, h_prev], dim=-1)
        z_raw, r_raw = self.gates(gate_input).chunk(2, dim=-1)
        z_t = ((z_raw + 1.0) * 0.5).clamp(0.0, 1.0)
        r_t = ((r_raw + 1.0) * 0.5).clamp(0.0, 1.0)
        candidate_input = torch.cat([x_t, r_t * h_prev], dim=-1)
        n_t = self.candidate(candidate_input).clamp(-1.0, 1.0)
        return (1.0 - z_t) * n_t + z_t * h_prev


class ClippedKANGRURegressor(nn.Module):
    """Sequence regressor using directly clipped KAN-GRU gates."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = 1,
        *,
        grid_size: int = 6,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = ClippedKANGRUCell(
            input_dim,
            hidden_dim,
            grid_size=grid_size,
            spline_order=spline_order,
            grid_range=grid_range,
        )
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected sequence input shape (batch, seq, features), got {tuple(x.shape)}")
        batch, seq_len, _ = x.shape
        h_t = x.new_zeros(batch, self.hidden_dim)
        outputs = []
        for t in range(seq_len):
            h_t = self.cell(x[:, t, :], h_t)
            outputs.append(self.head(h_t))
        return torch.stack(outputs, dim=1)


class FusedClippedKANGRUCell(nn.Module):
    """One-KAN-call clipped GRU-like cell for efficiency ablations.

    Unlike faithful GRU dynamics, the reset gate does not condition a second
    candidate map. The KAN emits update, reset, and candidate channels in one
    call, then the reset gate modulates the raw candidate:

    ``z, r, raw_n = KAN([x_t, h_{t-1}])`` and
    ``n_t = tanh(r_t * raw_n)``.

    This is a speed/complexity ablation, not a standard GRU replacement.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        grid_size: int = 6,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gates = KANLayer(
            KANLayerConfig(
                input_dim + hidden_dim,
                3 * hidden_dim,
                grid_size=grid_size,
                spline_order=spline_order,
                grid_range=grid_range,
            )
        )

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        gate_input = torch.cat([x_t, h_prev], dim=-1)
        z_raw, r_raw, n_raw = self.gates(gate_input).chunk(3, dim=-1)
        z_t = ((z_raw + 1.0) * 0.5).clamp(0.0, 1.0)
        r_t = ((r_raw + 1.0) * 0.5).clamp(0.0, 1.0)
        n_t = torch.tanh(r_t * n_raw)
        return (1.0 - z_t) * n_t + z_t * h_prev


class FusedClippedKANGRURegressor(nn.Module):
    """Sequence regressor using a one-call clipped KAN-GRU-like cell."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = 1,
        *,
        grid_size: int = 6,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = FusedClippedKANGRUCell(
            input_dim,
            hidden_dim,
            grid_size=grid_size,
            spline_order=spline_order,
            grid_range=grid_range,
        )
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected sequence input shape (batch, seq, features), got {tuple(x.shape)}")
        batch, seq_len, _ = x.shape
        h_t = x.new_zeros(batch, self.hidden_dim)
        outputs = []
        for t in range(seq_len):
            h_t = self.cell(x[:, t, :], h_t)
            outputs.append(self.head(h_t))
        return torch.stack(outputs, dim=1)


class HybridClippedKANGRUCell(nn.Module):
    """GRU cell with cheap dense gates and a clipped KAN candidate.

    The update/reset gates are standard dense sigmoid gates, while the
    candidate state uses a KAN layer over the reset-gated hidden state:

    ``z_t, r_t = sigmoid(Dense([x_t, h_{t-1}]))``
    ``n_t = clip(KAN([x_t, r_t * h_{t-1}]), -1, 1)``

    This tests whether KAN capacity is most useful in the nonlinear device
    state update rather than in every recurrent control gate.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        grid_size: int = 6,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gates = nn.Linear(input_dim + hidden_dim, 2 * hidden_dim)
        self.candidate = KANLayer(
            KANLayerConfig(
                input_dim + hidden_dim,
                hidden_dim,
                grid_size=grid_size,
                spline_order=spline_order,
                grid_range=grid_range,
            )
        )

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        gate_input = torch.cat([x_t, h_prev], dim=-1)
        z_raw, r_raw = self.gates(gate_input).chunk(2, dim=-1)
        z_t = torch.sigmoid(z_raw)
        r_t = torch.sigmoid(r_raw)
        candidate_input = torch.cat([x_t, r_t * h_prev], dim=-1)
        n_t = self.candidate(candidate_input).clamp(-1.0, 1.0)
        return (1.0 - z_t) * n_t + z_t * h_prev


class HybridClippedKANGRURegressor(nn.Module):
    """Sequence regressor with dense GRU gates and a clipped KAN candidate."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = 1,
        *,
        grid_size: int = 6,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = HybridClippedKANGRUCell(
            input_dim,
            hidden_dim,
            grid_size=grid_size,
            spline_order=spline_order,
            grid_range=grid_range,
        )
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected sequence input shape (batch, seq, features), got {tuple(x.shape)}")
        batch, seq_len, _ = x.shape
        h_t = x.new_zeros(batch, self.hidden_dim)
        outputs = []
        for t in range(seq_len):
            h_t = self.cell(x[:, t, :], h_t)
            outputs.append(self.head(h_t))
        return torch.stack(outputs, dim=1)


class BoundedKANLSTMCell(nn.Module):
    """KAN-LSTM cell with explicitly bounded, coupled memory gates.

    The KAN emits raw gate scores, but the forget/input pair is normalized with
    a softmax so ``f_t + i_t = 1`` and both gates remain in ``[0, 1]``. This is
    an experimental stable-memory variant, distinct from the standard KAN-LSTM
    that applies independent sigmoid gates.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        grid_size: int = 6,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gates = KANLayer(
            KANLayerConfig(
                input_dim + hidden_dim,
                4 * hidden_dim,
                grid_size=grid_size,
                spline_order=spline_order,
                grid_range=grid_range,
            )
        )

    def forward(self, x_t: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        h_prev, c_prev = state
        gate_input = torch.cat([x_t, h_prev], dim=-1)
        f_raw, i_raw, o_raw, g_raw = self.gates(gate_input).chunk(4, dim=-1)
        memory_mix = torch.softmax(torch.stack([f_raw, i_raw], dim=-1), dim=-1)
        f_t = memory_mix[..., 0]
        i_t = memory_mix[..., 1]
        o_t = torch.sigmoid(o_raw)
        g_t = torch.tanh(g_raw)
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t


class BoundedKANLSTMRegressor(nn.Module):
    """Sequence regressor using coupled bounded KAN-LSTM cells."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = 1,
        *,
        grid_size: int = 6,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = BoundedKANLSTMCell(
            input_dim,
            hidden_dim,
            grid_size=grid_size,
            spline_order=spline_order,
            grid_range=grid_range,
        )
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected sequence input shape (batch, seq, features), got {tuple(x.shape)}")
        batch, seq_len, _ = x.shape
        h_t = x.new_zeros(batch, self.hidden_dim)
        c_t = x.new_zeros(batch, self.hidden_dim)
        outputs = []
        for t in range(seq_len):
            h_t, c_t = self.cell(x[:, t, :], (h_t, c_t))
            outputs.append(self.head(h_t))
        return torch.stack(outputs, dim=1)


class ClippedKANLSTMCell(nn.Module):
    """KAN-LSTM cell with directly clipped gate outputs.

    This experimental variant removes sigmoid/tanh gate activations and instead
    uses hard clipping:

    ``f/i/o = clamp((raw + 1) / 2, 0, 1)`` and ``g = clamp(raw, -1, 1)``.

    It tests whether a KAN can directly represent bounded gate functions. The
    tradeoff is that saturated clipped regions have zero gradient.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        grid_size: int = 6,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gates = KANLayer(
            KANLayerConfig(
                input_dim + hidden_dim,
                4 * hidden_dim,
                grid_size=grid_size,
                spline_order=spline_order,
                grid_range=grid_range,
            )
        )

    def forward(self, x_t: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        h_prev, c_prev = state
        gate_input = torch.cat([x_t, h_prev], dim=-1)
        f_raw, i_raw, o_raw, g_raw = self.gates(gate_input).chunk(4, dim=-1)
        f_t = ((f_raw + 1.0) * 0.5).clamp(0.0, 1.0)
        i_t = ((i_raw + 1.0) * 0.5).clamp(0.0, 1.0)
        o_t = ((o_raw + 1.0) * 0.5).clamp(0.0, 1.0)
        g_t = g_raw.clamp(-1.0, 1.0)
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t


class ClippedKANLSTMRegressor(nn.Module):
    """Sequence regressor using directly clipped KAN-LSTM gates."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = 1,
        *,
        grid_size: int = 6,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = ClippedKANLSTMCell(
            input_dim,
            hidden_dim,
            grid_size=grid_size,
            spline_order=spline_order,
            grid_range=grid_range,
        )
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected sequence input shape (batch, seq, features), got {tuple(x.shape)}")
        batch, seq_len, _ = x.shape
        h_t = x.new_zeros(batch, self.hidden_dim)
        c_t = x.new_zeros(batch, self.hidden_dim)
        outputs = []
        for t in range(seq_len):
            h_t, c_t = self.cell(x[:, t, :], (h_t, c_t))
            outputs.append(self.head(h_t))
        return torch.stack(outputs, dim=1)
