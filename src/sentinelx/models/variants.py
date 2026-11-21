"""Justified world-model variants for controlled experiments.

The production candidate stays ``SentinelXLSTM`` (V3). These variants exist so
architecture choices can be measured rather than assumed:

``baseline_v3``    the V3 architecture, used as the control
``residual_head``  predicts a delta added to the last observed state, encoding a
                   persistence prior; useful when the network state is strongly
                   autocorrelated (which the recorded persistence baseline shows)
``stacked_two``    two recurrent layers with inter-layer dropout, testing extra
                   capacity/temporal depth

No variant changes the feature contract: all map 209 inputs to 209 outputs.
"""

from __future__ import annotations

import math
from typing import Dict

VARIANT_NAMES = ("baseline_v3", "residual_head", "stacked_two")

VARIANT_DESCRIPTIONS: Dict[str, str] = {
    "baseline_v3": "Control: 1-layer LSTM, dropout before the linear output head.",
    "residual_head": "LSTM predicts a delta added to the last input state (persistence prior).",
    "stacked_two": "2-layer LSTM with dropout between layers and before the output head.",
}


def _require_torch():
    try:
        import torch  # noqa: F401
        import torch.nn as nn  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "PyTorch is required for world-model variants. Install the training "
            "requirements before running experiments."
        ) from exc
    import torch
    import torch.nn as nn

    return torch, nn


def _validate(input_size: int, hidden_size: int, num_layers: int, dropout: float) -> None:
    for name, value in (("input_size", input_size), ("hidden_size", hidden_size), ("num_layers", num_layers)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be finite and in [0, 1).")


def build_world_model(
    variant: str = "baseline_v3",
    input_size: int = 209,
    hidden_size: int = 128,
    num_layers: int = 1,
    dropout: float = 0.2,
):
    """Construct one of the registered variants. V3 remains the default."""
    if variant not in VARIANT_NAMES:
        raise ValueError(f"Unknown variant {variant!r}. Choose from {VARIANT_NAMES}.")
    _validate(input_size, hidden_size, num_layers, dropout)
    torch, nn = _require_torch()

    if variant == "baseline_v3":
        from .lstm_world_model_v3 import SentinelXLSTM

        return SentinelXLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

    if variant == "residual_head":

        class ResidualWorldModel(nn.Module):
            """Predicts ``last_state + delta`` so the identity mapping is easy to learn."""

            def __init__(self) -> None:
                super().__init__()
                self.input_size = input_size
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0.0,
                )
                self.dropout = nn.Dropout(dropout)
                self.output_layer = nn.Linear(hidden_size, input_size)

            def forward(self, x):
                if x.ndim != 3 or x.shape[2] != self.input_size:
                    raise ValueError(
                        f"Expected (batch, sequence, {self.input_size}), received {tuple(x.shape)}."
                    )
                sequence_output, _ = self.lstm(x)
                delta = self.output_layer(self.dropout(sequence_output[:, -1, :]))
                return x[:, -1, :] + delta

        return ResidualWorldModel()

    if variant == "stacked_two":

        class StackedWorldModel(nn.Module):
            """Two recurrent layers; num_layers is forced to at least 2."""

            def __init__(self) -> None:
                super().__init__()
                self.input_size = input_size
                layers = max(2, num_layers)
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=layers,
                    batch_first=True,
                    dropout=dropout,
                )
                self.dropout = nn.Dropout(dropout)
                self.output_layer = nn.Linear(hidden_size, input_size)

            def forward(self, x):
                if x.ndim != 3 or x.shape[2] != self.input_size:
                    raise ValueError(
                        f"Expected (batch, sequence, {self.input_size}), received {tuple(x.shape)}."
                    )
                sequence_output, _ = self.lstm(x)
                return self.output_layer(self.dropout(sequence_output[:, -1, :]))

        return StackedWorldModel()

    raise AssertionError("unreachable")  # pragma: no cover
