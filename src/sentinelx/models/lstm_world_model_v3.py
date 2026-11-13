import math

import torch
import torch.nn as nn


class SentinelXLSTM(nn.Module):
    """
    SentinelX V3 LSTM World Model.

    Predicts the next network state from the previous
    sequence of observed network states.

    Architecture matches the trained V3 checkpoint:
      - Input features: 209
      - Hidden size: 128
      - LSTM layers: 1
      - Dropout: 0.2
      - Output features: 209
    """

    def __init__(
        self,
        input_size: int = 209,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()

        for name, value in (
            ("input_size", input_size),
            ("hidden_size", hidden_size),
            ("num_layers", num_layers),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(
                    f"{name} must be a positive integer."
                )

        try:
            dropout = float(dropout)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("dropout must be a finite number in [0, 1).")

        if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be a finite number in [0, 1).")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_probability = dropout

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)

        self.output_layer = nn.Linear(
            hidden_size,
            input_size,
        )

    def forward(self, x):
        """
        Input:
            (batch_size, sequence_length, input_size)

        Output:
            (batch_size, input_size)
        """

        if not isinstance(x, torch.Tensor):
            raise TypeError("Model input must be a PyTorch tensor.")

        if x.ndim != 3:
            raise ValueError(
                "Expected input with shape "
                "(batch_size, sequence_length, input_size)."
            )

        if x.shape[0] < 1:
            raise ValueError("Batch size cannot be zero.")

        if x.shape[1] < 1:
            raise ValueError("Sequence length cannot be zero.")

        if x.shape[2] != self.input_size:
            raise ValueError(
                f"Expected {self.input_size} features, "
                f"received {x.shape[2]}."
            )

        if not x.is_floating_point():
            raise TypeError("Model input must be floating-point.")

        sequence_output, _ = self.lstm(x)

        last_state = sequence_output[:, -1, :]

        last_state = self.dropout(last_state)

        return self.output_layer(last_state)
