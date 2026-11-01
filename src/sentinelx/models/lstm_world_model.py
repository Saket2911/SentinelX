
import torch
import torch.nn as nn


class LSTMWorldModel(nn.Module):
    """
    Predicts the next network state from a sequence of
    previously observed network states.
    """

    def __init__(
        self,
        input_size: int = 206,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        if input_size <= 0:
            raise ValueError("input_size must be positive.")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        self.input_size = input_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_layer = nn.Linear(
            hidden_size,
            input_size,
        )

    def forward(self, x):
        """
        Input:  (batch_size, sequence_length, input_size)
        Output: (batch_size, input_size)
        """

        if x.ndim != 3:
            raise ValueError(
                "Expected input shape "
                "(batch_size, sequence_length, input_size)."
            )

        if x.shape[-1] != self.input_size:
            raise ValueError(
                f"Expected {self.input_size} features, "
                f"received {x.shape[-1]}."
            )

        lstm_output, _ = self.lstm(x)

        # Use the final time step to predict the next state.
        final_state = lstm_output[:, -1, :]

        prediction = self.output_layer(final_state)

        return prediction
