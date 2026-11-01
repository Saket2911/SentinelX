
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sentinelx.models.lstm_world_model import LSTMWorldModel


def main():
    torch.manual_seed(42)

    sequence_length = 10
    feature_count = 206
    batch_size = 32

    model = LSTMWorldModel(
        input_size=feature_count,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
    )

    # Simulate a batch of 32 sequences.
    sample_input = torch.randn(
        batch_size,
        sequence_length,
        feature_count,
        dtype=torch.float32,
    )

    # Test prediction without calculating gradients.
    model.eval()
    with torch.no_grad():
        prediction = model(sample_input)

    print("LSTM World Model Test")
    print("-" * 35)
    print("Input shape:", tuple(sample_input.shape))
    print("Prediction shape:", tuple(prediction.shape))
    print("Parameters:", sum(p.numel() for p in model.parameters()))
    print("All predictions finite:", torch.isfinite(prediction).all().item())

    assert prediction.shape == (batch_size, feature_count)
    assert torch.isfinite(prediction).all()

    print("\nLSTM model test passed.")


if __name__ == "__main__":
    main()
