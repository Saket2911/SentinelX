from pathlib import Path
import math

import numpy as np

try:
    import torch
except Exception as exc:  # pragma: no cover - import is optional
    torch = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


def _positive_integer(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer.")


def load_world_model(
    checkpoint_path,
    device=None,
):
    """
    Load the trained SentinelX V3 model and its metadata.

    Returns:
        model, metadata
    """
    if torch is None:
        raise RuntimeError(
            "PyTorch is required to load the SentinelX world model. "
            "The environment cannot import torch."
        ) from _TORCH_IMPORT_ERROR

    from sentinelx.models.lstm_world_model_v3 import SentinelXLSTM

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available."
        )

    # Only load checkpoints from trusted sources.
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dictionary.")

    required = {
        "model_state_dict",
        "input_features",
        "hidden_size",
        "num_layers",
        "dropout",
        "sequence_length",
        "feature_names",
        "epoch",
    }

    missing = required - set(checkpoint)

    if missing:
        raise ValueError(
            f"Checkpoint is missing fields: {sorted(missing)}"
        )

    # Validate architecture dimensions.
    _positive_integer(checkpoint["input_features"], "input_features")
    _positive_integer(checkpoint["hidden_size"], "hidden_size")
    _positive_integer(checkpoint["num_layers"], "num_layers")
    _positive_integer(checkpoint["sequence_length"], "sequence_length")
    _positive_integer(checkpoint["epoch"], "epoch")

    # Validate the exact trained V3 configuration.
    expected_configuration = {
        "input_features": 209,
        "hidden_size": 128,
        "num_layers": 1,
        "sequence_length": 10,
    }

    for key, expected in expected_configuration.items():
        if checkpoint[key] != expected:
            raise ValueError(
                f"Unsupported V3 configuration: "
                f"{key}={checkpoint[key]}, expected {expected}."
            )

    try:
        dropout = float(checkpoint["dropout"])
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Checkpoint dropout must be numeric.")

    if not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError(
            "Checkpoint dropout must be finite and in [0, 1)."
        )

    # Validate feature names.
    feature_names = checkpoint["feature_names"]

    if not isinstance(feature_names, (list, tuple)):
        raise ValueError(
            "Checkpoint feature_names must be a list or tuple."
        )

    if len(feature_names) != checkpoint["input_features"]:
        raise ValueError(
            "Checkpoint feature-name count does not match "
            "the model input size."
        )

    if not all(isinstance(name, str) and name for name in feature_names):
        raise ValueError(
            "Checkpoint contains invalid feature names."
        )

    if len(feature_names) != len(set(feature_names)):
        raise ValueError(
            "Duplicate feature names found in checkpoint."
        )

    if not isinstance(checkpoint["model_state_dict"], dict):
        raise ValueError("model_state_dict must be a dictionary.")

    # Reconstruct the exact V3 architecture.
    model = SentinelXLSTM(
        input_size=checkpoint["input_features"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        dropout=dropout,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    # Attach the expected sequence length for inference validation.
    model.expected_sequence_length = checkpoint["sequence_length"]

    model.to(device)
    model.eval()

    metadata = {
        "epoch": checkpoint["epoch"],
        "sequence_length": checkpoint["sequence_length"],
        "input_features": checkpoint["input_features"],
        "hidden_size": checkpoint["hidden_size"],
        "num_layers": checkpoint["num_layers"],
        "dropout": dropout,
        "feature_names": list(feature_names),
        "device": str(device),
    }

    return model, metadata


def predict_next_state(model, sequence):
    """
    Predict the next network state.

    Accepts:
        Single sequence: (sequence_length, features)
        Batch:           (batch_size, sequence_length, features)

    Returns:
        NumPy array:
            Single sequence -> (features,)
            Batch           -> (batch_size, features)
    """
    if torch is None:
        raise RuntimeError(
            "PyTorch is required to run model inference. "
            "The environment cannot import torch."
        ) from _TORCH_IMPORT_ERROR

    if isinstance(sequence, np.ndarray):
        try:
            sequence = torch.from_numpy(sequence)
        except (TypeError, ValueError):
            raise ValueError(
                "NumPy sequence must contain supported numeric values."
            )

    if not isinstance(sequence, torch.Tensor):
        raise TypeError(
            "sequence must be a NumPy array or PyTorch tensor."
        )

    if sequence.ndim == 2:
        sequence = sequence.unsqueeze(0)
        single_sequence = True
    elif sequence.ndim == 3:
        single_sequence = False
    else:
        raise ValueError(
            "Expected a 2D sequence or 3D batch."
        )

    if sequence.shape[0] < 1:
        raise ValueError("Batch cannot be empty.")

    if sequence.shape[1] < 1:
        raise ValueError("Sequence length cannot be zero.")

    if sequence.shape[2] < 1:
        raise ValueError("Feature dimension cannot be zero.")

    if not sequence.is_floating_point():
        sequence = sequence.float()

    if not torch.isfinite(sequence).all():
        raise ValueError(
            "Input sequence contains non-finite values."
        )

    expected_features = model.input_size

    if sequence.shape[2] != expected_features:
        raise ValueError(
            f"Expected {expected_features} features, "
            f"received {sequence.shape[2]}."
        )

    expected_sequence_length = getattr(
        model,
        "expected_sequence_length",
        None,
    )

    if (
        expected_sequence_length is not None
        and sequence.shape[1] != expected_sequence_length
    ):
        raise ValueError(
            f"Expected sequence length {expected_sequence_length}, "
            f"received {sequence.shape[1]}."
        )

    device = next(model.parameters()).device

    sequence = sequence.to(
        device=device,
        dtype=torch.float32,
    )

    model.eval()

    with torch.inference_mode():
        prediction = model(sequence)

    expected_shape = (sequence.shape[0], expected_features)

    if tuple(prediction.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected model output shape: {tuple(prediction.shape)}; "
            f"expected {expected_shape}."
        )

    if not torch.isfinite(prediction).all():
        raise RuntimeError(
            "Model produced non-finite predictions."
        )

    result = prediction.cpu().numpy()

    return result[0] if single_sequence else result
