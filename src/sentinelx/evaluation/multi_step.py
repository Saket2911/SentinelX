import numpy as np
import pandas as pd
import torch


def evaluate_multi_step(
    model,
    X,
    y,
    metadata,
    horizons=(1, 2, 5, 10, 30),
    batch_size=128,
    huber_delta=1.0,
):
    """
    Evaluate recursive multi-step future-state forecasting.

    X[i] contains the observed input sequence for forecast origin i.
    y[i] contains the actual next state for that origin.

    Each recursive prediction is fed into the next model input.

    Forecast paths crossing a segment boundary or a timestamp
    interval other than exactly one second are excluded.

    Returns:
        results: DataFrame containing overall and label-wise metrics.
        diagnostics: Evaluation cohort and continuity information.
    """

    # Validate forecast horizons without silently truncating values.
    try:
        supplied_horizons = tuple(horizons)
    except TypeError:
        raise ValueError(
            "Horizons must be an iterable of positive integers."
        )

    if not supplied_horizons:
        raise ValueError("Horizons must contain positive integers.")

    for horizon in supplied_horizons:
        if (
            isinstance(horizon, (bool, np.bool_))
            or not isinstance(horizon, (int, np.integer))
            or horizon < 1
        ):
            raise ValueError(
                "Every horizon must be a positive integer."
            )

    horizons = tuple(sorted(set(int(h) for h in supplied_horizons)))

    # Validate batch size.
    if (
        isinstance(batch_size, (bool, np.bool_))
        or not isinstance(batch_size, (int, np.integer))
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer.")

    # Validate Huber delta before inference.
    try:
        huber_delta = float(huber_delta)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            "huber_delta must be a finite positive number."
        )

    if not np.isfinite(huber_delta) or huber_delta <= 0:
        raise ValueError(
            "huber_delta must be a finite positive number."
        )

    # Validate and normalize arrays.
    try:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("X and y must contain numeric values.")

    if X.ndim != 3 or y.ndim != 2:
        raise ValueError("X must be 3D and y must be 2D.")

    n, sequence_length, features = X.shape

    if n == 0:
        raise ValueError("Evaluation dataset is empty.")

    if sequence_length < 1 or features < 1:
        raise ValueError(
            "X must have positive sequence and feature dimensions."
        )

    if y.shape != (n, features):
        raise ValueError("X and y dimensions do not match.")

    if not np.isfinite(X).all():
        raise ValueError("Input sequences contain non-finite values.")

    if not np.isfinite(y).all():
        raise ValueError("Targets contain non-finite values.")

    # Validate metadata.
    if not isinstance(metadata, pd.DataFrame):
        raise ValueError("metadata must be a pandas DataFrame.")

    if len(metadata) != n:
        raise ValueError("Metadata count does not match X.")

    required = {
        "Target Timestamp",
        "Target Label",
        "Segment ID",
    }

    if not required.issubset(metadata.columns):
        raise ValueError(
            f"Missing metadata columns: "
            f"{sorted(required - set(metadata.columns))}"
        )

    if metadata["Segment ID"].isna().any():
        raise ValueError("Metadata contains missing Segment IDs.")

    # Validate model compatibility.
    if model.input_size != features:
        raise ValueError("Model input size does not match X.")

    # Validate timestamps and chronological ordering.
    timestamps = pd.to_datetime(
        metadata["Target Timestamp"],
        errors="coerce",
    )

    if timestamps.isna().any():
        raise ValueError("Metadata contains invalid target timestamps.")

    if not timestamps.is_monotonic_increasing:
        raise ValueError(
            "Target timestamps must be in chronological order."
        )

    timestamp_ns = timestamps.to_numpy(
        dtype="datetime64[ns]"
    ).astype("int64")

    segments = metadata["Segment ID"].to_numpy()

    labels = (
        metadata["Target Label"]
        .fillna("UNKNOWN")
        .astype(str)
        .to_numpy()
    )

    max_horizon = max(horizons)
    total_origins = n - max_horizon + 1

    if total_origins <= 0:
        raise ValueError(
            "Not enough sequences for the maximum forecast horizon."
        )

    # An edge is invalid when consecutive target states are not
    # exactly one second apart or belong to different segments.
    bad_edges = (
        (np.diff(timestamp_ns) != 1_000_000_000)
        | (segments[1:] != segments[:-1])
    )

    bad_edge_prefix = np.concatenate([
        np.array([0], dtype=np.int64),
        np.cumsum(bad_edges, dtype=np.int64),
    ])

    valid_origins = {}

    for horizon in horizons:
        possible = np.arange(0, n - horizon + 1)

        invalid_counts = (
            bad_edge_prefix[possible + horizon - 1]
            - bad_edge_prefix[possible]
        )

        valid_origins[horizon] = invalid_counts == 0

    # Accumulate sums rather than retaining all forecast tensors.
    stores = {}

    def add_metrics(key, prediction, actual):
        difference = (
            prediction.astype(np.float64)
            - actual.astype(np.float64)
        )

        absolute = np.abs(difference)

        huber = np.where(
            absolute <= huber_delta,
            0.5 * difference ** 2,
            huber_delta * (absolute - 0.5 * huber_delta),
        )

        if key not in stores:
            stores[key] = {
                "huber": 0.0,
                "absolute": 0.0,
                "squared": 0.0,
                "elements": 0,
                "states": 0,
            }

        store = stores[key]

        store["huber"] += huber.sum(dtype=np.float64)
        store["absolute"] += absolute.sum(dtype=np.float64)
        store["squared"] += np.square(
            difference
        ).sum(dtype=np.float64)

        store["elements"] += difference.size
        store["states"] += len(prediction)

    def calculate(store):
        count = store["elements"]

        return {
            "Huber": store["huber"] / count,
            "MAE": store["absolute"] / count,
            "RMSE": np.sqrt(store["squared"] / count),
            "States": store["states"],
        }

    device = next(model.parameters()).device
    model.eval()

    with torch.inference_mode():
        for batch_start in range(0, total_origins, batch_size):
            batch_end = min(
                batch_start + batch_size,
                total_origins,
            )

            starts = np.arange(batch_start, batch_end)

            batch_np = np.array(
                X[starts],
                dtype=np.float32,
                copy=True,
            )

            current = torch.from_numpy(batch_np).to(device)

            # Persistence repeats the last observed state.
            persistence_prediction = batch_np[:, -1, :].copy()

            for horizon in range(1, max_horizon + 1):
                prediction_tensor = model(current)

                expected_shape = (len(starts), features)

                if tuple(prediction_tensor.shape) != expected_shape:
                    raise ValueError(
                        f"Model output shape at horizon {horizon} "
                        f"was {tuple(prediction_tensor.shape)}, "
                        f"expected {expected_shape}."
                    )

                prediction = prediction_tensor.cpu().numpy()

                if not np.isfinite(prediction).all():
                    raise ValueError(
                        f"Non-finite prediction at horizon {horizon}."
                    )

                if horizon in horizons:
                    valid_mask = valid_origins[horizon][starts]

                    if valid_mask.any():
                        selected_starts = starts[valid_mask]
                        target_indices = selected_starts + horizon - 1

                        actual = np.array(
                            y[target_indices],
                            dtype=np.float32,
                            copy=True,
                        )

                        selected_prediction = prediction[valid_mask]

                        selected_persistence = (
                            persistence_prediction[valid_mask]
                        )

                        future_labels = labels[target_indices]

                        add_metrics(
                            (horizon, "ALL", "LSTM"),
                            selected_prediction,
                            actual,
                        )

                        add_metrics(
                            (horizon, "ALL", "Persistence"),
                            selected_persistence,
                            actual,
                        )

                        for label in np.unique(future_labels):
                            mask = future_labels == label

                            add_metrics(
                                (horizon, str(label), "LSTM"),
                                selected_prediction[mask],
                                actual[mask],
                            )

                            add_metrics(
                                (horizon, str(label), "Persistence"),
                                selected_persistence[mask],
                                actual[mask],
                            )

                # Recursive update: predicted state becomes the
                # newest input state. No actual future state is fed in.
                current = torch.cat(
                    [
                        current[:, 1:, :],
                        prediction_tensor.unsqueeze(1),
                    ],
                    dim=1,
                )

    rows = []

    for horizon in horizons:
        available_labels = sorted(
            label
            for h, label, method in stores
            if h == horizon and label != "ALL" and method == "LSTM"
        )

        for label in ["ALL"] + available_labels:
            lstm_key = (horizon, label, "LSTM")
            persistence_key = (horizon, label, "Persistence")

            if lstm_key not in stores:
                continue

            lstm = calculate(stores[lstm_key])
            persistence = calculate(stores[persistence_key])

            rows.append({
                "Horizon_Seconds": horizon,
                "Target_Label": label,
                "Valid_Origins": lstm["States"],
                "LSTM_Huber": lstm["Huber"],
                "Persistence_Huber": persistence["Huber"],
                "LSTM_MAE": lstm["MAE"],
                "Persistence_MAE": persistence["MAE"],
                "MAE_Improvement_Percent": (
                    (persistence["MAE"] - lstm["MAE"])
                    / persistence["MAE"] * 100
                    if persistence["MAE"] != 0 else 0.0
                ),
                "LSTM_RMSE": lstm["RMSE"],
                "Persistence_RMSE": persistence["RMSE"],
                "RMSE_Improvement_Percent": (
                    (persistence["RMSE"] - lstm["RMSE"])
                    / persistence["RMSE"] * 100
                    if persistence["RMSE"] != 0 else 0.0
                ),
            })

    results = pd.DataFrame(rows)

    diagnostics = {
        "samples": int(n),
        "sequence_length": int(sequence_length),
        "features": int(features),
        "horizons": list(horizons),
        "common_origins": int(total_origins),
        "valid_origins_by_horizon": {
            horizon: int(
                valid_origins[horizon][:total_origins].sum()
            )
            for horizon in horizons
        },
        "device": str(device),
        "huber_delta": float(huber_delta),
        "recursive_predictions": True,
        "persistence_repeats_last_observed_state": True,
        "continuity_rule": (
            "Same segment and exactly one second between "
            "consecutive target states."
        ),
    }

    return results, diagnostics
