import pandas as pd

from sentinelx.features.preprocessor_v3 import SentinelXV3Preprocessor
from sentinelx.inference.world_model import (
    load_world_model,
    predict_next_state,
)


class SentinelXV3Predictor:
    """
    End-to-end V3 predictor.

    Input:
        10 consecutive, one-second raw network-state rows.

    Processing:
        V3 preprocessing -> LSTM -> predicted next state.

    Output:
        Predicted next state in standardized model feature space.
    """

    def __init__(self, model, metadata, preprocessor):
        if not isinstance(metadata, dict):
            raise TypeError("Model metadata must be a dictionary.")

        if not isinstance(preprocessor, SentinelXV3Preprocessor):
            raise TypeError(
                "preprocessor must be a SentinelXV3Preprocessor."
            )

        required_metadata = {
            "sequence_length",
            "input_features",
            "feature_names",
        }

        missing = required_metadata - set(metadata)

        if missing:
            raise ValueError(
                f"Model metadata missing fields: {sorted(missing)}"
            )

        if metadata["sequence_length"] != 10:
            raise ValueError("SentinelX V3 requires sequence length 10.")

        if metadata["input_features"] != 209:
            raise ValueError("SentinelX V3 requires 209 input features.")

        if metadata["feature_names"] != preprocessor.feature_names:
            raise ValueError(
                "Model and preprocessor feature orders do not match."
            )

        if getattr(model, "input_size", None) != 209:
            raise ValueError("Model input size does not match V3.")

        if getattr(model, "expected_sequence_length", None) != 10:
            raise ValueError(
                "Model expected sequence length does not match V3."
            )

        self.model = model
        self.metadata = metadata
        self.preprocessor = preprocessor

        self.sequence_length = metadata["sequence_length"]
        self.feature_names = list(metadata["feature_names"])

    @classmethod
    def load(
        cls,
        checkpoint_path,
        data_dir="Datasets",
        device=None,
    ):
        model, metadata = load_world_model(
            checkpoint_path,
            device=device,
        )

        preprocessor = SentinelXV3Preprocessor.load(data_dir)

        return cls(model, metadata, preprocessor)

    def validate_sequence(self, raw_sequence):
        if not isinstance(raw_sequence, pd.DataFrame):
            raise TypeError(
                "raw_sequence must be a pandas DataFrame."
            )

        if raw_sequence.empty:
            raise ValueError("Sequence cannot be empty.")

        if not raw_sequence.columns.is_unique:
            raise ValueError(
                "Sequence contains duplicate column names."
            )

        required = {"Timestamp", "Segment ID"}

        missing = required - set(raw_sequence.columns)

        if missing:
            raise ValueError(
                f"Sequence metadata missing: {sorted(missing)}"
            )

        if raw_sequence["Segment ID"].isna().any():
            raise ValueError("Sequence contains missing Segment IDs.")

        if raw_sequence["Segment ID"].nunique() != 1:
            raise ValueError(
                "Sequence crosses a segment boundary."
            )

        try:
            timestamps = pd.to_datetime(
                raw_sequence["Timestamp"],
                errors="coerce",
            )
        except (TypeError, ValueError, OverflowError):
            raise ValueError(
                "Sequence contains invalid timestamps."
            )

        if timestamps.isna().any():
            raise ValueError(
                "Sequence contains invalid timestamps."
            )

        if not timestamps.is_monotonic_increasing:
            raise ValueError(
                "Sequence timestamps are not chronological."
            )

        differences = timestamps.diff().dropna()

        if not differences.eq(pd.Timedelta(seconds=1)).all():
            raise ValueError(
                "Every consecutive timestamp must be exactly "
                "one second apart."
            )

    def predict_raw_sequence(self, raw_sequence):
        """
        Predict from exactly 10 consecutive raw network states.

        Returns:
            pandas Series containing 209 predicted features.
        """

        if not isinstance(raw_sequence, pd.DataFrame):
            raise TypeError(
                "raw_sequence must be a pandas DataFrame."
            )

        if len(raw_sequence) != self.sequence_length:
            raise ValueError(
                f"Expected exactly {self.sequence_length} rows, "
                f"received {len(raw_sequence)}."
            )

        self.validate_sequence(raw_sequence)

        processed = self.preprocessor.transform_features(
            raw_sequence
        )

        prediction = predict_next_state(
            self.model,
            processed,
        )

        return pd.Series(
            prediction,
            index=self.feature_names,
            name="predicted_next_state",
        )
