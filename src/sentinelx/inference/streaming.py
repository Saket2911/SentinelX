from collections import deque

import pandas as pd


class SentinelXStreamingPredictor:
    """
    Offline/online streaming wrapper around the SentinelX V3 predictor.

    Accepts one timestamped network state at a time.
    Predicts the next state using the latest 10 consecutive observations.
    """

    def __init__(self, predictor):
        if not callable(getattr(predictor, "predict_raw_sequence", None)):
            raise TypeError(
                "predictor must provide predict_raw_sequence()."
            )

        self.predictor = predictor
        self.feature_names = list(predictor.feature_names)
        self.sequence_length = predictor.sequence_length

        if len(self.feature_names) != 209:
            raise ValueError("Expected exactly 209 model features.")

        if self.sequence_length != 10:
            raise ValueError("SentinelX V3 requires a 10-state window.")

        self.buffer = deque(maxlen=self.sequence_length)
        self.current_segment_id = None
        self.last_timestamp = None

    @classmethod
    def load(
        cls,
        checkpoint_path,
        data_dir="Datasets",
        device="cpu",
    ):
        from sentinelx.inference.pipeline import SentinelXV3Predictor

        predictor = SentinelXV3Predictor.load(
            checkpoint_path,
            data_dir=data_dir,
            device=device,
        )

        return cls(predictor)

    def reset(self):
        """Clear buffered observations and temporal tracking."""
        self.buffer.clear()
        self.current_segment_id = None
        self.last_timestamp = None

    def _to_dataframe(self, state):
        if isinstance(state, pd.DataFrame):
            if len(state) != 1:
                raise ValueError(
                    "Each streaming call must contain exactly one row."
                )
            row = state.copy()

        elif isinstance(state, pd.Series):
            if not state.index.is_unique:
                raise ValueError("Duplicate input column names.")
            row = pd.DataFrame([state.to_dict()])

        elif isinstance(state, dict):
            row = pd.DataFrame([state])

        else:
            raise TypeError(
                "State must be a pandas DataFrame, Series, or dictionary."
            )

        if not row.columns.is_unique:
            raise ValueError("Duplicate input column names.")

        return row

    def process_state(self, state):
        """
        Process one incoming network-state observation.

        Returns:
            WARMING_UP:
                Fewer than 10 consecutive states are buffered.

            PREDICTION_READY:
                A prediction for the next one-second state is available.
        """

        row = self._to_dataframe(state)

        required_metadata = ["Timestamp", "Segment ID"]

        for column in required_metadata:
            if column not in row.columns:
                raise ValueError(f"Missing required column: {column}")

        missing_features = [
            name for name in self.feature_names
            if name not in row.columns
        ]

        if missing_features:
            raise ValueError(
                f"Missing model features: {missing_features}"
            )

        raw_timestamp = row["Timestamp"].iloc[0]
        timestamp = pd.to_datetime(
            raw_timestamp,
            errors="coerce",
        )

        if pd.isna(timestamp):
            raise ValueError("Invalid timestamp.")

        timestamp = pd.Timestamp(timestamp)

        segment_id = row["Segment ID"].iloc[0]

        if pd.isna(segment_id):
            raise ValueError("Segment ID cannot be missing.")

        reset_reason = None

        # A new segment starts a fresh sequence.
        if self.current_segment_id is not None:
            if segment_id != self.current_segment_id:
                self.reset()
                reset_reason = "segment_change"

            else:
                difference = timestamp - self.last_timestamp

                if difference <= pd.Timedelta(0):
                    raise ValueError(
                        "Duplicate or out-of-order timestamp."
                    )

                if difference != pd.Timedelta(seconds=1):
                    self.reset()
                    reset_reason = "timestamp_gap"

        self.current_segment_id = segment_id
        self.last_timestamp = timestamp

        # Store a copy so caller-side modifications cannot alter history.
        self.buffer.append(row.iloc[0].copy())

        if len(self.buffer) < self.sequence_length:
            return {
                "status": "WARMING_UP",
                "buffer_size": len(self.buffer),
                "required": self.sequence_length,
                "timestamp": timestamp,
                "reset_reason": reset_reason,
                "prediction": None,
                "predicted_timestamp": None,
            }

        sequence = pd.DataFrame(list(self.buffer))

        prediction = self.predictor.predict_raw_sequence(sequence)

        return {
            "status": "PREDICTION_READY",
            "buffer_size": len(self.buffer),
            "required": self.sequence_length,
            "timestamp": timestamp,
            "reset_reason": reset_reason,
            "prediction": prediction,
            "predicted_timestamp": (
                timestamp + pd.Timedelta(seconds=1)
            ),
        }
