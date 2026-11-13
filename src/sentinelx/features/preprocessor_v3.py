import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class SentinelXV3Preprocessor:
    """
    Reusable inference preprocessor for SentinelX V3.

    Uses saved training-only:
      - Feature order
      - Median imputation values
      - Signed-log transformation
      - StandardScaler

    This class never fits or updates preprocessing parameters.
    """

    METADATA_COLUMNS = [
        "Timestamp",
        "CaptureDay",
        "Label",
        "Segment ID",
    ]

    def __init__(
        self,
        feature_names,
        medians,
        scaler,
    ):
        self.feature_names = list(feature_names)
        self.medians = np.asarray(medians, dtype=np.float64)
        self.scaler = scaler

        n_features = len(self.feature_names)

        if n_features != 209:
            raise ValueError(
                f"Expected 209 V3 features, received {n_features}."
            )

        if not all(
            isinstance(name, str) and name
            for name in self.feature_names
        ):
            raise ValueError("Invalid V3 feature names.")

        if len(set(self.feature_names)) != n_features:
            raise ValueError("Duplicate feature names detected.")

        if self.medians.shape != (n_features,):
            raise ValueError("Median array has an invalid shape.")

        if not np.isfinite(self.medians).all():
            raise ValueError("Saved medians contain non-finite values.")

        if not callable(getattr(self.scaler, "transform", None)):
            raise ValueError("Scaler does not provide a transform method.")

        if getattr(self.scaler, "n_features_in_", None) != n_features:
            raise ValueError(
                "Scaler feature count does not match V3 feature count."
            )

        # Validate saved scaler parameters without modifying them.
        for attribute in ("mean_", "scale_"):
            parameter = getattr(self.scaler, attribute, None)

            if parameter is None:
                raise ValueError(
                    f"Scaler is missing fitted parameter: {attribute}"
                )

            parameter = np.asarray(parameter)

            if parameter.shape != (n_features,):
                raise ValueError(
                    f"Scaler parameter {attribute} has an invalid shape."
                )

            if not np.isfinite(parameter).all():
                raise ValueError(
                    f"Scaler parameter {attribute} contains non-finite values."
                )

        if np.any(np.asarray(self.scaler.scale_) <= 0):
            raise ValueError("Scaler contains non-positive scale values.")

        # If feature names were saved by the scaler, verify their order.
        scaler_names = getattr(
            self.scaler,
            "feature_names_in_",
            None,
        )

        if scaler_names is not None:
            if list(scaler_names) != self.feature_names:
                raise ValueError(
                    "Scaler feature order does not match V3 feature order."
                )

    @classmethod
    def load(
        cls,
        data_dir="Datasets",
    ):
        data_dir = Path(data_dir)

        feature_path = (
            data_dir / "sentinelx_model_features_v3.json"
        )
        median_path = (
            data_dir / "sentinelx_train_medians_v3.npz"
        )
        scaler_path = (
            data_dir / "sentinelx_standard_scaler_v3.joblib"
        )

        for path in [feature_path, median_path, scaler_path]:
            if not path.is_file():
                raise FileNotFoundError(
                    f"Required V3 preprocessing artifact missing: {path}"
                )

        with feature_path.open("r", encoding="utf-8") as f:
            feature_names = json.load(f)

        if not isinstance(feature_names, list):
            raise ValueError(
                "V3 feature JSON must contain a list."
            )

        if not all(
            isinstance(name, str) and name
            for name in feature_names
        ):
            raise ValueError(
                "V3 feature JSON contains invalid feature names."
            )

        with np.load(median_path, allow_pickle=False) as archive:
            required = {"medians", "feature_names"}

            if not required.issubset(archive.files):
                raise ValueError(
                    "Median artifact has unexpected contents."
                )

            medians = archive["medians"].copy()
            saved_names = archive["feature_names"].tolist()

        if feature_names != saved_names:
            raise ValueError(
                "Feature JSON and median artifact order do not match."
            )

        # Only load trusted local preprocessing artifacts.
        scaler = joblib.load(scaler_path)

        return cls(
            feature_names=feature_names,
            medians=medians,
            scaler=scaler,
        )

    def transform_features(self, frame):
        """
        Transform a DataFrame containing the 209 raw V3 state features.

        Returns:
            float32 NumPy array with shape (rows, 209).
        """

        if not isinstance(frame, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        if frame.empty:
            raise ValueError("Input DataFrame cannot be empty.")

        if not frame.columns.is_unique:
            raise ValueError(
                "Input DataFrame contains duplicate column names."
            )

        missing = [
            name for name in self.feature_names
            if name not in frame.columns
        ]

        if missing:
            raise ValueError(
                f"Input is missing V3 features: {missing[:10]}"
            )

        # Enforce exact checkpoint feature order.
        features = frame.loc[:, self.feature_names].copy()

        # Match the V3 preprocessing implementation.
        features = features.apply(
            pd.to_numeric,
            errors="coerce",
        )

        features.replace(
            [np.inf, -np.inf],
            np.nan,
            inplace=True,
        )

        features = features.fillna(
            pd.Series(
                self.medians,
                index=self.feature_names,
            )
        )

        values = features.to_numpy(dtype=np.float32)

        # Signed log transformation on every feature.
        values = np.sign(values) * np.log1p(np.abs(values))

        # Apply the already-fitted training scaler.
        values = self.scaler.transform(values).astype(np.float32)

        if not np.isfinite(values).all():
            raise RuntimeError(
                "Non-finite values produced during V3 preprocessing."
            )

        return values

    def transform(self, frame, include_metadata=True):
        """
        Return a DataFrame with metadata followed by V3 features.
        """

        values = self.transform_features(frame)

        result = pd.DataFrame(
            values,
            columns=self.feature_names,
            index=frame.index,
        )

        if include_metadata:
            metadata = [
                column for column in self.METADATA_COLUMNS
                if column in frame.columns
            ]

            if metadata:
                result = pd.concat(
                    [frame.loc[:, metadata].copy(), result],
                    axis=1,
                )

        return result
