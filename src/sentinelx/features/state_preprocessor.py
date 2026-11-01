
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


METADATA_COLUMNS = {
    "Timestamp",
    "CaptureDay",
    "Label",
    "Segment ID",
}


class StatePreprocessor:
    """
    Impute, log-transform highly skewed nonnegative features,
    and standardize network-state features.

    All fitted statistics and transformation selections
    are learned exclusively from training states.
    """

    def __init__(self, skewness_threshold: float = 2.0):

        if skewness_threshold < 0:
            raise ValueError(
                "skewness_threshold cannot be negative."
            )

        self.skewness_threshold = skewness_threshold

        self.imputer = SimpleImputer(
            strategy="median",
            keep_empty_features=True,
        )

        self.scaler = StandardScaler()

        self.feature_columns = None
        self.log_columns = []
        self.is_fitted = False

    def _get_features(
        self,
        states: pd.DataFrame,
    ) -> pd.DataFrame:

        features = states.drop(
            columns=[
                col for col in METADATA_COLUMNS
                if col in states.columns
            ]
        ).copy()

        if features.empty:
            raise ValueError(
                "No numerical state features found."
            )

        non_numeric = features.select_dtypes(
            exclude=[np.number]
        ).columns.tolist()

        if non_numeric:
            raise ValueError(
                f"Non-numeric state features found: {non_numeric}"
            )

        return (
            features
            .replace([np.inf, -np.inf], np.nan)
            .astype(np.float64)
        )

    def fit_transform(
        self,
        train_states: pd.DataFrame,
    ) -> pd.DataFrame:

        if self.is_fitted:
            raise RuntimeError(
                "This preprocessor is already fitted. "
                "Create a new instance to fit on another training set."
            )

        features = self._get_features(train_states)

        self.feature_columns = features.columns.tolist()

        # Fit median imputation using training data only.
        imputed = self.imputer.fit_transform(features)

        imputed_df = pd.DataFrame(
            imputed,
            columns=self.feature_columns,
            index=features.index,
        )

        # Identify log1p features using training data only.
        self.log_columns = []

        for column in self.feature_columns:

            values = imputed_df[column].to_numpy()

            # log1p is valid only for nonnegative values.
            if np.any(values < 0):
                continue

            skewness = pd.Series(values).skew()

            if (
                np.isfinite(skewness)
                and skewness > self.skewness_threshold
            ):
                self.log_columns.append(column)

        transformed = self._apply_log_transform(
            imputed_df
        )

        # Fit the scaler using training data only.
        scaled = self.scaler.fit_transform(transformed)

        if not np.isfinite(scaled).all():
            raise ValueError(
                "Non-finite values found after training preprocessing."
            )

        self.is_fitted = True

        return self._restore_metadata(
            train_states,
            scaled,
            self.feature_columns,
        )

    def transform(
        self,
        states: pd.DataFrame,
    ) -> pd.DataFrame:

        if not self.is_fitted:
            raise RuntimeError(
                "Fit the preprocessor on training states first."
            )

        features = self._get_features(states)

        if features.columns.tolist() != self.feature_columns:
            raise ValueError(
                "State feature columns differ from training data."
            )

        # Reuse training imputation statistics.
        imputed = self.imputer.transform(features)

        imputed_df = pd.DataFrame(
            imputed,
            columns=self.feature_columns,
            index=features.index,
        )

        # Reuse training log-transform selections.
        transformed = self._apply_log_transform(
            imputed_df
        )

        # Reuse the training scaler.
        scaled = self.scaler.transform(transformed)

        if not np.isfinite(scaled).all():
            raise ValueError(
                "Non-finite values found after preprocessing."
            )

        return self._restore_metadata(
            states,
            scaled,
            self.feature_columns,
        )

    def _apply_log_transform(
        self,
        features: pd.DataFrame,
    ) -> pd.DataFrame:

        result = features.copy()

        for column in self.log_columns:

            values = result[column].to_numpy()

            if np.any(values < 0):
                raise ValueError(
                    f"Negative values found in log1p feature: {column}"
                )

            result[column] = np.log1p(values)

        return result

    @staticmethod
    def _restore_metadata(
        states: pd.DataFrame,
        scaled: np.ndarray,
        feature_columns: list,
    ) -> pd.DataFrame:

        # Preserve timestamps, labels, capture days,
        # segment IDs, and any other non-feature columns.
        metadata = states.drop(
            columns=feature_columns
        ).copy()

        scaled_df = pd.DataFrame(
            scaled.astype(np.float32),
            columns=feature_columns,
            index=states.index,
        )

        # Combine in one operation to reduce fragmentation.
        return pd.concat(
            [metadata, scaled_df],
            axis=1,
        )
