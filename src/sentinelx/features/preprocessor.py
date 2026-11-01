
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


METADATA_COLUMNS = {"Timestamp", "Label"}


class NetworkFeaturePreprocessor:
    """
    Fit imputation and scaling on training data only.
    Reuse the fitted transformations on validation and test data.
    """

    def __init__(self):
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_fitted = False

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = METADATA_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"Required metadata columns missing: {sorted(missing)}"
            )

        features = df.drop(columns=list(METADATA_COLUMNS)).copy()

        # Reject unexpected non-numeric feature columns.
        non_numeric = features.select_dtypes(
            exclude=[np.number]
        ).columns.tolist()

        if non_numeric:
            raise ValueError(
                f"Non-numeric feature columns found: {non_numeric}"
            )

        features = features.replace([np.inf, -np.inf], np.nan)

        # Convert any remaining numeric values to float64.
        return features.astype("float64")

    def fit_transform(self, train_df: pd.DataFrame) -> np.ndarray:
        features = self._extract_features(train_df)

        if features.empty:
            raise ValueError("Training data has no feature columns.")

        self.feature_columns = features.columns.tolist()

        imputed = self.imputer.fit_transform(features)
        scaled = self.scaler.fit_transform(imputed)

        self.is_fitted = True
        return scaled.astype(np.float32)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(
                "Preprocessor must be fitted on training data first."
            )

        features = self._extract_features(df)

        if features.columns.tolist() != self.feature_columns:
            raise ValueError(
                "Feature columns or their order differ from training data."
            )

        imputed = self.imputer.transform(features)
        scaled = self.scaler.transform(imputed)

        return scaled.astype(np.float32)
