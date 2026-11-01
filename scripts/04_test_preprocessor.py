
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sentinelx.data.loader import load_dataset
from sentinelx.data.splitter import split_dataset
from sentinelx.features.preprocessor import NetworkFeaturePreprocessor


DATASET_PATH = (
    PROJECT_ROOT
    / "Datasets"
    / "Wednesday-14-02-2018_prepared.csv"
)


def main():
    df = load_dataset(DATASET_PATH)

    train_df, validation_df, test_df = split_dataset(df)

    preprocessor = NetworkFeaturePreprocessor()

    # Fit ONLY on training data.
    X_train = preprocessor.fit_transform(train_df)
    X_validation = preprocessor.transform(validation_df)
    X_test = preprocessor.transform(test_df)

    for name, X in [
        ("TRAIN", X_train),
        ("VALIDATION", X_validation),
        ("TEST", X_test),
    ]:
        print(f"\n{name}")
        print("Shape:", X.shape)
        print("Data type:", X.dtype)
        print("All values finite:", np.isfinite(X).all())
        print("Missing values:", np.isnan(X).sum())

    print("\nNumber of features:", len(preprocessor.feature_columns))
    print("First 5 features:", preprocessor.feature_columns[:5])


if __name__ == "__main__":
    main()
