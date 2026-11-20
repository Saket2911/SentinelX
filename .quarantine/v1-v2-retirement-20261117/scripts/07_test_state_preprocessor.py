
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sentinelx.data.loader import load_dataset
from sentinelx.data.splitter import split_dataset
from sentinelx.features.sequences import build_network_states
from sentinelx.features.state_preprocessor import StatePreprocessor


DATASET_PATH = (
    PROJECT_ROOT
    / "Datasets"
    / "Wednesday-14-02-2018_prepared.csv"
)


def main():
    df = load_dataset(DATASET_PATH)
    train_df, validation_df, test_df = split_dataset(df)

    train_states = build_network_states(train_df)
    validation_states = build_network_states(validation_df)
    test_states = build_network_states(test_df)

    preprocessor = StatePreprocessor()

    train_scaled = preprocessor.fit_transform(train_states)
    validation_scaled = preprocessor.transform(validation_states)
    test_scaled = preprocessor.transform(test_states)

    for name, states in [
        ("TRAIN", train_scaled),
        ("VALIDATION", validation_scaled),
        ("TEST", test_scaled),
    ]:
        features = states[preprocessor.feature_columns].to_numpy()

        print(f"\n{name}")
        print("States:", len(states))
        print("Feature shape:", features.shape)
        print("All values finite:", np.isfinite(features).all())
        print("Missing values:", np.isnan(features).sum())
        print("Segment IDs preserved:", "Segment ID" in states.columns)
        print("Labels preserved:", "Label" in states.columns)

    print("\nState preprocessing test completed.")


if __name__ == "__main__":
    main()
