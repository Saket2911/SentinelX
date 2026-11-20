
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sentinelx.data.loader import load_dataset
from sentinelx.data.splitter import split_dataset
from sentinelx.features.sequences import (
    build_network_states,
    make_sequences,
)


DATASET_PATH = (
    PROJECT_ROOT
    / "Datasets"
    / "Wednesday-14-02-2018_prepared.csv"
)


def main():
    df = load_dataset(DATASET_PATH)

    # Keep the chronological split.
    train_df, validation_df, test_df = split_dataset(df)

    # Build states separately so sequences cannot cross split boundaries.
    train_states = build_network_states(train_df, window="1s")
    validation_states = build_network_states(
        validation_df, window="1s"
    )
    test_states = build_network_states(test_df, window="1s")

    print("NETWORK STATE SUMMARY")
    for name, states in [
        ("TRAIN", train_states),
        ("VALIDATION", validation_states),
        ("TEST", test_states),
    ]:
        print(f"\n{name}")
        print("Number of states:", len(states))
        print("Columns:", len(states.columns))
        print("Time range:",
              states["Timestamp"].min(),
              "to",
              states["Timestamp"].max())
        print("Empty-flow windows:",
              int((states["Flow Count"] == 0).sum()))
        print("Window labels:")
        print(states["Label"].value_counts().to_string())

    # Create sequences independently for each split.
    sequence_length = 10

    for name, states in [
        ("TRAIN", train_states),
        ("VALIDATION", validation_states),
        ("TEST", test_states),
    ]:
        X, y, timestamps, labels, feature_columns = make_sequences(
            states,
            sequence_length=sequence_length,
        )

        print(f"\n{name} SEQUENCES")
        print("X shape:", X.shape)
        print("y shape:", y.shape)
        print("Target timestamps:", len(timestamps))
        print("Target labels:", len(labels))
        print("All X values finite:", np.isfinite(X).all())
        print("All y values finite:", np.isfinite(y).all())
        print("First target timestamp:", timestamps[0])
        print("First target label:", labels[0])

    print("\nSequence construction test completed.")


if __name__ == "__main__":
    main()
