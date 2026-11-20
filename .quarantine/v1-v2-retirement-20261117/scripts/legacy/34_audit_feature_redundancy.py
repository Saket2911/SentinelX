
from pathlib import Path
import json
import numpy as np
import pandas as pd

# ============================================================
# SENTINELX STEP 34
# FEATURE REDUNDANCY AND DATASET CONSISTENCY AUDIT
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Datasets"
REPORTS = ROOT / "reports"

REPORTS.mkdir(exist_ok=True)

FEATURE_FILE = DATA / "sentinelx_model_features_v1.json"

SPLITS = {
    "Train": {
        "X": DATA / "sentinelx_train_X_seq10_v1.npy",
        "y": DATA / "sentinelx_train_y_next_v1.npy",
        "metadata": DATA / "sentinelx_train_sequence_metadata_v1.csv",
    },
    "Validation": {
        "X": DATA / "sentinelx_validation_X_seq10_v1.npy",
        "y": DATA / "sentinelx_validation_y_next_v1.npy",
        "metadata": DATA / "sentinelx_validation_sequence_metadata_v1.csv",
    },
    "Test": {
        "X": DATA / "sentinelx_test_X_seq10_v1.npy",
        "y": DATA / "sentinelx_test_y_next_v1.npy",
        "metadata": DATA / "sentinelx_test_sequence_metadata_v1.csv",
    },
}

FEATURE_PAIRS = [
    ("CWE Flag Count__max", "Fwd URG Flags__max"),
    ("CWE Flag Count__std", "Fwd URG Flags__std"),
]

EXPECTED_FEATURES = 212
EXPECTED_SEQUENCE_LENGTH = 10


def load_features():
    features = json.loads(
        FEATURE_FILE.read_text(encoding="utf-8")
    )

    if len(features) != EXPECTED_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_FEATURES} features, "
            f"found {len(features)}"
        )

    if len(set(features)) != len(features):
        raise ValueError("Duplicate feature names in feature mapping.")

    return features


def compare_columns(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    exact = np.array_equal(a, b)

    close = np.allclose(
        a,
        b,
        rtol=1e-6,
        atol=1e-8,
        equal_nan=True,
    )

    max_difference = float(
        np.nanmax(np.abs(a.astype(np.float64) -
                         b.astype(np.float64)))
    )

    if np.std(a) == 0 or np.std(b) == 0:
        correlation = np.nan
    else:
        correlation = float(np.corrcoef(a, b)[0, 1])

    return {
        "exact": exact,
        "close": close,
        "max_difference": max_difference,
        "correlation": correlation,
    }


def audit_pair(split_name, X, y, features, pair):
    first, second = pair

    if first not in features or second not in features:
        print(f"\nMissing feature pair: {pair}")
        return

    i = features.index(first)
    j = features.index(second)

    print("\n" + "-" * 70)
    print(f"{split_name}: {first} VS {second}")
    print(f"Feature indices: {i} and {j}")

    # Compare next-state target values
    target_result = compare_columns(y[:, i], y[:, j])

    print("\nNEXT-STATE TARGET (y)")
    print(f"Exactly identical: {target_result['exact']}")
    print(f"Numerically equal: {target_result['close']}")
    print(f"Maximum absolute difference: "
          f"{target_result['max_difference']:.12f}")
    print(f"Correlation: {target_result['correlation']}")

    # Compare all input sequence timesteps
    x_first = X[:, :, i].reshape(-1)
    x_second = X[:, :, j].reshape(-1)

    input_result = compare_columns(x_first, x_second)

    print("\nINPUT SEQUENCES (X)")
    print(f"Exactly identical: {input_result['exact']}")
    print(f"Numerically equal: {input_result['close']}")
    print(f"Maximum absolute difference: "
          f"{input_result['max_difference']:.12f}")
    print(f"Correlation: {input_result['correlation']}")


def main():
    print("=" * 75)
    print("SENTINELX STEP 34")
    print("FEATURE REDUNDANCY AND DATASET CONSISTENCY AUDIT")
    print("=" * 75)

    features = load_features()

    print(f"\nFeature mapping loaded: {len(features)} features")

    report_lines = []

    for split_name, paths in SPLITS.items():

        print("\n" + "=" * 75)
        print(f"ANALYZING {split_name.upper()}")
        print("=" * 75)

        for key, path in paths.items():
            if not path.exists():
                raise FileNotFoundError(
                    f"Required {key} file not found: {path}"
                )

        X = np.load(paths["X"], mmap_mode="r")
        y = np.load(paths["y"], mmap_mode="r")
        metadata = pd.read_csv(paths["metadata"])

        print(f"\nX shape: {X.shape}")
        print(f"y shape: {y.shape}")
        print(f"Metadata shape: {metadata.shape}")

        # Structural checks
        checks = {
            "X is 3-dimensional": X.ndim == 3,
            "y is 2-dimensional": y.ndim == 2,
            "X feature count is correct": X.shape[2] == EXPECTED_FEATURES,
            "y feature count is correct": y.shape[1] == EXPECTED_FEATURES,
            "Sequence length is correct":
                X.shape[1] == EXPECTED_SEQUENCE_LENGTH,
            "X/y sample counts match": X.shape[0] == y.shape[0],
            "y/metadata sample counts match":
                y.shape[0] == len(metadata),
        }

        print("\nSTRUCTURAL VALIDATION")

        for description, passed in checks.items():
            status = "PASS" if passed else "FAIL"
            print(f"{status}: {description}")

        if not all(checks.values()):
            raise ValueError(
                f"{split_name} structural validation failed."
            )

        # Numerical checks
        print("\nNUMERICAL VALIDATION")

        x_finite = bool(np.isfinite(X).all())
        y_finite = bool(np.isfinite(y).all())

        print(f"X contains only finite values: {x_finite}")
        print(f"y contains only finite values: {y_finite}")

        if not x_finite or not y_finite:
            raise ValueError(
                f"Non-finite values found in {split_name}."
            )

        # Metadata labels
        if "Target Label" in metadata.columns:
            print("\nTARGET LABEL COUNTS")
            print(metadata["Target Label"].value_counts().to_string())

        # Segment counts
        if "Segment ID" in metadata.columns:
            print(
                "\nUnique segments:",
                metadata["Segment ID"].nunique()
            )

        # Feature redundancy checks
        for pair in FEATURE_PAIRS:
            audit_pair(
                split_name,
                X,
                y,
                features,
                pair,
            )

        report_lines.append({
            "Split": split_name,
            "X samples": X.shape[0],
            "X sequence length": X.shape[1],
            "X features": X.shape[2],
            "y samples": y.shape[0],
            "Metadata samples": len(metadata),
            "Metadata labels": (
                metadata["Target Label"].nunique()
                if "Target Label" in metadata.columns
                else None
            ),
            "Unique segments": (
                metadata["Segment ID"].nunique()
                if "Segment ID" in metadata.columns
                else None
            ),
        })

    # Summary
    summary = pd.DataFrame(report_lines)

    summary_path = REPORTS / "step34_dataset_consistency.csv"
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 75)
    print("STEP 34 COMPLETED")
    print("=" * 75)

    print("\nDATASET SUMMARY")
    print(summary.to_string(index=False))

    print(f"\nCSV report: {summary_path}")

    print("\nNo features were removed.")
    print("No sequence arrays were modified.")
    print("No model was retrained.")
    print("No checkpoint was modified.")


if __name__ == "__main__":
    main()

