
from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# SENTINELX STEP 33 — FEATURE VALUE DISTRIBUTION AUDIT
# ============================================================

ROOT = Path(r"F:\Projects\SentinelX")
DATA = ROOT / "Datasets"
REPORTS = ROOT / "reports"

REPORTS.mkdir(parents=True, exist_ok=True)

FEATURES_FILE = DATA / "sentinelx_model_features_v1.json"

CONFIG = {
    "Validation": {
        "y": DATA / "sentinelx_validation_y_next_v1.npy",
        "metadata": DATA / "sentinelx_validation_sequence_metadata_v1.csv",
        "target": "Infiltration",
        "features": [
            "CWE Flag Count__max",
            "Fwd URG Flags__max",
            "CWE Flag Count__std",
            "Fwd URG Flags__std",
        ],
    },
    "Test": {
        "y": DATA / "sentinelx_test_y_next_v1.npy",
        "metadata": DATA / "sentinelx_test_sequence_metadata_v1.csv",
        "target": "Bot",
        "features": [
            "Bwd Pkts/s__mean",
            "Bwd Pkts/s__std",
            "Flow Byts/s__mean",
            "Flow Byts/s__std",
        ],
    },
}

# ------------------------------------------------------------
# 1. Load feature names
# ------------------------------------------------------------

import json

feature_names = json.loads(
    FEATURES_FILE.read_text(encoding="utf-8")
)

if len(feature_names) != 212:
    raise ValueError(
        f"Expected 212 feature names, found {len(feature_names)}"
    )

if len(set(feature_names)) != len(feature_names):
    raise ValueError("Duplicate feature names detected.")

print(f"Features loaded: {len(feature_names)}")
print("Feature mapping source: sentinelx_model_features_v1.json")

print("=" * 75)
print("SENTINELX STEP 33 — FEATURE VALUE DISTRIBUTION AUDIT")
print("=" * 75)

print(f"Features loaded: {len(feature_names)}")

results = []


# ------------------------------------------------------------
# 2. Helper: identify actual label spelling
# ------------------------------------------------------------

def normalize_label(value):
    return "".join(
        char.lower()
        for char in str(value)
        if char.isalnum()
    )


def find_label(labels, requested):
    aliases = {
        "infiltration": {"infiltration", "infilteration"},
        "infilteration": {"infiltration", "infilteration"},
    }

    target = normalize_label(requested)
    accepted_names = aliases.get(target, {target})

    matches = [
        label for label in labels
        if normalize_label(label) in accepted_names
    ]

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise ValueError(
            f"Multiple matching labels for '{requested}': {matches}"
        )

    raise ValueError(
        f"Could not find label '{requested}'. "
        f"Available labels: {labels}"
    )


# ------------------------------------------------------------
# 3. Audit one feature for one label
# ------------------------------------------------------------

def audit_feature(values, feature, label, split):

    values = np.asarray(values, dtype=np.float64)

    finite_mask = np.isfinite(values)
    finite = values[finite_mask]

    if len(finite) == 0:
        raise ValueError(
            f"No finite values for {feature}, {label}"
        )

    zero_count = np.count_nonzero(
        np.abs(finite) <= 1e-12
    )

    near_zero_count = np.count_nonzero(
        np.abs(finite) <= 1e-6
    )

    unique_count = np.unique(finite).size

    return {
        "Split": split,
        "Label": label,
        "Feature": feature,
        "Samples": len(values),
        "Finite Samples": len(finite),
        "Nonfinite Samples": len(values) - len(finite),
        "Zero Percentage Model Space": (
            zero_count / len(finite)
        ) * 100,
        "Near Zero Percentage Model Space": (
            near_zero_count / len(finite)
        ) * 100,
        "Mean": np.mean(finite),
        "Std": np.std(finite),
        "Minimum": np.min(finite),
        "Maximum": np.max(finite),
        "Unique Values": unique_count,
        "Constant": unique_count == 1,
        "Nearly Constant": np.std(finite) < 1e-8,
    }


# ------------------------------------------------------------
# 4. Process validation and test separately
# ------------------------------------------------------------

for split, config in CONFIG.items():

    print(f"\n{'=' * 75}")
    print(f"ANALYZING {split.upper()} DATA")
    print("=" * 75)

    y = np.load(config["y"], mmap_mode="r")
    metadata = pd.read_csv(config["metadata"])

    print(f"Target array shape: {y.shape}")
    print(f"Metadata shape: {metadata.shape}")

    # Alignment checks
    if y.ndim != 2:
        raise ValueError(f"Expected 2D y array, got {y.shape}")

    if y.shape[0] != len(metadata):
        raise ValueError("Target/metadata sample mismatch")

    if y.shape[1] != len(feature_names):
        raise ValueError(
            f"Feature mismatch: y has {y.shape[1]} "
            f"features, names file has {len(feature_names)}"
        )

    labels = metadata["Target Label"].dropna().unique().tolist()

    print("\nAvailable labels:")
    for label in labels:
        print(f"  {label}")

    benign_label = find_label(labels, "Benign")
    target_label = find_label(labels, config["target"])

    print(f"\nComparing: {benign_label} vs {target_label}")

    selected_labels = [benign_label, target_label]

    for feature in config["features"]:

        if feature not in feature_names:
            raise ValueError(
                f"Feature not found: {feature}"
            )

        feature_index = feature_names.index(feature)

        print(f"\nFeature: {feature}")
        print(f"Feature index: {feature_index}")

        for label in selected_labels:

            mask = (
                metadata["Target Label"].astype(str).to_numpy()
                == str(label)
            )

            values = np.asarray(y[mask, feature_index])

            result = audit_feature(
                values,
                feature,
                label,
                split
            )

            results.append(result)

            print(f"\n  Label: {label}")
            print(f"  Samples: {result['Samples']:,}")
            print(
                f"  Zero percentage (model space): "
                f"{result['Zero Percentage Model Space']:.4f}%"
            )
            print(
                f"  Near-zero percentage: "
                f"{result['Near Zero Percentage Model Space']:.4f}%"
            )
            print(f"  Mean: {result['Mean']:.6f}")
            print(f"  Std: {result['Std']:.6f}")
            print(
                f"  Range: [{result['Minimum']:.6f}, "
                f"{result['Maximum']:.6f}]"
            )
            print(f"  Unique values: {result['Unique Values']:,}")
            print(f"  Constant: {result['Constant']}")
            print(f"  Nearly constant: {result['Nearly Constant']}")


# ------------------------------------------------------------
# 5. Save results
# ------------------------------------------------------------

report = pd.DataFrame(results)

csv_path = (
    REPORTS /
    "step33_feature_value_distribution_audit.csv"
)

txt_path = (
    REPORTS /
    "step33_feature_value_distribution_audit.txt"
)

report.to_csv(csv_path, index=False)

with open(txt_path, "w", encoding="utf-8") as file:

    file.write(
        "SENTINELX STEP 33 — FEATURE VALUE DISTRIBUTION AUDIT\n"
    )

    file.write("=" * 75 + "\n\n")

    file.write(
        "Note: Zero percentages refer to standardized model-space "
        "values, not necessarily original raw traffic zeros.\n\n"
    )

    file.write(report.to_string(index=False))

print("\n" + "=" * 75)
print("STEP 33 COMPLETED")
print("=" * 75)

print(f"\nCSV report: {csv_path}")
print(f"Text report: {txt_path}")

print("\nNo model was retrained.")
print("No checkpoint was modified.")
print("No sequence arrays were modified.")
