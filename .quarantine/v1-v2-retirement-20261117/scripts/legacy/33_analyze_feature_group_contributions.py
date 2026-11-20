from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(r"F:\Projects\SentinelX")
DATA = ROOT / "Datasets"
MODEL = ROOT / "Models" / "lstm_world_model_v1"

JOINED = MODEL / "attack_ratio_error_analysis_v1"
OUTPUT = MODEL / "feature_group_analysis_v1"
OUTPUT.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "validation": {
        "features": DATA / "sentinelx_validation_preprocessed.csv",
        "joined": JOINED / "validation_joined_states.csv",
        "label": "label"
    },
    "test": {
        "features": DATA / "sentinelx_test_preprocessed.csv",
        "joined": JOINED / "test_joined_states.csv",
        "label": "actual_label"
    }
}


def classify_feature(feature):

    name = feature.replace("__std", "").lower()

    # 1. Timing behaviour
    if name.startswith((
        "flow duration",
        "flow iat",
        "fwd iat",
        "bwd iat",
        "active ",
        "idle "
    )):
        return "Timing Behaviour"

    # 2. Packet volume
    if name.startswith((
        "tot fwd pkts",
        "tot bwd pkts",
        "subflow fwd pkts",
        "subflow bwd pkts",
        "fwd act data pkts"
    )):
        return "Packet Volume"

    # 3. Packet size and bytes
    if name.startswith((
        "totlen fwd",
        "totlen bwd",
        "subflow fwd byts",
        "subflow bwd byts",
        "fwd pkt len",
        "bwd pkt len",
        "pkt len",
        "pkt size avg",
        "fwd seg size avg",
        "bwd seg size avg",
        "fwd seg size min"
    )):
        return "Packet Size and Bytes"

    # 4. Directional behaviour and traffic rates
    if name.startswith((
        "down/up ratio",
        "fwd header len",
        "bwd header len",
        "fwd pkts/s",
        "bwd pkts/s",
        "flow byts/s",
        "flow pkts/s"
    )):
        return "Directional Behaviour and Rates"

    # 5. Protocol and control characteristics
    return "Protocol and Control"


def residualize(values, controls):

    matrix = np.column_stack([
        np.ones(len(controls)),
        controls.to_numpy(dtype=float)
    ])

    coefficients = np.linalg.lstsq(
        matrix,
        values.to_numpy(dtype=float),
        rcond=None
    )[0]

    return values.to_numpy(dtype=float) - matrix @ coefficients


def analyze(name, config):

    print(f"\nProcessing {name}...")

    features = pd.read_csv(config["features"])
    states = pd.read_csv(config["joined"])

    std_features = [
        c for c in features.columns if c.endswith("__std")
    ]

    if len(std_features) != 68:
        raise ValueError(
            f"Expected 68 variability features, found {len(std_features)}"
        )

    # Assign every variability feature to one group.
    groups = {}

    for feature in std_features:
        group = classify_feature(feature)
        groups.setdefault(group, []).append(feature)

    print("\nFeature group membership:")

    membership = []

    for group, members in groups.items():
        print(f"\n{group}: {len(members)} features")

        for feature in members:
            print(f"  {feature}")
            membership.append({
                "Group": group,
                "Feature": feature
            })

    pd.DataFrame(membership).to_csv(
        OUTPUT / f"{name}_feature_group_membership.csv",
        index=False
    )

    features["Timestamp"] = pd.to_datetime(
        features["Timestamp"], errors="coerce"
    )

    states["timestamp"] = pd.to_datetime(
        states["timestamp"], errors="coerce"
    )

    merged = states.merge(
        features[["Timestamp"] + std_features],
        left_on="timestamp",
        right_on="Timestamp",
        how="left",
        validate="one_to_one"
    )

    if merged[std_features].isna().any().any():
        raise ValueError("Missing feature values after timestamp matching")

    label_col = config["label"]

    required = [
        "Prediction MAE",
        "Flow Count",
        "Attack Flow Ratio",
        label_col
    ]

    for column in required:
        if column not in merged.columns:
            raise ValueError(f"Missing required column: {column}")

    merged = merged[
        merged[label_col].isin(["Benign", "Bot"])
    ].copy()

    for column in [
        "Prediction MAE",
        "Flow Count",
        "Attack Flow Ratio"
    ]:
        merged[column] = pd.to_numeric(
            merged[column], errors="coerce"
        )

    merged = merged.dropna(
        subset=[
            "Prediction MAE",
            "Flow Count",
            "Attack Flow Ratio"
        ]
    ).copy()

    merged["Label Binary"] = (
        merged[label_col] == "Bot"
    ).astype(float)

    merged["Log Flow Count"] = np.log1p(
        merged["Flow Count"]
    )

    # Calculate one index for each feature group.
    for group, members in groups.items():

        merged[group] = merged[members].abs().mean(axis=1)

    controls = pd.DataFrame({
        "Flow_Count": merged["Log Flow Count"].rank(),
        "Attack_Ratio": merged["Attack Flow Ratio"].rank(),
        "Label": merged["Label Binary"]
    })

    ranked_error = merged["Prediction MAE"].rank()

    records = []

    for group in groups:

        group_index = merged[group]

        ranked_group = group_index.rank()

        raw_spearman = group_index.corr(
            merged["Prediction MAE"],
            method="spearman"
        )

        x_residual = residualize(
            ranked_group,
            controls
        )

        y_residual = residualize(
            ranked_error,
            controls
        )

        if np.std(x_residual) == 0 or np.std(y_residual) == 0:
            partial_corr = np.nan
        else:
            partial_corr = np.corrcoef(
                x_residual,
                y_residual
            )[0, 1]

        records.append({
            "Feature_Group": group,
            "Feature_Count": len(groups[group]),
            "Mean_Group_Variability": group_index.mean(),
            "Median_Group_Variability": group_index.median(),
            "Raw_Spearman_With_MAE": raw_spearman,
            "Controlled_Partial_Spearman": partial_corr
        })

    result = pd.DataFrame(records)

    result["Absolute_Controlled_Correlation"] = (
        result["Controlled_Partial_Spearman"].abs()
    )

    result = result.sort_values(
        "Absolute_Controlled_Correlation",
        ascending=False
    )

    result.to_csv(
        OUTPUT / f"{name}_group_correlations.csv",
        index=False
    )

    # Save state-level group indices for further analysis.
    state_output = merged[
        [
            "timestamp",
            label_col,
            "Prediction MAE",
            "Flow Count",
            "Attack Flow Ratio"
        ] + list(groups.keys())
    ]

    state_output.to_csv(
        OUTPUT / f"{name}_state_group_indices.csv",
        index=False
    )

    print(f"\nStates analysed: {len(merged):,}")

    print("\nFeature group correlation results:")
    print(result.to_string(index=False))

    return result


def main():

    print("=" * 70)
    print("SENTINELX - FEATURE GROUP CONTRIBUTION ANALYSIS")
    print("=" * 70)

    results = {}

    for name, config in CONFIG.items():

        for path in config.values():
            if isinstance(path, Path) and not path.exists():
                raise FileNotFoundError(path)

        results[name] = analyze(name, config)

    comparison = results["validation"].merge(
        results["test"],
        on="Feature_Group",
        suffixes=("_Validation", "_Test"),
        validate="one_to_one"
    )

    comparison.to_csv(
        OUTPUT / "validation_test_group_comparison.csv",
        index=False
    )

    print("\nValidation vs Test comparison:")

    print(
        comparison[[
            "Feature_Group",
            "Controlled_Partial_Spearman_Validation",
            "Controlled_Partial_Spearman_Test"
        ]].to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print(f"Output folder: {OUTPUT}")
    print("No model or original dataset files were modified.")


if __name__ == "__main__":
    main()
