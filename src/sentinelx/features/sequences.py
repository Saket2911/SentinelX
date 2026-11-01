
import numpy as np
import pandas as pd


# Metadata must never become numerical model inputs.
METADATA_COLUMNS = {
    "Timestamp",
    "Label",
    "CaptureDay",
    "Segment ID",
}

# Verified constant across the complete 3,145,720-row dataset.
CONSTANT_FEATURES = {
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "CWE Flag Count",
    "Fwd Byts/b Avg",
    "Fwd Pkts/b Avg",
    "Fwd Blk Rate Avg",
    "Bwd Byts/b Avg",
    "Bwd Pkts/b Avg",
    "Bwd Blk Rate Avg",
}


def build_network_states(
    df: pd.DataFrame,
    window: str = "1s",
    max_gap_seconds: int = 8,
) -> pd.DataFrame:
    """
    Aggregate network-flow records into timestamped network states.

    - Excludes verified constant features.
    - Preserves CaptureDay as metadata.
    - Converts infinite values to NaN.
    - Preserves missing measurements in observed windows.
    - Uses zero-valued features only for genuinely empty windows.
    - Creates independent segments across long temporal gaps.
    """

    if df.empty:
        raise ValueError("Cannot build states from an empty dataset.")

    if not {"Timestamp", "Label"}.issubset(df.columns):
        raise ValueError("Dataset must contain Timestamp and Label.")

    if max_gap_seconds < 1:
        raise ValueError("max_gap_seconds must be at least 1.")

    work = df.copy()

    work["Timestamp"] = pd.to_datetime(
        work["Timestamp"], errors="coerce"
    )

    if work["Timestamp"].isna().any():
        raise ValueError("Invalid timestamps found.")

    if work["Label"].isna().any():
        raise ValueError("Missing labels found.")

    # Validate CaptureDay when it is available.
    if "CaptureDay" in work.columns:
        if work["CaptureDay"].isna().any():
            raise ValueError("Missing CaptureDay values found.")

        work["CaptureDay"] = work["CaptureDay"].astype(str)

        expected_days = work["Timestamp"].dt.strftime("%Y-%m-%d")

        if not work["CaptureDay"].eq(expected_days).all():
            raise ValueError("CaptureDay/Timestamp mismatch found.")

    work = work.sort_values(
        "Timestamp", kind="stable"
    ).reset_index(drop=True)

    # Identify numerical network-flow features.
    excluded_columns = {
        "Timestamp",
        "Label",
        "CaptureDay",
    } | CONSTANT_FEATURES

    feature_columns = [
        column
        for column in work.columns
        if column not in excluded_columns
    ]

    if not feature_columns:
        raise ValueError("No numerical flow features found.")

    non_numeric = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(work[column])
    ]

    if non_numeric:
        raise ValueError(
            f"Non-numerical flow features found: {non_numeric}"
        )

    # Convert infinities to missing values.
    work[feature_columns] = work[feature_columns].replace(
        [np.inf, -np.inf], np.nan
    )

    # Assign each flow to its time window.
    work["_window"] = work["Timestamp"].dt.floor(window)

    # Aggregate each numerical feature.
    aggregated = work.groupby(
        "_window", sort=True
    )[feature_columns].agg(
        ["mean", "std", "max"]
    )

    aggregated.columns = [
        f"{feature}__{stat}"
        for feature, stat in aggregated.columns
    ]

    # Identify observed network windows.
    observed = pd.DatetimeIndex(
        aggregated.index
    ).sort_values()

    # Split the timeline at gaps greater than the threshold.
    gap_seconds = observed.to_series().diff().dt.total_seconds()

    segment_ids = (
        gap_seconds > max_gap_seconds
    ).cumsum().to_numpy()

    # Summarize labels within each observed window.
    def summarize_label(labels):
        attacks = sorted({
            str(label)
            for label in labels
            if str(label).strip().lower() != "benign"
        })

        if not attacks:
            return "Benign"

        if len(attacks) == 1:
            return attacks[0]

        return "MIXED_ATTACK"

    window_labels = (
        work.groupby("_window")["Label"]
        .apply(summarize_label)
    )

    flow_counts = work.groupby("_window").size()

    state_parts = []

    for segment_id in np.unique(segment_ids):

        segment_times = observed[
            segment_ids == segment_id
        ]

        # Create continuous one-second timestamps only
        # inside this segment.
        full_index = pd.date_range(
            segment_times.min(),
            segment_times.max(),
            freq=window,
            name="Timestamp",
        )

        segment_states = aggregated.reindex(full_index)

        # Distinguish genuinely empty windows from observed
        # windows containing missing measurements.
        empty_mask = ~full_index.isin(aggregated.index)

        # Only empty windows receive zero-valued features.
        segment_states.loc[empty_mask, :] = 0.0

        counts = flow_counts.reindex(
            full_index, fill_value=0
        ).astype(np.float32)

        labels = (
            window_labels.reindex(full_index)
            .fillna("NO_FLOWS")
        )

        # Preserve CaptureDay as metadata.
        capture_days = full_index.strftime("%Y-%m-%d")

        metadata = pd.DataFrame(
            {
                "CaptureDay": capture_days,
                "Flow Count": counts.to_numpy(),
                "Has Flows": (
                    counts.to_numpy() > 0
                ).astype(np.float32),
                "Label": labels.to_numpy(),
                "Segment ID": np.full(
                    len(full_index),
                    int(segment_id),
                    dtype=np.int32,
                ),
            },
            index=full_index,
        )

        segment_states = pd.concat(
            [segment_states, metadata],
            axis=1,
        )

        state_parts.append(segment_states)

    # Combine all independent segments.
    combined = pd.concat(state_parts, axis=0)

    states = combined.reset_index()

    return states


def make_sequences(
    states: pd.DataFrame,
    sequence_length: int = 10,
):
    """
    Predict the next network state from previous states.

    Sequences never cross Segment ID boundaries.

    Returns:
        X: Previous network states.
        y: Next network states.
        timestamps: Target-state timestamps.
        labels: Target-state labels.
        feature_columns: Numerical input feature names.
    """

    if sequence_length < 1:
        raise ValueError(
            "sequence_length must be at least 1."
        )

    required = {
        "Timestamp",
        "Label",
        "Segment ID",
    }

    if not required.issubset(states.columns):
        raise ValueError(
            f"Required columns missing: "
            f"{sorted(required - set(states.columns))}"
        )

    feature_columns = [
    column
    for column in states.columns
    if column not in METADATA_COLUMNS
]

    if not feature_columns:
        raise ValueError(
            "No numerical state features found."
        )

    non_numeric = states[
        feature_columns
    ].select_dtypes(exclude=[np.number]).columns.tolist()

    if non_numeric:
        raise ValueError(
            f"Non-numerical state features found: {non_numeric}"
        )

    # Determine the exact number of sequences.
    segment_groups = list(
        states.groupby("Segment ID", sort=False)
    )

    total_sequences = sum(
        max(0, len(segment) - sequence_length)
        for _, segment in segment_groups
    )

    if total_sequences == 0:
        raise ValueError(
            "No sequences could be created. "
            "Check segment lengths and sequence_length."
        )

    number_of_features = len(feature_columns)

    # Preallocate arrays to avoid retaining thousands of
    # overlapping sequence arrays in Python lists.
    X = np.empty(
        (
            total_sequences,
            sequence_length,
            number_of_features,
        ),
        dtype=np.float32,
    )

    y = np.empty(
        (total_sequences, number_of_features),
        dtype=np.float32,
    )

    timestamps = []
    labels = []

    position = 0

    for _, segment in segment_groups:

        segment = segment.sort_values(
            "Timestamp", kind="stable"
        )

        values = segment[
            feature_columns
        ].to_numpy(dtype=np.float32)

        if not np.isfinite(values).all():
            raise ValueError(
                "State features contain NaN or infinite values. "
                "Run StatePreprocessor before make_sequences."
            )

        number_of_segment_sequences = (
            len(segment) - sequence_length
        )

        if number_of_segment_sequences <= 0:
            continue

        # Create sliding windows without constructing a
        # separate Python list for every sequence.
        windows = np.lib.stride_tricks.sliding_window_view(
            values,
            window_shape=sequence_length + 1,
            axis=0,
        )

        windows = windows.transpose(0, 2, 1)

        end = position + number_of_segment_sequences

        X[position:end] = windows[:, :-1, :]
        y[position:end] = windows[:, -1, :]

        timestamps.extend(
            segment["Timestamp"]
            .iloc[sequence_length:]
            .to_list()
        )

        labels.extend(
            segment["Label"]
            .iloc[sequence_length:]
            .to_list()
        )

        position = end

    return (
        X,
        y,
        np.asarray(timestamps),
        np.asarray(labels),
        feature_columns,
    )
