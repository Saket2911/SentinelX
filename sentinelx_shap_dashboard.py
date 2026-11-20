from itertools import product
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


SEQUENCE_LENGTH = 10
EXPLANATION_COUNT = 64
EXPECTED_LABELS = ("Benign", "Bot")
REPLAY_METADATA_COLUMNS = (
    "Target State Row",
    "Target Timestamp",
    "Target CaptureDay",
    "Target Label",
    "Segment ID",
)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _string_vector(
    values: np.ndarray, name: str, require_unique: bool = True
) -> np.ndarray:
    if values.ndim != 1 or values.dtype.kind not in "US":
        raise ValueError(f"{name} must be a one-dimensional string array.")
    result = values.astype(str)
    if any(not value.strip() for value in result):
        raise ValueError(f"{name} contains empty names.")
    if require_unique and len(set(result.tolist())) != len(result):
        raise ValueError(f"{name} contains duplicate names.")
    return result


def _index_vector(values: np.ndarray, name: str) -> np.ndarray:
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"{name} must be a one-dimensional integer array.")
    return values.astype(np.int64, copy=False)


def _require_finite(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{name} column {column!r} contains non-finite values.")


def _validate_combinations(
    frame: pd.DataFrame,
    columns: Sequence[str],
    expected: set,
    name: str,
) -> None:
    actual_rows = list(frame.loc[:, columns].itertuples(index=False, name=None))
    if len(actual_rows) != len(expected) or set(actual_rows) != expected:
        raise ValueError(f"{name} has missing, duplicate, or unexpected rows.")


def _same_replay_value(metadata_value, replay_value, column: str) -> bool:
    if pd.isna(metadata_value) or pd.isna(replay_value):
        return False
    if column in ("Target State Row", "Segment ID"):
        return int(metadata_value) == int(replay_value)
    return str(metadata_value).strip() == str(replay_value).strip()


def load_shap_artifacts(
    shap_dir: Path,
    test_metadata_path: Path,
    replay_csv_path: Path,
    expected_feature_names: Sequence[str],
) -> dict:
    shap_dir = Path(shap_dir)
    test_metadata_path = Path(test_metadata_path)
    replay_csv_path = Path(replay_csv_path)
    paths = {
        "SHAP values": shap_dir / "shap_values.npz",
        "feature importance": shap_dir / "shap_feature_importance_by_label.csv",
        "temporal importance": shap_dir / "shap_temporal_importance_by_label.csv",
        "Benign/Bot comparison": shap_dir / "shap_benign_bot_comparison.csv",
        "V2/V3 stability": shap_dir / "stability_comparison_v2_v3.csv",
        "test metadata": test_metadata_path,
        "historical replay": replay_csv_path,
    }
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing SHAP dashboard artifacts:\n" + "\n".join(missing))

    required_arrays = (
        "shap_values",
        "explain_indices",
        "explain_labels",
        "output_indices",
        "feature_names",
        "output_names",
    )
    with np.load(paths["SHAP values"], allow_pickle=False) as archive:
        missing_arrays = [key for key in required_arrays if key not in archive.files]
        if missing_arrays:
            raise ValueError(f"SHAP values archive is missing arrays: {missing_arrays}")
        shap_values = archive["shap_values"].copy()
        explain_indices = archive["explain_indices"].copy()
        explain_labels = archive["explain_labels"].copy()
        output_indices = archive["output_indices"].copy()
        feature_names = _string_vector(archive["feature_names"], "feature_names")
        output_names = _string_vector(archive["output_names"], "output_names")

    explain_indices = _index_vector(explain_indices, "explain_indices")
    output_indices = _index_vector(output_indices, "output_indices")
    explain_labels = _string_vector(
        explain_labels, "explain_labels", require_unique=False
    )
    expected_feature_names = np.asarray(expected_feature_names, dtype=str)

    if len(feature_names) != len(expected_feature_names):
        raise ValueError("SHAP feature count does not match replay feature count.")
    if not np.array_equal(feature_names, expected_feature_names):
        raise ValueError("SHAP feature names or order do not match replay features.")
    if len(output_names) != 5 or len(output_indices) != len(output_names):
        raise ValueError("V3 must contain five aligned output names and indices.")
    if np.any(output_indices < 0) or np.any(output_indices >= len(feature_names)):
        raise ValueError("SHAP output indices are outside the model feature range.")
    if not np.array_equal(feature_names[output_indices], output_names):
        raise ValueError("SHAP output names do not match their saved output indices.")
    if shap_values.shape != (
        EXPLANATION_COUNT,
        SEQUENCE_LENGTH,
        len(feature_names),
        len(output_names),
    ):
        raise ValueError(
            "SHAP values have invalid shape "
            f"{shap_values.shape}; expected "
            f"({EXPLANATION_COUNT}, {SEQUENCE_LENGTH}, "
            f"{len(feature_names)}, {len(output_names)})."
        )
    if not np.issubdtype(shap_values.dtype, np.number) or not np.isfinite(shap_values).all():
        raise ValueError("SHAP values must contain only finite numeric values.")
    if len(explain_indices) != EXPLANATION_COUNT or len(explain_labels) != EXPLANATION_COUNT:
        raise ValueError("V3 must contain exactly 64 aligned explanation indices and labels.")
    if len(np.unique(explain_indices)) != EXPLANATION_COUNT:
        raise ValueError("SHAP explanation indices must be unique.")

    metadata = pd.read_csv(
        test_metadata_path,
        dtype={
            "Target Timestamp": str,
            "Target CaptureDay": str,
            "Target Label": str,
        },
    )
    replay = pd.read_csv(
        replay_csv_path,
        dtype={
            "Target Timestamp": str,
            "Target CaptureDay": str,
            "Target Label": str,
        },
    )
    _require_columns(metadata, REPLAY_METADATA_COLUMNS, "Test metadata")
    _require_columns(replay, ("Replay Step",) + REPLAY_METADATA_COLUMNS, "Replay CSV")
    if len(metadata) != len(replay) or len(metadata) == 0:
        raise ValueError("Test metadata and replay rows have different or empty lengths.")
    steps = pd.to_numeric(replay["Replay Step"], errors="coerce").to_numpy()
    expected_steps = np.arange(1, len(replay) + 1)
    if (
        not np.isfinite(steps).all()
        or not np.array_equal(steps, expected_steps)
    ):
        raise ValueError("Replay steps must be unique, contiguous, and start at 1.")
    if np.any(explain_indices < 0) or np.any(explain_indices >= len(metadata)):
        raise ValueError("A SHAP explanation index is outside the test metadata range.")
    if set(explain_labels.tolist()) != set(EXPECTED_LABELS):
        raise ValueError("V3 explanations must include Benign and Bot labels.")
    label_counts = pd.Series(explain_labels).value_counts().to_dict()
    if label_counts != {"Benign": 32, "Bot": 32}:
        raise ValueError("V3 must contain 32 Benign and 32 Bot explanations.")

    replay_by_step = replay.assign(_ReplayStep=steps).set_index("_ReplayStep")
    observations = []
    for sample_position, sequence_index in enumerate(explain_indices):
        replay_step = int(sequence_index) + 1
        metadata_row = metadata.iloc[int(sequence_index)]
        replay_row = replay_by_step.loc[replay_step]
        for column in REPLAY_METADATA_COLUMNS:
            if not _same_replay_value(metadata_row[column], replay_row[column], column):
                raise ValueError(
                    f"Replay step {replay_step} does not match test metadata "
                    f"for {column!r}."
                )
        if explain_labels[sample_position] != str(metadata_row["Target Label"]):
            raise ValueError(
                f"SHAP label does not match test metadata at replay step {replay_step}."
            )
        observations.append({
            "Sample Position": sample_position,
            "Replay Step": replay_step,
            **{column: metadata_row[column] for column in REPLAY_METADATA_COLUMNS},
        })

    feature_importance = pd.read_csv(paths["feature importance"])
    temporal_importance = pd.read_csv(paths["temporal importance"])
    comparison = pd.read_csv(paths["Benign/Bot comparison"])
    stability = pd.read_csv(paths["V2/V3 stability"])

    _require_columns(
        feature_importance,
        (
            "Target Label",
            "Predicted Output Feature",
            "Input Feature",
            "Mean Absolute SHAP",
            "Mean Signed SHAP",
            "Feature Rank",
        ),
        "Feature importance CSV",
    )
    feature_importance["Target Label"] = feature_importance["Target Label"].astype(str)
    feature_importance["Predicted Output Feature"] = (
        feature_importance["Predicted Output Feature"].astype(str)
    )
    feature_importance["Input Feature"] = feature_importance["Input Feature"].astype(str)
    expected_feature_rows = set(product(
        EXPECTED_LABELS, output_names.tolist(), feature_names.tolist()
    ))
    _validate_combinations(
        feature_importance,
        ("Target Label", "Predicted Output Feature", "Input Feature"),
        expected_feature_rows,
        "Feature importance CSV",
    )
    _require_finite(
        feature_importance,
        ("Mean Absolute SHAP", "Mean Signed SHAP", "Feature Rank"),
        "Feature importance CSV",
    )
    if (feature_importance["Mean Absolute SHAP"] < 0).any():
        raise ValueError("Feature importance contains negative absolute attributions.")

    _require_columns(
        temporal_importance,
        (
            "Target Label",
            "Predicted Output Feature",
            "Input Time Position",
            "Mean Absolute SHAP",
        ),
        "Temporal importance CSV",
    )
    temporal_importance["Target Label"] = temporal_importance["Target Label"].astype(str)
    temporal_importance["Predicted Output Feature"] = (
        temporal_importance["Predicted Output Feature"].astype(str)
    )
    temporal_positions = pd.to_numeric(
        temporal_importance["Input Time Position"], errors="coerce"
    )
    temporal_importance["Input Time Position"] = temporal_positions
    expected_temporal_rows = set(product(
        EXPECTED_LABELS, output_names.tolist(), range(SEQUENCE_LENGTH)
    ))
    _validate_combinations(
        temporal_importance,
        ("Target Label", "Predicted Output Feature", "Input Time Position"),
        expected_temporal_rows,
        "Temporal importance CSV",
    )
    _require_finite(
        temporal_importance, ("Mean Absolute SHAP",), "Temporal importance CSV"
    )
    if (temporal_importance["Mean Absolute SHAP"] < 0).any():
        raise ValueError("Temporal importance contains negative absolute attributions.")

    comparison_columns = (
        "Predicted Output Feature",
        "Input Feature",
        "Benign",
        "Bot",
        "Bot_minus_Benign_SHAP",
        "Absolute_Difference",
    )
    _require_columns(comparison, comparison_columns, "Benign/Bot comparison CSV")
    comparison["Predicted Output Feature"] = (
        comparison["Predicted Output Feature"].astype(str)
    )
    comparison["Input Feature"] = comparison["Input Feature"].astype(str)
    expected_comparison_rows = set(product(
        output_names.tolist(), feature_names.tolist()
    ))
    _validate_combinations(
        comparison,
        ("Predicted Output Feature", "Input Feature"),
        expected_comparison_rows,
        "Benign/Bot comparison CSV",
    )
    _require_finite(
        comparison,
        ("Benign", "Bot", "Bot_minus_Benign_SHAP", "Absolute_Difference"),
        "Benign/Bot comparison CSV",
    )
    if (comparison["Absolute_Difference"] < 0).any():
        raise ValueError("Benign/Bot comparison contains negative absolute differences.")

    _require_columns(
        stability,
        ("Label", "Output", "Top10 Overlap", "Jaccard", "Spearman"),
        "V2/V3 stability CSV",
    )
    stability["Label"] = stability["Label"].astype(str)
    stability["Output"] = stability["Output"].astype(str)
    expected_stability_rows = set(product(EXPECTED_LABELS, output_names.tolist()))
    _validate_combinations(
        stability, ("Label", "Output"), expected_stability_rows, "V2/V3 stability CSV"
    )
    _require_finite(
        stability, ("Top10 Overlap", "Jaccard", "Spearman"), "V2/V3 stability CSV"
    )
    if (
        (stability["Top10 Overlap"] < 0).any()
        or (stability["Top10 Overlap"] > 10).any()
        or (stability["Jaccard"] < 0).any()
        or (stability["Jaccard"] > 1).any()
        or (stability["Spearman"] < -1).any()
        or (stability["Spearman"] > 1).any()
    ):
        raise ValueError("V2/V3 stability metrics are outside their valid ranges.")

    return {
        "shap_values": shap_values,
        "explain_indices": explain_indices,
        "explain_labels": explain_labels,
        "output_indices": output_indices,
        "feature_names": feature_names,
        "output_names": output_names,
        "observations": pd.DataFrame(observations),
        "feature_importance": feature_importance,
        "temporal_importance": temporal_importance,
        "comparison": comparison,
        "stability": stability,
    }


def sample_position_for_replay_step(
    explain_indices: Sequence[int], replay_step: int
) -> int | None:
    matches = np.flatnonzero(np.asarray(explain_indices) + 1 == int(replay_step))
    if len(matches) > 1:
        raise ValueError(f"Replay step {replay_step} has multiple SHAP samples.")
    return int(matches[0]) if len(matches) else None


def aggregate_local_contributions(
    values_by_time: np.ndarray, feature_names: Sequence[str]
) -> pd.DataFrame:
    values_by_time = np.asarray(values_by_time)
    if values_by_time.shape != (SEQUENCE_LENGTH, len(feature_names)):
        raise ValueError(
            "Local SHAP values must have shape "
            f"({SEQUENCE_LENGTH}, {len(feature_names)})."
        )
    if not np.issubdtype(values_by_time.dtype, np.number) or not np.isfinite(values_by_time).all():
        raise ValueError("Local SHAP values must contain finite numeric values.")

    contributions = pd.DataFrame({
        "Input Feature": list(feature_names),
        "Signed Contribution": values_by_time.sum(axis=0),
        "Mean Absolute Attribution": np.abs(values_by_time).mean(axis=0),
    })
    return contributions.sort_values(
        "Mean Absolute Attribution", ascending=False, kind="stable"
    ).reset_index(drop=True)
