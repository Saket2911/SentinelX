import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from sentinelx_shap_dashboard import (
    aggregate_local_contributions,
    load_shap_artifacts,
    sample_position_for_replay_step,
)


FEATURE_NAMES = [f"feature_{index}" for index in range(209)]
OUTPUT_INDICES = [207, 51, 60, 15, 135]
OUTPUT_NAMES = [FEATURE_NAMES[index] for index in OUTPUT_INDICES]
METADATA_COLUMNS = [
    "Target State Row",
    "Target Timestamp",
    "Target CaptureDay",
    "Target Label",
    "Segment ID",
]


def write_artifacts(directory, shap_values=None, feature_names=None):
    directory = Path(directory)
    if shap_values is None:
        shap_values = np.zeros((64, 10, 209, 5), dtype=np.float32)
    if feature_names is None:
        feature_names = FEATURE_NAMES

    labels = ["Benign"] * 32 + ["Bot"] * 32
    metadata_rows = []
    replay_rows = []
    for index in range(80):
        label = labels[index] if index < len(labels) else "Benign"
        row = {
            "Target State Row": 100 + index,
            "Target Timestamp": f"2020-01-01 00:{index // 60:02d}:{index % 60:02d}",
            "Target CaptureDay": "2020-01-01",
            "Target Label": label,
            "Segment ID": 1,
        }
        metadata_rows.append(row)
        replay_rows.append({"Replay Step": index + 1, **row})

    metadata_path = directory / "test_metadata.csv"
    replay_path = directory / "test_replay.csv"
    pd.DataFrame(metadata_rows).to_csv(metadata_path, index=False)
    pd.DataFrame(replay_rows).to_csv(replay_path, index=False)

    np.savez_compressed(
        directory / "shap_values.npz",
        shap_values=shap_values,
        explain_indices=np.arange(64, dtype=np.int64),
        explain_labels=np.asarray(labels),
        output_indices=np.asarray(OUTPUT_INDICES, dtype=np.int64),
        feature_names=np.asarray(feature_names),
        output_names=np.asarray(OUTPUT_NAMES),
    )

    feature_rows = []
    for label in ("Benign", "Bot"):
        for output in OUTPUT_NAMES:
            for rank, feature in enumerate(FEATURE_NAMES, start=1):
                feature_rows.append({
                    "Target Label": label,
                    "Predicted Output Feature": output,
                    "Input Feature": feature,
                    "Mean Absolute SHAP": 1.0 / rank,
                    "Mean Signed SHAP": 0.01,
                    "Feature Rank": rank,
                })
    pd.DataFrame(feature_rows).to_csv(
        directory / "shap_feature_importance_by_label.csv", index=False
    )

    temporal_rows = [
        {
            "Target Label": label,
            "Predicted Output Feature": output,
            "Input Time Position": position,
            "Mean Absolute SHAP": 0.01,
        }
        for label in ("Benign", "Bot")
        for output in OUTPUT_NAMES
        for position in range(10)
    ]
    pd.DataFrame(temporal_rows).to_csv(
        directory / "shap_temporal_importance_by_label.csv", index=False
    )

    comparison_rows = [
        {
            "Predicted Output Feature": output,
            "Input Feature": feature,
            "Benign": 0.1,
            "Bot": 0.2,
            "Bot_minus_Benign_SHAP": 0.1,
            "Absolute_Difference": 0.1,
        }
        for output in OUTPUT_NAMES
        for feature in FEATURE_NAMES
    ]
    pd.DataFrame(comparison_rows).to_csv(
        directory / "shap_benign_bot_comparison.csv", index=False
    )

    stability_rows = [
        {
            "Label": label,
            "Output": output,
            "Top10 Overlap": 8,
            "Jaccard": 2 / 3,
            "Spearman": 0.95,
        }
        for label in ("Benign", "Bot")
        for output in OUTPUT_NAMES
    ]
    pd.DataFrame(stability_rows).to_csv(
        directory / "stability_comparison_v2_v3.csv", index=False
    )
    return metadata_path, replay_path


class TestShapDashboardArtifacts(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.directory = Path(self.temporary_directory.name)
        self.metadata_path, self.replay_path = write_artifacts(self.directory)

    def load_artifacts(self):
        return load_shap_artifacts(
            self.directory,
            self.metadata_path,
            self.replay_path,
            FEATURE_NAMES,
        )

    def test_loads_v3_artifacts_and_maps_all_samples_to_replay_rows(self):
        artifacts = self.load_artifacts()

        self.assertEqual(artifacts["shap_values"].shape, (64, 10, 209, 5))
        self.assertEqual(artifacts["output_names"].tolist(), OUTPUT_NAMES)
        self.assertEqual(artifacts["output_indices"].tolist(), OUTPUT_INDICES)
        self.assertEqual(len(artifacts["observations"]), 64)
        self.assertEqual(
            artifacts["observations"]["Replay Step"].tolist(),
            list(range(1, 65)),
        )
        self.assertTrue(
            (artifacts["observations"]["Target Label"].to_numpy()
             == artifacts["explain_labels"]).all()
        )
        for sample_position in range(64):
            self.assertEqual(
                sample_position_for_replay_step(
                    artifacts["explain_indices"], sample_position + 1
                ),
                sample_position,
            )
        self.assertIsNone(sample_position_for_replay_step(
            artifacts["explain_indices"], 65
        ))

    def test_missing_artifact_is_reported(self):
        (self.directory / "stability_comparison_v2_v3.csv").unlink()

        with self.assertRaisesRegex(FileNotFoundError, "V2/V3 stability"):
            self.load_artifacts()

    def test_invalid_shap_shape_is_rejected(self):
        write_artifacts(
            self.directory,
            shap_values=np.zeros((63, 10, 209, 5), dtype=np.float32),
        )

        with self.assertRaisesRegex(ValueError, "invalid shape"):
            self.load_artifacts()

    def test_feature_name_mismatch_is_rejected(self):
        names = FEATURE_NAMES.copy()
        names[0] = "unexpected_feature"
        write_artifacts(self.directory, feature_names=names)

        with self.assertRaisesRegex(ValueError, "feature names or order"):
            self.load_artifacts()

    def test_nonfinite_shap_value_is_rejected(self):
        values = np.zeros((64, 10, 209, 5), dtype=np.float32)
        values[0, 0, 0, 0] = np.nan
        write_artifacts(self.directory, shap_values=values)

        with self.assertRaisesRegex(ValueError, "finite numeric values"):
            self.load_artifacts()

    def test_replay_timestamp_mismatch_is_rejected(self):
        replay = pd.read_csv(self.replay_path)
        replay.loc[0, "Target Timestamp"] = "2020-01-02 00:00:00"
        replay.to_csv(self.replay_path, index=False)

        with self.assertRaisesRegex(ValueError, "Target Timestamp"):
            self.load_artifacts()

    def test_nonfinite_csv_value_is_rejected(self):
        comparison_path = self.directory / "shap_benign_bot_comparison.csv"
        comparison = pd.read_csv(comparison_path)
        comparison.loc[0, "Benign"] = np.inf
        comparison.to_csv(comparison_path, index=False)

        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.load_artifacts()

    def test_local_contributions_sum_sign_and_mean_absolute_by_feature(self):
        values = np.zeros((10, 209), dtype=np.float64)
        values[:, 0] = np.arange(10) - 4
        values[:, 1] = -2
        contributions = aggregate_local_contributions(values, FEATURE_NAMES)

        self.assertEqual(contributions.iloc[0]["Input Feature"], "feature_0")
        self.assertAlmostEqual(contributions.iloc[0]["Signed Contribution"], 5.0)
        self.assertAlmostEqual(
            contributions.iloc[0]["Mean Absolute Attribution"], 2.5
        )
        feature_one = contributions.set_index("Input Feature").loc["feature_1"]
        self.assertAlmostEqual(feature_one["Signed Contribution"], -20.0)
        self.assertAlmostEqual(feature_one["Mean Absolute Attribution"], 2.0)


if __name__ == "__main__":
    unittest.main()
