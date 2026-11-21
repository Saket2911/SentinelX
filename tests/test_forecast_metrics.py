"""Tests for the forecast evaluation primitives and experiment guardrails.

These tests use small synthetic arrays; they assert implementation correctness,
not model performance.
"""

from __future__ import annotations

import unittest

import numpy as np

from sentinelx.evaluation.forecast_metrics import (
    calibrate_threshold,
    detection_metrics,
    error_metrics,
    error_quantiles,
    huber_loss,
    multi_step_metrics,
    multi_step_recursive,
    per_feature_metrics,
    per_label_metrics,
    persistence_forecast,
)
from sentinelx.training.experiments import recommend_promotion


class TestErrorMetrics(unittest.TestCase):
    def test_known_values(self) -> None:
        actual = np.array([[1.0, 2.0], [3.0, 4.0]])
        predicted = np.array([[0.0, 0.0], [0.0, 0.0]])
        metrics = error_metrics(actual, predicted)
        self.assertAlmostEqual(metrics["mae"], 2.5)
        self.assertAlmostEqual(metrics["mse"], 7.5)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(7.5))
        self.assertEqual(metrics["samples"], 2.0)
        self.assertEqual(metrics["features"], 2.0)

    def test_huber_matches_definition(self) -> None:
        residuals = np.array([0.5, 2.0, -3.0])
        # delta=1 keeps the quadratic region below 1: 0.5*a^2,
        # and the linear region above it: delta*(|a| - 0.5*delta).
        expected = (0.5 * 0.5**2 + 1.0 * (2.0 - 0.5) + 1.0 * (3.0 - 0.5)) / 3
        self.assertAlmostEqual(huber_loss(residuals, delta=1.0), expected)

    def test_perfect_prediction(self) -> None:
        values = np.arange(12, dtype=float).reshape(3, 4)
        metrics = error_metrics(values, values)
        self.assertEqual(metrics["mae"], 0.0)
        self.assertEqual(metrics["rmse"], 0.0)
        self.assertEqual(metrics["max_abs_error"], 0.0)

    def test_shape_and_finiteness_are_enforced(self) -> None:
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            error_metrics(np.zeros((2, 3)), np.zeros((2, 4)))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            error_metrics(np.array([[1.0, np.nan]]), np.zeros((1, 2)))
        with self.assertRaisesRegex(ValueError, "2-dimensional"):
            error_metrics(np.zeros(3), np.zeros(3))

    def test_per_feature_metrics_are_ordered_and_named(self) -> None:
        actual = np.array([[1.0, 9.0], [1.0, 9.0]])
        predicted = np.array([[0.0, 4.0], [0.0, 4.0]])
        rows = per_feature_metrics(actual, predicted, ["small", "big"])
        self.assertEqual([row["feature"] for row in rows], ["big", "small"])
        self.assertAlmostEqual(rows[0]["mae"], 5.0)
        self.assertAlmostEqual(rows[1]["mae"], 1.0)
        with self.assertRaisesRegex(ValueError, "feature_names"):
            per_feature_metrics(actual, predicted, ["only-one"])

    def test_error_quantiles_monotonic(self) -> None:
        residuals = np.linspace(0.0, 10.0, 101)
        quantiles = error_quantiles(residuals)
        self.assertAlmostEqual(quantiles["0.5"], 5.0, places=6)
        self.assertLess(quantiles["0.5"], quantiles["0.99"])
        self.assertAlmostEqual(quantiles["0.999"], np.percentile(residuals, 99.9))

    def test_per_label_metrics(self) -> None:
        actual = np.array([[1.0], [2.0], [3.0], [4.0]])
        predicted = np.zeros((4, 1))
        rows = per_label_metrics(actual, predicted, ["Benign", "Bot", "Benign", "NO_FLOWS"])
        by_label = {row["label"]: row for row in rows}
        self.assertEqual(by_label["Benign"]["samples"], 2)
        self.assertAlmostEqual(by_label["Benign"]["mae"], 2.0)
        self.assertAlmostEqual(by_label["Bot"]["mae"], 2.0)


class TestBaselines(unittest.TestCase):
    def test_persistence_repeats_last_state(self) -> None:
        sequences = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
        baseline = persistence_forecast(sequences)
        np.testing.assert_array_equal(baseline, sequences[:, -1, :])
        self.assertEqual(baseline.shape, (2, 4))

    def test_persistence_rejects_bad_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            persistence_forecast(np.zeros((2, 3)))

    def test_multi_step_recursive_repeated_identity(self) -> None:
        window = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        def predict_next(current: np.ndarray) -> np.ndarray:
            return current[-1] + 1.0

        forecasts = multi_step_recursive(predict_next, window, horizon=3)
        self.assertEqual(forecasts.shape, (3, 2))
        np.testing.assert_allclose(forecasts[0], [6.0, 7.0])
        np.testing.assert_allclose(forecasts[2], [8.0, 9.0])

    def test_multi_step_recursive_validates_predictions(self) -> None:
        with self.assertRaisesRegex(ValueError, "predict_next must return"):
            multi_step_recursive(lambda window: np.zeros((2, 2)), np.zeros((3, 2)), 1)
        with self.assertRaisesRegex(ValueError, "horizon"):
            multi_step_recursive(lambda window: window[-1], np.zeros((3, 2)), 0)

    def test_multi_step_metrics_per_horizon(self) -> None:
        actual = np.array([[[1.0], [2.0]], [[3.0], [4.0]]])
        predicted = np.zeros_like(actual)
        rows = multi_step_metrics(actual, predicted, horizons=[1, 2])
        self.assertEqual([row["horizon"] for row in rows], [1, 2])
        self.assertAlmostEqual(rows[0]["mae"], 2.0)
        self.assertAlmostEqual(rows[1]["mae"], 3.0)
        with self.assertRaisesRegex(ValueError, "outside the available range"):
            multi_step_metrics(actual, predicted, horizons=[3])


class TestCalibrationAndDetection(unittest.TestCase):
    def test_threshold_uses_requested_percentile(self) -> None:
        scores = np.linspace(0, 100, 1001)
        result = calibrate_threshold(scores, percentile=99.0)
        self.assertAlmostEqual(result["threshold"], np.percentile(scores, 99))
        self.assertEqual(result["benign_samples"], 1001.0)

    def test_threshold_rejects_invalid_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            calibrate_threshold(np.array([]))
        with self.assertRaisesRegex(ValueError, "percentile"):
            calibrate_threshold(np.arange(10.0), percentile=100)

    def test_detection_metrics_confusion_matrix(self) -> None:
        scores = np.array([0.9, 0.95, 0.1, 0.2, 0.99])
        labels = ["Benign", "Bot", "Benign", "Bot", "NO_FLOWS"]
        result = detection_metrics(scores, labels, threshold=0.9)
        self.assertEqual(result["TP"], 1)  # 0.95 Bot
        self.assertEqual(result["FP"], 1)  # 0.9? 0.9 >= 0.9 -> Benign flagged
        self.assertEqual(result["FN"], 1)  # 0.2 Bot
        self.assertEqual(result["TN"], 1)  # 0.1 Benign
        self.assertEqual(result["excluded_observations"], 1)
        self.assertAlmostEqual(result["precision"], 0.5)
        self.assertAlmostEqual(result["recall"], 0.5)
        self.assertAlmostEqual(result["specificity"], 0.5)
        self.assertAlmostEqual(result["f1_score"], 0.5)
        self.assertIn("Not a validated", str(result["interpretation"]))

    def test_detection_metrics_rejects_bad_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "same length"):
            detection_metrics(np.zeros(3), ["Benign"], 1.0)
        with self.assertRaisesRegex(ValueError, "all observations were excluded"):
            detection_metrics(np.zeros(2), ["NO_FLOWS", "NO_FLOWS"], 1.0)
        with self.assertRaisesRegex(ValueError, "negative"):
            detection_metrics(np.zeros(2), ["Benign", "Bot"], -1.0)


class TestPromotionGuardrails(unittest.TestCase):
    def _report(self, mae: float, rmse: float, checkpoint: bool = True, test_used: bool = False):
        return {
            "final_validation": {"mae": mae, "rmse": rmse, "huber": mae / 2},
            "checkpoint_written": checkpoint,
            "test_partition_used": test_used,
        }

    def test_promotes_clear_improvement(self) -> None:
        decision = recommend_promotion(self._report(0.30, 0.50), self._report(0.35, 0.60))
        self.assertTrue(decision["promote"])
        self.assertAlmostEqual(decision["mae_gain"], (0.35 - 0.30) / 0.35)

    def test_blocks_without_checkpoint(self) -> None:
        decision = recommend_promotion(
            self._report(0.30, 0.50, checkpoint=False), self._report(0.35, 0.60)
        )
        self.assertFalse(decision["promote"])
        self.assertTrue(any("checkpoint" in reason for reason in decision["reasons"]))

    def test_blocks_marginal_or_mixed_gains(self) -> None:
        marginal = recommend_promotion(self._report(0.3495, 0.5985), self._report(0.35, 0.60))
        self.assertFalse(marginal["promote"])
        mixed = recommend_promotion(self._report(0.30, 0.65), self._report(0.35, 0.60))
        self.assertFalse(mixed["promote"])
        self.assertTrue(any("RMSE gain" in reason for reason in mixed["reasons"]))

    def test_blocks_test_partition_use(self) -> None:
        decision = recommend_promotion(
            self._report(0.30, 0.50, test_used=True), self._report(0.35, 0.60)
        )
        self.assertFalse(decision["promote"])

    def test_missing_metrics_blocks(self) -> None:
        decision = recommend_promotion({"checkpoint_written": True}, self._report(0.35, 0.60))
        self.assertFalse(decision["promote"])


if __name__ == "__main__":
    unittest.main()
