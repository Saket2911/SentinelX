import unittest

import numpy as np
import pandas as pd

from sentinelx.inference.streaming import SentinelXStreamingPredictor


FEATURES = [f"feature_{i}" for i in range(209)]


class FakePredictor:
    def __init__(self):
        self.feature_names = FEATURES
        self.sequence_length = 10
        self.call_count = 0
        self.last_sequence = None

    def predict_raw_sequence(self, sequence):
        self.call_count += 1
        self.last_sequence = sequence.copy()

        return pd.Series(
            np.zeros(209, dtype=np.float32),
            index=self.feature_names,
        )


def make_state(timestamp, segment_id=1, value=1.0):
    state = {
        "Timestamp": timestamp,
        "Segment ID": segment_id,
    }
    state.update({name: value for name in FEATURES})
    return state


class TestSentinelXStreamingPredictor(unittest.TestCase):

    def setUp(self):
        self.fake = FakePredictor()
        self.stream = SentinelXStreamingPredictor(self.fake)

    def submit(self, second, segment=1, value=1.0):
        timestamp = (
            pd.Timestamp("2026-01-01 00:00:00")
            + pd.Timedelta(seconds=second)
        )
        return self.stream.process_state(
            make_state(timestamp, segment, value)
        )

    def test_initialization(self):
        self.assertEqual(self.stream.sequence_length, 10)
        self.assertEqual(len(self.stream.feature_names), 209)
        self.assertEqual(len(self.stream.buffer), 0)

    def test_rejects_invalid_predictor(self):
        with self.assertRaises(TypeError):
            SentinelXStreamingPredictor(object())

    def test_rejects_wrong_feature_count(self):
        class BadPredictor(FakePredictor):
            def __init__(self):
                self.feature_names = FEATURES[:-1]
                self.sequence_length = 10

        with self.assertRaises(ValueError):
            SentinelXStreamingPredictor(BadPredictor())

    def test_rejects_wrong_sequence_length(self):
        class BadPredictor(FakePredictor):
            def __init__(self):
                self.feature_names = FEATURES
                self.sequence_length = 5

        with self.assertRaises(ValueError):
            SentinelXStreamingPredictor(BadPredictor())

    def test_nine_states_warm_up_tenth_predicts(self):
        for second in range(9):
            result = self.submit(second)
            self.assertEqual(result["status"], "WARMING_UP")

        self.assertEqual(self.fake.call_count, 0)

        result = self.submit(9)

        self.assertEqual(result["status"], "PREDICTION_READY")
        self.assertEqual(result["buffer_size"], 10)
        self.assertEqual(self.fake.call_count, 1)
        self.assertEqual(result["prediction"].shape, (209,))
        self.assertEqual(
            result["predicted_timestamp"],
            result["timestamp"] + pd.Timedelta(seconds=1),
        )

    def test_buffer_keeps_latest_ten_states(self):
        for second in range(15):
            self.submit(second, value=float(second))

        self.assertEqual(len(self.stream.buffer), 10)

        buffered_values = [
            row["feature_0"] for row in self.stream.buffer
        ]

        self.assertEqual(
            buffered_values,
            [float(i) for i in range(5, 15)],
        )

    def test_segment_change_resets_buffer(self):
        for second in range(10):
            self.submit(second, segment=1)

        result = self.submit(10, segment=2)

        self.assertEqual(result["status"], "WARMING_UP")
        self.assertEqual(result["reset_reason"], "segment_change")
        self.assertEqual(result["buffer_size"], 1)
        self.assertEqual(self.fake.call_count, 1)

    def test_timestamp_gap_resets_buffer(self):
        for second in range(10):
            self.submit(second)

        result = self.submit(12)

        self.assertEqual(result["status"], "WARMING_UP")
        self.assertEqual(result["reset_reason"], "timestamp_gap")
        self.assertEqual(result["buffer_size"], 1)
        self.assertEqual(self.fake.call_count, 1)

    def test_duplicate_timestamp_rejected_without_mutation(self):
        for second in range(10):
            self.submit(second)

        before = list(self.stream.buffer)

        with self.assertRaisesRegex(ValueError, "Duplicate or out-of-order"):
            self.submit(9, value=999.0)

        self.assertEqual(len(self.stream.buffer), 10)

        for before_row, after_row in zip(
            before, self.stream.buffer
        ):
            self.assertTrue(before_row.equals(after_row))

    def test_out_of_order_timestamp_rejected_without_mutation(self):
        for second in range(10):
            self.submit(second)

        before = list(self.stream.buffer)

        with self.assertRaisesRegex(ValueError, "Duplicate or out-of-order"):
            self.submit(8)

        self.assertEqual(len(self.stream.buffer), 10)

        for before_row, after_row in zip(
            before, self.stream.buffer
        ):
            self.assertTrue(before_row.equals(after_row))

    def test_missing_feature_rejected_without_mutation(self):
        for second in range(5):
            self.submit(second)

        before = list(self.stream.buffer)
        state = make_state(5)
        del state["feature_100"]

        with self.assertRaisesRegex(ValueError, "Missing model features"):
            self.stream.process_state(state)

        self.assertEqual(len(self.stream.buffer), 5)

        for before_row, after_row in zip(
            before, self.stream.buffer
        ):
            self.assertTrue(before_row.equals(after_row))

    def test_missing_timestamp_rejected(self):
        state = make_state(0)
        del state["Timestamp"]

        with self.assertRaisesRegex(ValueError, "Missing required column"):
            self.stream.process_state(state)

        self.assertEqual(len(self.stream.buffer), 0)

    def test_invalid_timestamp_rejected(self):
        state = make_state("not-a-timestamp")

        with self.assertRaisesRegex(ValueError, "Invalid timestamp"):
            self.stream.process_state(state)

        self.assertEqual(len(self.stream.buffer), 0)

    def test_missing_segment_id_rejected(self):
        state = make_state(0)
        state["Segment ID"] = None

        with self.assertRaisesRegex(ValueError, "Segment ID cannot be missing"):
            self.stream.process_state(state)

        self.assertEqual(len(self.stream.buffer), 0)

    def test_multiple_rows_rejected(self):
        state = pd.DataFrame([
            make_state(0),
            make_state(1),
        ])

        with self.assertRaisesRegex(ValueError, "exactly one row"):
            self.stream.process_state(state)

        self.assertEqual(len(self.stream.buffer), 0)

    def test_duplicate_input_columns_rejected(self):
        state = pd.DataFrame(
            [[0, 0]],
            columns=["Timestamp", "Timestamp"],
        )

        with self.assertRaisesRegex(ValueError, "Duplicate input column"):
            self.stream.process_state(state)

        self.assertEqual(len(self.stream.buffer), 0)

    def test_recovers_after_rejected_observation(self):
        for second in range(9):
            self.submit(second)

        with self.assertRaises(ValueError):
            self.submit(8)

        result = self.submit(9)

        self.assertEqual(result["status"], "PREDICTION_READY")
        self.assertEqual(self.fake.call_count, 1)
        self.assertEqual(len(self.stream.buffer), 10)

    def test_recovers_after_segment_change(self):
        for second in range(10):
            self.submit(second, segment=1)

        for second in range(10, 19):
            result = self.submit(second, segment=2)
            self.assertEqual(result["status"], "WARMING_UP")

        result = self.submit(19, segment=2)

        self.assertEqual(result["status"], "PREDICTION_READY")
        self.assertEqual(self.fake.call_count, 2)

    def test_recovers_after_timestamp_gap(self):
        for second in range(10):
            self.submit(second)

        for second in range(12, 21):
            result = self.submit(second)
            self.assertEqual(result["status"], "WARMING_UP")

        result = self.submit(21)

        self.assertEqual(result["status"], "PREDICTION_READY")
        self.assertEqual(self.fake.call_count, 2)

    def test_caller_modification_does_not_change_buffer(self):
        state = make_state(0, value=5.0)

        self.stream.process_state(state)

        state["feature_0"] = 99999.0

        buffered_value = self.stream.buffer[0]["feature_0"]

        self.assertEqual(buffered_value, 5.0)

    def test_manual_reset_clears_temporal_state(self):
        for second in range(5):
            self.submit(second)

        self.stream.reset()

        self.assertEqual(len(self.stream.buffer), 0)
        self.assertIsNone(self.stream.current_segment_id)
        self.assertIsNone(self.stream.last_timestamp)


if __name__ == "__main__":
    unittest.main(verbosity=2)
