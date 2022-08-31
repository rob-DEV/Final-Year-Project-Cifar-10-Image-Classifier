import unittest
import numpy as np
from src.model.scoring.metrics import Metrics


class MetricsTests(unittest.TestCase):

    def test_accuracy_binary_and_multiclass(self):

        test_cases = [
            (np.array([1, 0, 1, 1, 1, 0]), np.array([1, 0, 1, 1, 1, 0]), 1.0),
            (np.array([1, 3, 4, 9, 6, 1]), np.array(
                [1, 4, 4, 10, 6, 1]), 4.0 / 6.0),
            (np.array([1, 1, 1, 1, 1]), np.array([1, 1, 1, 1, 0]), 4.0 / 5.0),
        ]

        for y_true, y_pred, expected_accuracy in test_cases:
            with self.subTest():
                accuracy = Metrics.accuracy(y_true, y_pred)
                self.assertAlmostEquals(expected_accuracy, accuracy, 3)

    def test_precision_binary(self):
        y_true = np.array([1, 0, 0, 1, 1, 1, 0])
        y_pred = np.array([1, 1, 1, 0, 1, 1, 0])

        # TP = 3
        # FP = 2
        expected_precision = 3 / 5.0

        precision = Metrics.precision(y_true, y_pred)
        self.assertEqual(expected_precision, precision)

    def test_precision_multiclass(self):
        y_true = np.array([0, 1, 1, 1, 2, 3, 4, 4])
        y_pred = np.array([0, 0, 2, 1, 4, 3, 3, 3])

        # Precision value for each class
        expected_precision_tuple = [
            (0, 0.5),
            (1, 1.0),
            (2, 0.0),
            (3, 0.333),
            (4, 0.0),
        ]

        precision = Metrics.precision(y_true, y_pred)

        for expected_tuple, precision_tuple in zip(expected_precision_tuple, precision):
            # Assert that classes match
            self.assertEquals(expected_tuple[0], precision_tuple[0])
            # Assert that precision values are as expected (allowing some float precision / rounding errors)
            self.assertAlmostEquals(expected_tuple[1], precision_tuple[1], 3)

    def test_recall_binary(self):
        y_true = np.array([1, 0, 0, 1, 1, 1, 0])
        y_pred = np.array([1, 1, 1, 0, 1, 1, 0])

        # TP = 3
        # FN = 1
        expected_recall = 3 / 4.0

        recall = Metrics.recall(y_true, y_pred)
        self.assertEqual(expected_recall, recall)

    def test_recall_multiclass(self):
        y_true = np.array([0, 1, 1, 1, 2, 2, 3, 4, 4, 4, 4])
        y_pred = np.array([0, 0, 2, 4, 2, 2, 2, 1, 3, 3, 4])

        # Precision value for each class
        expected_recall_tuple = [
            (0, 1.0),
            (1, 0.0),
            (2, 1.0),
            (3, 0.0),
            (4, 0.25),
        ]

        recall = Metrics.recall(y_true, y_pred)

        for expected_tuple, recall_tuple in zip(expected_recall_tuple, recall):
            # Assert that classes match
            self.assertEquals(expected_tuple[0], recall_tuple[0])
            # Assert that recall values are as expected (allowing some float precision errors)
            self.assertAlmostEquals(expected_tuple[1], recall_tuple[1], 7)

    def test_f1_binary(self):
        y_true = np.array([1, 0, 0, 1, 1, 1, 0])
        y_pred = np.array([1, 1, 1, 0, 1, 1, 0])

        # TP = 3
        # FP = 2
        # FN = 1
        precision = 3.0 / 5.0
        recall = 3.0 / 4.0
        expected_f1 = round((2.0 * precision * recall) / float((precision + recall)), Metrics.DECIMAL_PLACES)

        f1 = Metrics.f1(y_true, y_pred)
        self.assertEqual(expected_f1, f1)


if __name__ == '__main__':
    unittest.main()
