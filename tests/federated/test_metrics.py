from __future__ import annotations

import unittest

import numpy as np

from src.federated.core.metrics import (
    aggregate_confusion_matrices,
    classification_metrics,
    confusion_matrix_from_predictions,
)


class MetricTests(unittest.TestCase):
    def test_aggregate_confusion_matches_union_predictions(self) -> None:
        cm0 = confusion_matrix_from_predictions(
            [0, 0, 1], [0, 1, 1], num_classes=2
        )
        cm1 = confusion_matrix_from_predictions(
            [0, 1, 1, 1], [0, 0, 1, 1], num_classes=2
        )
        union = confusion_matrix_from_predictions(
            [0, 0, 1, 0, 1, 1, 1],
            [0, 1, 1, 0, 0, 1, 1],
            num_classes=2,
        )
        np.testing.assert_array_equal(
            aggregate_confusion_matrices([cm0, cm1]), union
        )

    def test_fixed_class_macro_f1_counts_absent_class_as_zero(self) -> None:
        matrix = np.array([[4, 0], [0, 0]], dtype=np.int64)
        metrics = classification_metrics(matrix, class_names=("a", "b"))
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 0.5)
        self.assertEqual(metrics["per_class"]["b"]["f1"], 0.0)


if __name__ == "__main__":
    unittest.main()
