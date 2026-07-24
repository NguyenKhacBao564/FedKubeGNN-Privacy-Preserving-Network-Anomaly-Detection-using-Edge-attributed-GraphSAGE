"""Classification metrics computed from additive sufficient statistics."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def _validate_matrix(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("Confusion matrix must be square.")
    if np.any(value < 0):
        raise ValueError("Confusion matrix counts must be non-negative.")
    if not np.all(np.equal(value, np.floor(value))):
        raise ValueError("Confusion matrix must contain integer counts.")
    return value.astype(np.int64, copy=False)


def confusion_matrix_from_predictions(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    *,
    num_classes: int,
) -> np.ndarray:
    """Build a fixed-K confusion matrix without a scikit-learn dependency."""
    if num_classes < 1:
        raise ValueError("num_classes must be >= 1.")
    truth = np.asarray(y_true, dtype=np.int64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if truth.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if truth.size and (
        truth.min() < 0
        or pred.min() < 0
        or truth.max() >= num_classes
        or pred.max() >= num_classes
    ):
        raise ValueError("Predictions/labels are outside [0, num_classes).")
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    if truth.size:
        np.add.at(matrix, (truth, pred), 1)
    return matrix


def aggregate_confusion_matrices(
    matrices: Iterable[np.ndarray],
    *,
    num_classes: int | None = None,
) -> np.ndarray:
    """Sum client confusion matrices, preserving true global macro-F1."""
    items = [_validate_matrix(matrix) for matrix in matrices]
    if not items:
        if num_classes is None or num_classes < 1:
            raise ValueError(
                "num_classes is required when aggregating no matrices."
            )
        return np.zeros((num_classes, num_classes), dtype=np.int64)
    shape = items[0].shape
    if num_classes is not None and shape != (num_classes, num_classes):
        raise ValueError(
            f"Matrix shape {shape} does not match num_classes={num_classes}."
        )
    if any(matrix.shape != shape for matrix in items[1:]):
        raise ValueError("All confusion matrices must use the same class space.")
    return np.sum(np.stack(items, axis=0), axis=0, dtype=np.int64)


def classification_metrics(
    confusion_matrix: np.ndarray,
    *,
    class_names: Sequence[str] | None = None,
) -> dict[str, object]:
    """Compute fixed-class metrics using the same zero-division rule as Phase 1."""
    matrix = _validate_matrix(confusion_matrix)
    num_classes = matrix.shape[0]
    if class_names is None:
        names = tuple(f"class_{index}" for index in range(num_classes))
    else:
        names = tuple(str(name) for name in class_names)
        if len(names) != num_classes:
            raise ValueError("class_names length must match confusion matrix size.")

    true_positive = np.diag(matrix).astype(np.float64)
    support = matrix.sum(axis=1).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros_like(true_positive),
        where=predicted > 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros_like(true_positive),
        where=support > 0,
    )
    denominator = precision + recall
    f1 = np.divide(
        2.0 * precision * recall,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator > 0,
    )
    total = float(support.sum())
    accuracy = float(true_positive.sum() / total) if total else 0.0
    macro_f1 = float(f1.mean())
    weighted_f1 = float(np.dot(f1, support) / total) if total else 0.0

    per_class = {
        name: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, name in enumerate(names)
    }
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "num_examples": int(total),
        "per_class": per_class,
    }
