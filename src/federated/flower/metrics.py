"""Flower callback preserving global classification metrics."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.federated.core.metrics import (
    aggregate_confusion_matrices,
    classification_metrics,
)


def aggregate_evaluation_records(
    records: list[Any], weighting_metric_name: str
) -> Any:
    """Aggregate additive confusion matrices instead of client macro-F1."""
    try:
        from flwr.app import MetricRecord
    except ImportError as exc:  # pragma: no cover - dependency-specific
        raise RuntimeError("Flower is required for metric aggregation.") from exc
    if not records:
        return MetricRecord()

    matrices: list[np.ndarray] = []
    losses: list[tuple[float, int]] = []
    num_classes: int | None = None
    for record in records:
        metric_records = list(record.metric_records.values())
        if len(metric_records) != 1:
            raise ValueError("Each Flower reply must contain one MetricRecord.")
        metrics = metric_records[0]
        current_classes = int(metrics["num-classes"])
        if num_classes is None:
            num_classes = current_classes
        elif current_classes != num_classes:
            raise ValueError("Clients returned different num-classes values.")
        flat = np.asarray(metrics["confusion-matrix"], dtype=np.int64)
        if flat.size != current_classes * current_classes:
            raise ValueError("Client confusion matrix has the wrong size.")
        matrix = flat.reshape(current_classes, current_classes)
        weight = int(metrics[weighting_metric_name])
        if int(matrix.sum()) != weight:
            raise ValueError(
                "Client confusion matrix sum differs from aggregation weight."
            )
        matrices.append(matrix)
        losses.append((float(metrics["loss"]), weight))

    assert num_classes is not None
    matrix = aggregate_confusion_matrices(
        matrices, num_classes=num_classes
    )
    computed = classification_metrics(matrix)
    total_examples = int(matrix.sum())
    weighted_loss = (
        sum(loss * weight for loss, weight in losses) / total_examples
        if total_examples
        else 0.0
    )
    output: dict[str, int | float | list[int]] = {
        weighting_metric_name: total_examples,
        "loss": float(weighted_loss),
        "accuracy": float(computed["accuracy"]),
        "macro-f1": float(computed["macro_f1"]),
        "weighted-f1": float(computed["weighted_f1"]),
        "num-classes": num_classes,
        "confusion-matrix": matrix.reshape(-1).tolist(),
    }
    per_class = computed["per_class"]
    for index in range(num_classes):
        values = per_class[f"class_{index}"]
        output[f"f1-class-{index}"] = float(values["f1"])
        output[f"support-class-{index}"] = int(values["support"])
    return MetricRecord(output)
