"""Framework-independent federated algorithms and evaluation."""

from src.federated.core.aggregation import weighted_fedavg
from src.federated.core.metrics import (
    aggregate_confusion_matrices,
    classification_metrics,
)
from src.federated.core.simulation import (
    FederatedRunResult,
    FederatedRoundResult,
    run_federated_simulation,
)

__all__ = [
    "FederatedRoundResult",
    "FederatedRunResult",
    "aggregate_confusion_matrices",
    "classification_metrics",
    "run_federated_simulation",
    "weighted_fedavg",
]
