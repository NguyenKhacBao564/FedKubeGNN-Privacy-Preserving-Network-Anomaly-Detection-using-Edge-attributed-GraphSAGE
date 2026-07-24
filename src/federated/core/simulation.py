"""Small deterministic in-process runner for proof independent of Flower."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from src.federated.contracts.task import (
    ArrayState,
    EvaluationResult,
    FederatedTask,
    LocalTrainConfig,
    LocalTrainResult,
)
from src.federated.core.aggregation import weighted_fedavg
from src.federated.core.metrics import (
    aggregate_confusion_matrices,
    classification_metrics,
)
from src.federated.core.state import copy_array_state, state_nbytes


@dataclass
class FederatedRoundResult:
    round_number: int
    participating_clients: tuple[str, ...]
    train_examples: int
    evaluation_examples: int
    train_metrics: dict[str, float]
    global_metrics: dict[str, object]
    confusion_matrix: np.ndarray
    upload_bytes: int
    download_bytes: int


@dataclass
class FederatedRunResult:
    final_state: ArrayState
    rounds: list[FederatedRoundResult] = field(default_factory=list)


def _weighted_scalar_metrics(results: Sequence[LocalTrainResult]) -> dict[str, float]:
    keys = sorted(set.intersection(*(set(result.metrics) for result in results)))
    total = float(sum(result.num_examples for result in results))
    return {
        key: float(
            sum(result.metrics[key] * result.num_examples for result in results)
            / total
        )
        for key in keys
    }


def _evaluate_clients(
    task: FederatedTask,
    client_ids: Sequence[str],
    state: Mapping[str, np.ndarray],
    *,
    split: str,
) -> tuple[list[EvaluationResult], np.ndarray, dict[str, object]]:
    results = [
        task.evaluate_local(client_id, state, split=split)
        for client_id in client_ids
    ]
    matrix = aggregate_confusion_matrices(
        (result.confusion_matrix for result in results),
        num_classes=task.label_schema.num_classes,
    )
    metrics = classification_metrics(
        matrix, class_names=task.label_schema.classes
    )
    total_examples = sum(result.num_examples for result in results)
    metrics["loss"] = (
        float(
            sum(result.loss * result.num_examples for result in results)
            / total_examples
        )
        if total_examples
        else 0.0
    )
    return results, matrix, metrics


def run_federated_simulation(
    task: FederatedTask,
    *,
    num_rounds: int,
    train_config: LocalTrainConfig,
    client_ids: Sequence[str] | None = None,
    evaluate_split: str = "test",
) -> FederatedRunResult:
    """Run full-participation FedAvg through the public task contract."""
    if num_rounds < 1:
        raise ValueError("num_rounds must be >= 1.")
    participants = tuple(client_ids or task.client_ids)
    if not participants:
        raise ValueError("At least one client is required.")
    unknown = sorted(set(participants) - set(task.client_ids))
    if unknown:
        raise KeyError(f"Unknown client ids: {unknown}.")

    state = task.initial_state()
    task.model_spec.validate_state(state)
    payload_bytes = state_nbytes(state)
    round_results: list[FederatedRoundResult] = []

    for round_number in range(1, num_rounds + 1):
        local_results = [
            task.train_local(client_id, copy_array_state(state), train_config)
            for client_id in participants
        ]
        state = weighted_fedavg(local_results, model_spec=task.model_spec)
        evaluations, matrix, metrics = _evaluate_clients(
            task, participants, state, split=evaluate_split
        )
        round_results.append(
            FederatedRoundResult(
                round_number=round_number,
                participating_clients=participants,
                train_examples=sum(
                    result.num_examples for result in local_results
                ),
                evaluation_examples=sum(
                    result.num_examples for result in evaluations
                ),
                train_metrics=_weighted_scalar_metrics(local_results),
                global_metrics=metrics,
                confusion_matrix=matrix,
                upload_bytes=payload_bytes * len(participants),
                download_bytes=payload_bytes * len(participants),
            )
        )
    return FederatedRunResult(
        final_state=copy_array_state(state), rounds=round_results
    )
