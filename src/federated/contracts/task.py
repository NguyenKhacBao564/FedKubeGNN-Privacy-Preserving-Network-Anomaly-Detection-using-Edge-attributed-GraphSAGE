"""Task protocol separating federated orchestration from Phase 1 code."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np

from src.federated.contracts.schema import (
    FeatureSchema,
    GraphSchema,
    LabelSchema,
    ModelSpec,
)


ArrayState = dict[str, np.ndarray]


@dataclass(frozen=True)
class LocalTrainConfig:
    """Settings controlled by one federated run, not by a task adapter."""

    local_epochs: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    optimizer: str = "sgd"
    proximal_mu: float = 0.0
    seed: int = 42

    def __post_init__(self) -> None:
        if self.local_epochs < 1:
            raise ValueError("local_epochs must be >= 1.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0.")
        if self.weight_decay < 0 or self.grad_clip < 0:
            raise ValueError("weight_decay and grad_clip must be >= 0.")
        if self.optimizer not in {"sgd", "adam", "adamw"}:
            raise ValueError("optimizer must be one of: sgd, adam, adamw.")
        if self.proximal_mu < 0:
            raise ValueError("proximal_mu must be >= 0.")


@dataclass
class LocalTrainResult:
    """One client's updated state and aggregation weight."""

    state: ArrayState
    num_examples: int
    metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.num_examples <= 0:
            raise ValueError("LocalTrainResult.num_examples must be > 0.")


@dataclass
class EvaluationResult:
    """Sufficient statistics for correct global classification metrics."""

    confusion_matrix: np.ndarray
    num_examples: int
    loss: float
    metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        matrix = np.asarray(self.confusion_matrix)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("confusion_matrix must be a square 2-D matrix.")
        if np.any(matrix < 0):
            raise ValueError("confusion_matrix must contain non-negative counts.")
        if self.num_examples < 0:
            raise ValueError("EvaluationResult.num_examples must be >= 0.")
        if int(matrix.sum()) != self.num_examples:
            raise ValueError(
                "confusion_matrix sum must equal EvaluationResult.num_examples."
            )
        self.confusion_matrix = matrix.astype(np.int64, copy=False)


@runtime_checkable
class FederatedTask(Protocol):
    """Plugin contract consumed by the core and Flower boundary."""

    @property
    def task_id(self) -> str:
        ...

    @property
    def client_ids(self) -> tuple[str, ...]:
        ...

    @property
    def feature_schema(self) -> FeatureSchema:
        ...

    @property
    def label_schema(self) -> LabelSchema:
        ...

    @property
    def graph_schema(self) -> GraphSchema | None:
        ...

    @property
    def model_spec(self) -> ModelSpec:
        ...

    def initial_state(self) -> ArrayState:
        ...

    def train_local(
        self,
        client_id: str,
        global_state: Mapping[str, np.ndarray],
        config: LocalTrainConfig,
    ) -> LocalTrainResult:
        ...

    def evaluate_local(
        self,
        client_id: str,
        state: Mapping[str, np.ndarray],
        *,
        split: str,
    ) -> EvaluationResult:
        ...

    def metadata(self) -> Mapping[str, Any]:
        ...
