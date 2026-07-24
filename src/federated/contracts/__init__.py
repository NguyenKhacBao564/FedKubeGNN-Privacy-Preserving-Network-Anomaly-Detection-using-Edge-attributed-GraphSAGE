"""Versioned boundary contracts used by the federated core and adapters."""

from src.federated.contracts.artifacts import ContractBundle
from src.federated.contracts.schema import (
    ContractError,
    FeatureField,
    FeatureSchema,
    GraphSchema,
    LabelSchema,
    ModelSpec,
    ParameterSpec,
)
from src.federated.contracts.task import (
    ArrayState,
    EvaluationResult,
    FederatedTask,
    LocalTrainConfig,
    LocalTrainResult,
)

__all__ = [
    "ArrayState",
    "ContractBundle",
    "ContractError",
    "EvaluationResult",
    "FeatureField",
    "FeatureSchema",
    "FederatedTask",
    "GraphSchema",
    "LabelSchema",
    "LocalTrainConfig",
    "LocalTrainResult",
    "ModelSpec",
    "ParameterSpec",
]
