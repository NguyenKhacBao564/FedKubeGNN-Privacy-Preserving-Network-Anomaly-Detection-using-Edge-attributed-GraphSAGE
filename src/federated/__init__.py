"""Reusable federated-learning foundation.

The package deliberately depends on contracts instead of Phase 1 modules.
Phase-specific code belongs in :mod:`src.federated.adapters`.
"""

from src.federated.contracts.schema import (
    ContractError,
    FeatureField,
    FeatureSchema,
    GraphSchema,
    LabelSchema,
    ModelSpec,
    ParameterSpec,
)

__all__ = [
    "ContractError",
    "FeatureField",
    "FeatureSchema",
    "GraphSchema",
    "LabelSchema",
    "ModelSpec",
    "ParameterSpec",
]
