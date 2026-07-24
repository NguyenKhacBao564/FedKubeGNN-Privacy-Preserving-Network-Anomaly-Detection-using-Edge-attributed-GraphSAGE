"""Portable, versioned contracts for data, graphs, and model parameters."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


CURRENT_CONTRACT_VERSION = 1


class ContractError(ValueError):
    """Raised when an adapter or artifact violates a federated contract."""


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _normalise_dtype(dtype: Any) -> str:
    text = str(dtype)
    for prefix in ("torch.", "numpy."):
        if text.startswith(prefix):
            text = text[len(prefix):]
    return text


@dataclass(frozen=True)
class FeatureField:
    """One ordered edge-feature field at the Phase 2 boundary."""

    name: str
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ContractError("FeatureField.name must not be empty.")
        if not self.dtype or not self.dtype.strip():
            raise ContractError(f"Feature '{self.name}' has an empty dtype.")

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "dtype": self.dtype}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FeatureField":
        return cls(name=str(value["name"]), dtype=str(value["dtype"]))


@dataclass(frozen=True)
class FeatureSchema:
    """Ordered model-input features; order is part of the contract."""

    fields: tuple[FeatureField, ...]
    version: int = CURRENT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.version < 1:
            raise ContractError("FeatureSchema.version must be >= 1.")
        if not self.fields:
            raise ContractError("FeatureSchema must contain at least one field.")
        names = [field.name for field in self.fields]
        if len(names) != len(set(names)):
            raise ContractError("FeatureSchema contains duplicate feature names.")

    @classmethod
    def from_names(
        cls,
        names: Iterable[str],
        *,
        dtype: str = "float32",
        version: int = CURRENT_CONTRACT_VERSION,
    ) -> "FeatureSchema":
        return cls(
            fields=tuple(FeatureField(str(name), dtype) for name in names),
            version=version,
        )

    @property
    def feature_dim(self) -> int:
        return len(self.fields)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.fields)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "fields": [field.to_dict() for field in self.fields],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FeatureSchema":
        return cls(
            fields=tuple(
                FeatureField.from_dict(field) for field in value["fields"]
            ),
            version=int(value["version"]),
        )

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())


@dataclass(frozen=True)
class LabelSchema:
    """Stable ordered class vocabulary shared by every client."""

    classes: tuple[str, ...]
    version: int = CURRENT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.version < 1:
            raise ContractError("LabelSchema.version must be >= 1.")
        if not self.classes:
            raise ContractError("LabelSchema must contain at least one class.")
        if any(not name or not name.strip() for name in self.classes):
            raise ContractError("LabelSchema contains an empty class name.")
        if len(self.classes) != len(set(self.classes)):
            raise ContractError("LabelSchema contains duplicate classes.")

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def class_to_idx(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.classes)}

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "classes": list(self.classes)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LabelSchema":
        return cls(
            classes=tuple(str(name) for name in value["classes"]),
            version=int(value["version"]),
        )

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())


@dataclass(frozen=True)
class GraphSchema:
    """Observable graph layout expected by a graph-task adapter."""

    feature_schema_digest: str
    label_schema_digest: str
    node_feature_dim: int = 1
    node_semantics: str = "ip"
    edge_semantics: str = "network_flow"
    target_semantics: str = "edge_classification"
    message_passing_bidirectional: bool = True
    required_fields: tuple[str, ...] = (
        "x",
        "edge_index",
        "edge_attr",
        "edge_label",
        "edge_index_mp",
        "edge_attr_mp",
        "train_mask",
        "val_mask",
        "test_mask",
    )
    version: int = CURRENT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.version < 1:
            raise ContractError("GraphSchema.version must be >= 1.")
        if self.node_feature_dim < 1:
            raise ContractError("GraphSchema.node_feature_dim must be >= 1.")
        if not self.feature_schema_digest or not self.label_schema_digest:
            raise ContractError("GraphSchema requires feature and label digests.")
        if len(self.required_fields) != len(set(self.required_fields)):
            raise ContractError("GraphSchema.required_fields contains duplicates.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "feature_schema_digest": self.feature_schema_digest,
            "label_schema_digest": self.label_schema_digest,
            "node_feature_dim": self.node_feature_dim,
            "node_semantics": self.node_semantics,
            "edge_semantics": self.edge_semantics,
            "target_semantics": self.target_semantics,
            "message_passing_bidirectional": self.message_passing_bidirectional,
            "required_fields": list(self.required_fields),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GraphSchema":
        return cls(
            version=int(value["version"]),
            feature_schema_digest=str(value["feature_schema_digest"]),
            label_schema_digest=str(value["label_schema_digest"]),
            node_feature_dim=int(value["node_feature_dim"]),
            node_semantics=str(value["node_semantics"]),
            edge_semantics=str(value["edge_semantics"]),
            target_semantics=str(value["target_semantics"]),
            message_passing_bidirectional=bool(
                value["message_passing_bidirectional"]
            ),
            required_fields=tuple(str(name) for name in value["required_fields"]),
        )

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())


@dataclass(frozen=True)
class ParameterSpec:
    """Name, shape, and dtype of one state entry sent through federation."""

    name: str
    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ContractError("ParameterSpec.name must not be empty.")
        if any(dimension < 0 for dimension in self.shape):
            raise ContractError(f"Parameter '{self.name}' has a negative shape.")
        if not self.dtype:
            raise ContractError(f"Parameter '{self.name}' has an empty dtype.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ParameterSpec":
        return cls(
            name=str(value["name"]),
            shape=tuple(int(size) for size in value["shape"]),
            dtype=str(value["dtype"]),
        )


@dataclass(frozen=True)
class ModelSpec:
    """Architecture identity plus the exact state schema to aggregate."""

    family: str
    model_version: int
    feature_dim: int
    num_classes: int
    node_feature_dim: int
    hyperparameters: Mapping[str, Any] = field(default_factory=dict)
    parameters: tuple[ParameterSpec, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.family:
            raise ContractError("ModelSpec.family must not be empty.")
        if self.model_version < 1:
            raise ContractError("ModelSpec.model_version must be >= 1.")
        if self.feature_dim < 1 or self.num_classes < 1:
            raise ContractError("ModelSpec dimensions must be positive.")
        if self.node_feature_dim < 1:
            raise ContractError("ModelSpec.node_feature_dim must be positive.")
        names = [parameter.name for parameter in self.parameters]
        if len(names) != len(set(names)):
            raise ContractError("ModelSpec contains duplicate parameter names.")
        # Fail early if metadata cannot be encoded portably.
        try:
            _canonical_json(dict(self.hyperparameters))
        except (TypeError, ValueError) as exc:
            raise ContractError(
                "ModelSpec.hyperparameters must be JSON serializable."
            ) from exc

    @classmethod
    def from_state(
        cls,
        *,
        family: str,
        model_version: int,
        feature_dim: int,
        num_classes: int,
        node_feature_dim: int,
        hyperparameters: Mapping[str, Any],
        state: Mapping[str, Any],
    ) -> "ModelSpec":
        parameters = tuple(
            ParameterSpec(
                name=str(name),
                shape=tuple(int(size) for size in value.shape),
                dtype=_normalise_dtype(value.dtype),
            )
            for name, value in state.items()
        )
        return cls(
            family=family,
            model_version=model_version,
            feature_dim=feature_dim,
            num_classes=num_classes,
            node_feature_dim=node_feature_dim,
            hyperparameters=dict(hyperparameters),
            parameters=parameters,
        )

    def validate_state(self, state: Mapping[str, Any]) -> None:
        expected_names = tuple(parameter.name for parameter in self.parameters)
        actual_names = tuple(str(name) for name in state.keys())
        if actual_names != expected_names:
            raise ContractError(
                "Model state keys/order do not match the contract: "
                f"expected={expected_names}, actual={actual_names}."
            )
        for parameter in self.parameters:
            value = state[parameter.name]
            shape = tuple(int(size) for size in value.shape)
            dtype = _normalise_dtype(value.dtype)
            if shape != parameter.shape:
                raise ContractError(
                    f"Parameter '{parameter.name}' shape {shape} does not match "
                    f"{parameter.shape}."
                )
            if dtype != parameter.dtype:
                raise ContractError(
                    f"Parameter '{parameter.name}' dtype {dtype} does not match "
                    f"{parameter.dtype}."
                )

    def assert_architecture_compatible(self, other: "ModelSpec") -> None:
        if self.digest != other.digest:
            raise ContractError(
                "Model contracts are incompatible: "
                f"expected={self.digest}, actual={other.digest}."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "model_version": self.model_version,
            "feature_dim": self.feature_dim,
            "num_classes": self.num_classes,
            "node_feature_dim": self.node_feature_dim,
            "hyperparameters": dict(self.hyperparameters),
            "parameters": [parameter.to_dict() for parameter in self.parameters],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModelSpec":
        return cls(
            family=str(value["family"]),
            model_version=int(value["model_version"]),
            feature_dim=int(value["feature_dim"]),
            num_classes=int(value["num_classes"]),
            node_feature_dim=int(value["node_feature_dim"]),
            hyperparameters=dict(value.get("hyperparameters", {})),
            parameters=tuple(
                ParameterSpec.from_dict(parameter)
                for parameter in value.get("parameters", [])
            ),
        )

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())


def validate_exact_keys(
    mapping: Mapping[str, Any],
    expected: Sequence[str],
    *,
    context: str,
) -> None:
    """Validate exact key order for serialization-sensitive mappings."""
    actual = tuple(mapping.keys())
    expected_tuple = tuple(expected)
    if actual != expected_tuple:
        raise ContractError(
            f"{context}: expected keys {expected_tuple}, got {actual}."
        )
