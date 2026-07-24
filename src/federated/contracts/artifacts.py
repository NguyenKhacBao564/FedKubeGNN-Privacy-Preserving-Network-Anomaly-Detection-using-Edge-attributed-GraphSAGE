"""Portable contract bundles with integrity verification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.federated.contracts.schema import (
    ContractError,
    FeatureSchema,
    GraphSchema,
    LabelSchema,
    ModelSpec,
)


BUNDLE_VERSION = 1


def _json_text(value: Mapping[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class ContractBundle:
    """Portable boundary artifact independent of Phase 1 Python class paths."""

    feature_schema: FeatureSchema
    label_schema: LabelSchema
    graph_schema: GraphSchema
    model_spec: ModelSpec | None = None
    categories: dict[str, tuple[str, ...]] = field(default_factory=dict)
    learned_arrays: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.graph_schema.feature_schema_digest != self.feature_schema.digest:
            raise ContractError("Graph schema references a different feature schema.")
        if self.graph_schema.label_schema_digest != self.label_schema.digest:
            raise ContractError("Graph schema references a different label schema.")
        if self.model_spec is not None:
            if self.model_spec.feature_dim != self.feature_schema.feature_dim:
                raise ContractError("ModelSpec.feature_dim does not match features.")
            if self.model_spec.num_classes != self.label_schema.num_classes:
                raise ContractError("ModelSpec.num_classes does not match labels.")

    def write(self, directory: str | Path) -> Path:
        """Write into a new or empty directory; never overwrite existing files."""
        root = Path(directory)
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(
                f"Contract bundle directory is not empty: {root}"
            )
        root.mkdir(parents=True, exist_ok=True)

        files: dict[str, Mapping[str, Any]] = {
            "feature_schema.json": self.feature_schema.to_dict(),
            "label_schema.json": self.label_schema.to_dict(),
            "graph_schema.json": self.graph_schema.to_dict(),
            "categories.json": {
                key: list(values) for key, values in sorted(self.categories.items())
            },
        }
        if self.model_spec is not None:
            files["model_spec.json"] = self.model_spec.to_dict()

        for filename, value in files.items():
            (root / filename).write_text(_json_text(value), encoding="utf-8")

        if self.learned_arrays:
            np.savez(
                root / "learned_arrays.npz",
                **{
                    key: np.asarray(value)
                    for key, value in sorted(self.learned_arrays.items())
                },
            )

        data_files = sorted(path.name for path in root.iterdir() if path.is_file())
        checksums = {name: _sha256(root / name) for name in data_files}
        manifest = {
            "bundle_version": BUNDLE_VERSION,
            "feature_schema_digest": self.feature_schema.digest,
            "label_schema_digest": self.label_schema.digest,
            "graph_schema_digest": self.graph_schema.digest,
            "model_spec_digest": (
                self.model_spec.digest if self.model_spec is not None else None
            ),
            "files": data_files,
            "metadata": self.metadata,
        }
        (root / "manifest.json").write_text(
            _json_text(manifest), encoding="utf-8"
        )
        checksums["manifest.json"] = _sha256(root / "manifest.json")
        (root / "checksums.json").write_text(
            _json_text(checksums), encoding="utf-8"
        )
        return root

    @classmethod
    def load(cls, directory: str | Path) -> "ContractBundle":
        root = Path(directory)
        if not root.is_dir():
            raise FileNotFoundError(f"Contract bundle directory not found: {root}")

        checksums = json.loads(
            (root / "checksums.json").read_text(encoding="utf-8")
        )
        for filename, expected in checksums.items():
            path = root / filename
            if not path.is_file():
                raise ContractError(f"Contract bundle is missing '{filename}'.")
            actual = _sha256(path)
            if actual != expected:
                raise ContractError(
                    f"Checksum mismatch for '{filename}': "
                    f"expected={expected}, actual={actual}."
                )

        manifest = json.loads(
            (root / "manifest.json").read_text(encoding="utf-8")
        )
        if int(manifest["bundle_version"]) != BUNDLE_VERSION:
            raise ContractError(
                f"Unsupported bundle_version={manifest['bundle_version']}."
            )

        def read_json(name: str) -> dict[str, Any]:
            return json.loads((root / name).read_text(encoding="utf-8"))

        feature_schema = FeatureSchema.from_dict(read_json("feature_schema.json"))
        label_schema = LabelSchema.from_dict(read_json("label_schema.json"))
        graph_schema = GraphSchema.from_dict(read_json("graph_schema.json"))
        model_path = root / "model_spec.json"
        model_spec = (
            ModelSpec.from_dict(read_json("model_spec.json"))
            if model_path.exists()
            else None
        )
        categories = {
            key: tuple(str(value) for value in values)
            for key, values in read_json("categories.json").items()
        }
        arrays_path = root / "learned_arrays.npz"
        learned_arrays: dict[str, np.ndarray] = {}
        if arrays_path.exists():
            with np.load(arrays_path, allow_pickle=False) as archive:
                learned_arrays = {
                    key: np.asarray(archive[key]).copy() for key in archive.files
                }

        bundle = cls(
            feature_schema=feature_schema,
            label_schema=label_schema,
            graph_schema=graph_schema,
            model_spec=model_spec,
            categories=categories,
            learned_arrays=learned_arrays,
            metadata=dict(manifest.get("metadata", {})),
        )
        expected_digests = {
            "feature_schema_digest": feature_schema.digest,
            "label_schema_digest": label_schema.digest,
            "graph_schema_digest": graph_schema.digest,
            "model_spec_digest": model_spec.digest if model_spec else None,
        }
        for key, actual in expected_digests.items():
            if manifest.get(key) != actual:
                raise ContractError(
                    f"Manifest {key} does not match the loaded contract."
                )
        return bundle
