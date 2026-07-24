from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.federated.contracts.artifacts import ContractBundle
from src.federated.contracts.schema import (
    ContractError,
    FeatureSchema,
    GraphSchema,
    LabelSchema,
    ModelSpec,
)


class ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.features = FeatureSchema.from_names(("f0", "f1"))
        self.labels = LabelSchema(("benign", "attack"))
        self.graph = GraphSchema(
            feature_schema_digest=self.features.digest,
            label_schema_digest=self.labels.digest,
        )
        state = {
            "weight": np.zeros((2, 2), dtype=np.float32),
            "bias": np.zeros((2,), dtype=np.float32),
        }
        self.model = ModelSpec.from_state(
            family="linear",
            model_version=1,
            feature_dim=2,
            num_classes=2,
            node_feature_dim=1,
            hyperparameters={},
            state=state,
        )

    def test_schema_digest_changes_when_feature_order_changes(self) -> None:
        reversed_schema = FeatureSchema.from_names(("f1", "f0"))
        self.assertNotEqual(self.features.digest, reversed_schema.digest)

    def test_model_state_rejects_key_reordering(self) -> None:
        with self.assertRaisesRegex(ContractError, "keys/order"):
            self.model.validate_state(
                {
                    "bias": np.zeros((2,), dtype=np.float32),
                    "weight": np.zeros((2, 2), dtype=np.float32),
                }
            )

    def test_contract_bundle_round_trip_and_checksum(self) -> None:
        bundle = ContractBundle(
            feature_schema=self.features,
            label_schema=self.labels,
            graph_schema=self.graph,
            model_spec=self.model,
            categories={"protocol": ("tcp", "udp")},
            learned_arrays={
                "scaler_mean": np.array([1.0, 2.0]),
                "scaler_scale": np.array([0.5, 0.25]),
            },
            metadata={"source": "unit-test"},
        )
        with tempfile.TemporaryDirectory() as directory:
            root = bundle.write(Path(directory) / "bundle")
            loaded = ContractBundle.load(root)
            self.assertEqual(loaded.feature_schema.digest, self.features.digest)
            self.assertEqual(loaded.label_schema.classes, self.labels.classes)
            np.testing.assert_array_equal(
                loaded.learned_arrays["scaler_mean"], np.array([1.0, 2.0])
            )

            feature_path = root / "feature_schema.json"
            value = json.loads(feature_path.read_text(encoding="utf-8"))
            value["fields"][0]["name"] = "tampered"
            feature_path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "Checksum mismatch"):
                ContractBundle.load(root)


if __name__ == "__main__":
    unittest.main()
