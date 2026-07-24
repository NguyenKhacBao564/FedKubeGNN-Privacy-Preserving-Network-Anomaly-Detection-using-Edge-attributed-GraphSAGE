from __future__ import annotations

import unittest

import numpy as np

from src.federated.contracts.schema import ContractError, ModelSpec
from src.federated.contracts.task import LocalTrainResult
from src.federated.core.aggregation import weighted_fedavg


class AggregationTests(unittest.TestCase):
    def test_weighted_fedavg_uses_num_examples(self) -> None:
        results = [
            LocalTrainResult(
                state={"weight": np.array([1.0, 3.0], dtype=np.float32)},
                num_examples=1,
            ),
            LocalTrainResult(
                state={"weight": np.array([5.0, 7.0], dtype=np.float32)},
                num_examples=3,
            ),
        ]
        state = weighted_fedavg(results)
        np.testing.assert_allclose(state["weight"], np.array([4.0, 6.0]))

    def test_rejects_shape_mismatch(self) -> None:
        results = [
            LocalTrainResult(
                state={"weight": np.zeros((2,), dtype=np.float32)},
                num_examples=1,
            ),
            LocalTrainResult(
                state={"weight": np.zeros((3,), dtype=np.float32)},
                num_examples=1,
            ),
        ]
        with self.assertRaisesRegex(ContractError, "shape"):
            weighted_fedavg(results)

    def test_non_floating_state_must_be_identical(self) -> None:
        results = [
            LocalTrainResult(
                state={"counter": np.array(1, dtype=np.int64)},
                num_examples=1,
            ),
            LocalTrainResult(
                state={"counter": np.array(2, dtype=np.int64)},
                num_examples=1,
            ),
        ]
        with self.assertRaisesRegex(ContractError, "Non-floating"):
            weighted_fedavg(results)

    def test_model_contract_is_checked_before_aggregation(self) -> None:
        expected = {"weight": np.zeros((2,), dtype=np.float32)}
        spec = ModelSpec.from_state(
            family="test",
            model_version=1,
            feature_dim=1,
            num_classes=2,
            node_feature_dim=1,
            hyperparameters={},
            state=expected,
        )
        result = LocalTrainResult(
            state={"weight": np.zeros((2,), dtype=np.float64)},
            num_examples=1,
        )
        with self.assertRaisesRegex(ContractError, "dtype"):
            weighted_fedavg([result], model_spec=spec)


if __name__ == "__main__":
    unittest.main()
