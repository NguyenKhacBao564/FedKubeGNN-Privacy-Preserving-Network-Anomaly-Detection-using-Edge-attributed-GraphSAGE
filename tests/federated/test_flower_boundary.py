from __future__ import annotations

import importlib.util
import unittest

import numpy as np


FLOWER_AVAILABLE = importlib.util.find_spec("flwr") is not None


class FlowerConfigTests(unittest.TestCase):
    def test_direct_execution_has_complete_defaults(self) -> None:
        from src.federated.flower.config import resolve_run_config

        resolved = resolve_run_config({})
        self.assertEqual(resolved["num-server-rounds"], 3)
        self.assertEqual(resolved["local-epochs"], 1)
        self.assertEqual(resolved["learning-rate"], 0.15)


@unittest.skipUnless(FLOWER_AVAILABLE, "Flower is an optional Phase 2 dependency")
class FlowerBoundaryTests(unittest.TestCase):
    def test_current_message_api_apps_are_constructible(self) -> None:
        from flwr.clientapp import ClientApp
        from flwr.serverapp import ServerApp

        from src.federated.flower.toy_client_app import app as client_app
        from src.federated.flower.toy_server_app import app as server_app

        self.assertIsInstance(client_app, ClientApp)
        self.assertIsInstance(server_app, ServerApp)

    def test_confusion_matrix_callback_computes_global_metrics(self) -> None:
        from flwr.app import MetricRecord, RecordDict

        from src.federated.flower.metrics import aggregate_evaluation_records

        records = [
            RecordDict(
                {
                    "metrics": MetricRecord(
                        {
                            "num-examples": 6,
                            "loss": 0.4,
                            "num-classes": 2,
                            "confusion-matrix": [3, 1, 0, 2],
                        }
                    )
                }
            ),
            RecordDict(
                {
                    "metrics": MetricRecord(
                        {
                            "num-examples": 4,
                            "loss": 0.2,
                            "num-classes": 2,
                            "confusion-matrix": [1, 0, 1, 2],
                        }
                    )
                }
            ),
        ]
        aggregated = aggregate_evaluation_records(records, "num-examples")
        self.assertEqual(aggregated["num-examples"], 10)
        self.assertEqual(aggregated["confusion-matrix"], [4, 1, 1, 4])
        self.assertAlmostEqual(aggregated["loss"], 0.32)
        self.assertAlmostEqual(aggregated["accuracy"], 0.8)
        self.assertAlmostEqual(aggregated["macro-f1"], 0.8)

    def test_arrayrecord_preserves_named_state_schema(self) -> None:
        from flwr.app import ArrayRecord

        from src.federated.adapters.toy import ToyFederatedTask
        from src.federated.core.state import (
            arrays_to_torch_state,
            torch_state_to_arrays,
        )

        task = ToyFederatedTask()
        original = task.initial_state()
        record = ArrayRecord(
            torch_state_dict=arrays_to_torch_state(original)
        )
        restored = torch_state_to_arrays(record.to_torch_state_dict())
        task.model_spec.validate_state(restored)
        for name in original:
            np.testing.assert_array_equal(original[name], restored[name])


if __name__ == "__main__":
    unittest.main()
