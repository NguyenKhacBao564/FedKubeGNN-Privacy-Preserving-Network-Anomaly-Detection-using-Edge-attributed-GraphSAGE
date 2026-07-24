"""Adapters connecting concrete ML tasks to the federated contracts."""

from src.federated.adapters.phase1_iot23 import (
    Phase1AdapterError,
    Phase1IoT23Task,
    make_phase1_model_factory,
)
from src.federated.adapters.toy import ToyFederatedTask

__all__ = [
    "Phase1AdapterError",
    "Phase1IoT23Task",
    "ToyFederatedTask",
    "make_phase1_model_factory",
]
