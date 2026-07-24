"""Runnable two-client Flower smoke app."""

from __future__ import annotations

from typing import Any

from src.federated.adapters.toy import ToyFederatedTask
from src.federated.flower.client_app import build_client_app


def _task_factory(context: Any) -> ToyFederatedTask:
    return ToyFederatedTask(seed=int(context.run_config.get("seed", 42)))


app = build_client_app(_task_factory)
