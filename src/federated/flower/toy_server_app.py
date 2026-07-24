"""Runnable two-client Flower smoke server."""

from __future__ import annotations

from typing import Any

from src.federated.adapters.toy import ToyFederatedTask
from src.federated.flower.server_app import build_server_app


def _task_factory(context: Any) -> ToyFederatedTask:
    return ToyFederatedTask(seed=int(context.run_config.get("seed", 42)))


app = build_server_app(_task_factory)
