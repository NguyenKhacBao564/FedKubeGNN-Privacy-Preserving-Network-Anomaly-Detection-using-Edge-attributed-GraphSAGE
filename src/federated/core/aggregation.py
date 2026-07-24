"""Strict named-parameter implementations of federated aggregation."""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

from src.federated.contracts.schema import ContractError, ModelSpec
from src.federated.contracts.task import ArrayState, LocalTrainResult


def _validate_state_against_reference(
    state: Mapping[str, np.ndarray],
    reference: Mapping[str, np.ndarray],
    *,
    client_index: int,
) -> None:
    if tuple(state.keys()) != tuple(reference.keys()):
        raise ContractError(
            f"Client {client_index} state keys/order differ from client 0."
        )
    for name, reference_value in reference.items():
        value = np.asarray(state[name])
        reference_array = np.asarray(reference_value)
        if value.shape != reference_array.shape:
            raise ContractError(
                f"Client {client_index} parameter '{name}' shape "
                f"{value.shape} != {reference_array.shape}."
            )
        if value.dtype != reference_array.dtype:
            raise ContractError(
                f"Client {client_index} parameter '{name}' dtype "
                f"{value.dtype} != {reference_array.dtype}."
            )


def weighted_fedavg(
    results: Iterable[LocalTrainResult],
    *,
    model_spec: ModelSpec | None = None,
) -> ArrayState:
    """Aggregate named states with sample weighting and fail-closed validation.

    Floating and complex values are averaged. Non-floating state entries are
    copied only when every client has exactly the same value; silently averaging
    counters or booleans would not have a well-defined model meaning.
    """
    items = list(results)
    if not items:
        raise ValueError("weighted_fedavg requires at least one client result.")
    if any(item.num_examples <= 0 for item in items):
        raise ValueError("Every client aggregation weight must be positive.")

    reference = items[0].state
    if not reference:
        raise ContractError("Cannot aggregate an empty model state.")
    for index, item in enumerate(items):
        _validate_state_against_reference(
            item.state, reference, client_index=index
        )
        if model_spec is not None:
            model_spec.validate_state(item.state)

    total_weight = float(sum(item.num_examples for item in items))
    aggregated: ArrayState = {}
    for name, reference_value in reference.items():
        reference_array = np.asarray(reference_value)
        if np.issubdtype(reference_array.dtype, np.floating):
            accumulator = np.zeros(reference_array.shape, dtype=np.float64)
            for item in items:
                value = np.asarray(item.state[name], dtype=np.float64)
                if not np.all(np.isfinite(value)):
                    raise ContractError(
                        f"Client state parameter '{name}' contains NaN/Inf."
                    )
                accumulator += value * (item.num_examples / total_weight)
            aggregated[name] = accumulator.astype(reference_array.dtype)
        elif np.issubdtype(reference_array.dtype, np.complexfloating):
            accumulator = np.zeros(reference_array.shape, dtype=np.complex128)
            for item in items:
                accumulator += np.asarray(item.state[name], dtype=np.complex128) * (
                    item.num_examples / total_weight
                )
            aggregated[name] = accumulator.astype(reference_array.dtype)
        else:
            for item in items[1:]:
                if not np.array_equal(reference_array, item.state[name]):
                    raise ContractError(
                        f"Non-floating state entry '{name}' differs across clients."
                    )
            aggregated[name] = reference_array.copy()
    if model_spec is not None:
        model_spec.validate_state(aggregated)
    return aggregated
