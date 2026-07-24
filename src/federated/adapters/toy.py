"""Deterministic toy task proving Phase 2 does not depend on Phase 1."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.federated.contracts.schema import (
    FeatureSchema,
    LabelSchema,
    ModelSpec,
)
from src.federated.contracts.task import (
    ArrayState,
    EvaluationResult,
    LocalTrainConfig,
    LocalTrainResult,
)
from src.federated.core.metrics import confusion_matrix_from_predictions
from src.federated.core.state import arrays_to_torch_state, torch_state_to_arrays


@dataclass(frozen=True)
class _ToyPartition:
    train_x: torch.Tensor
    train_y: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor


class _ToyLinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(2, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


def _stable_client_seed(base_seed: int, client_id: str) -> int:
    suffix = int.from_bytes(
        hashlib.sha256(client_id.encode("utf-8")).digest()[:4], "big"
    )
    return (base_seed + suffix) % (2**31)


def _make_examples(
    rng: np.random.Generator,
    *,
    n_class_zero: int,
    n_class_one: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    zero = rng.normal(loc=(-1.0, -0.8), scale=0.65, size=(n_class_zero, 2))
    one = rng.normal(loc=(1.0, 0.8), scale=0.65, size=(n_class_one, 2))
    features = np.concatenate([zero, one], axis=0).astype(np.float32)
    labels = np.concatenate(
        [
            np.zeros(n_class_zero, dtype=np.int64),
            np.ones(n_class_one, dtype=np.int64),
        ]
    )
    order = rng.permutation(len(labels))
    return torch.from_numpy(features[order]), torch.from_numpy(labels[order])


class ToyFederatedTask:
    """Two non-IID clients training a small linear classifier."""

    def __init__(self, *, seed: int = 42) -> None:
        self._seed = int(seed)
        self._feature_schema = FeatureSchema.from_names(("x0", "x1"))
        self._label_schema = LabelSchema(("negative", "positive"))
        self._client_ids = ("toy-client-0", "toy-client-1")
        self._partitions = self._build_partitions()

        torch.manual_seed(self._seed)
        initial_model = _ToyLinearModel()
        initial_state = initial_model.state_dict()
        self._model_spec = ModelSpec.from_state(
            family="toy-linear",
            model_version=1,
            feature_dim=2,
            num_classes=2,
            node_feature_dim=1,
            hyperparameters={"bias": True},
            state=initial_state,
        )
        self._initial_state = torch_state_to_arrays(initial_state)

    def _build_partitions(self) -> dict[str, _ToyPartition]:
        partitions: dict[str, _ToyPartition] = {}
        for index, client_id in enumerate(self._client_ids):
            rng = np.random.default_rng(_stable_client_seed(self._seed, client_id))
            if index == 0:
                train_counts = (72, 18)
            else:
                train_counts = (18, 72)
            train_x, train_y = _make_examples(
                rng,
                n_class_zero=train_counts[0],
                n_class_one=train_counts[1],
            )
            val_x, val_y = _make_examples(
                rng, n_class_zero=15, n_class_one=15
            )
            test_x, test_y = _make_examples(
                rng, n_class_zero=30, n_class_one=30
            )
            partitions[client_id] = _ToyPartition(
                train_x=train_x,
                train_y=train_y,
                val_x=val_x,
                val_y=val_y,
                test_x=test_x,
                test_y=test_y,
            )
        return partitions

    @property
    def task_id(self) -> str:
        return "toy-linear-v1"

    @property
    def client_ids(self) -> tuple[str, ...]:
        return self._client_ids

    @property
    def feature_schema(self) -> FeatureSchema:
        return self._feature_schema

    @property
    def label_schema(self) -> LabelSchema:
        return self._label_schema

    @property
    def graph_schema(self) -> None:
        return None

    @property
    def model_spec(self) -> ModelSpec:
        return self._model_spec

    def initial_state(self) -> ArrayState:
        return {
            name: value.copy() for name, value in self._initial_state.items()
        }

    def _model_from_state(self, state: Mapping[str, np.ndarray]) -> nn.Module:
        self._model_spec.validate_state(state)
        model = _ToyLinearModel()
        template = model.state_dict()
        model.load_state_dict(arrays_to_torch_state(state, template=template))
        return model

    def _partition(self, client_id: str) -> _ToyPartition:
        try:
            return self._partitions[client_id]
        except KeyError as exc:
            raise KeyError(f"Unknown toy client '{client_id}'.") from exc

    def train_local(
        self,
        client_id: str,
        global_state: Mapping[str, np.ndarray],
        config: LocalTrainConfig,
    ) -> LocalTrainResult:
        partition = self._partition(client_id)
        seed = _stable_client_seed(config.seed, client_id)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        model = self._model_from_state(global_state)
        global_parameters = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        if config.optimizer == "sgd":
            optimizer: torch.optim.Optimizer = torch.optim.SGD(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        elif config.optimizer == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )

        final_loss = 0.0
        model.train()
        for _ in range(config.local_epochs):
            optimizer.zero_grad()
            logits = model(partition.train_x)
            loss = F.cross_entropy(logits, partition.train_y)
            if config.proximal_mu > 0:
                proximal = torch.zeros((), dtype=loss.dtype)
                for name, parameter in model.named_parameters():
                    proximal = proximal + torch.sum(
                        (parameter - global_parameters[name]) ** 2
                    )
                loss = loss + (config.proximal_mu / 2.0) * proximal
            loss.backward()
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.grad_clip
                )
            optimizer.step()
            final_loss = float(loss.detach())

        return LocalTrainResult(
            state=torch_state_to_arrays(model.state_dict()),
            num_examples=int(partition.train_y.numel()),
            metrics={"train_loss": final_loss},
        )

    def evaluate_local(
        self,
        client_id: str,
        state: Mapping[str, np.ndarray],
        *,
        split: str,
    ) -> EvaluationResult:
        partition = self._partition(client_id)
        if split not in {"val", "test"}:
            raise ValueError("Toy task evaluation split must be 'val' or 'test'.")
        features = partition.val_x if split == "val" else partition.test_x
        labels = partition.val_y if split == "val" else partition.test_y
        model = self._model_from_state(state)
        model.eval()
        with torch.no_grad():
            logits = model(features)
            loss = float(F.cross_entropy(logits, labels))
            predictions = logits.argmax(dim=-1)
        matrix = confusion_matrix_from_predictions(
            labels.numpy(),
            predictions.numpy(),
            num_classes=self._label_schema.num_classes,
        )
        return EvaluationResult(
            confusion_matrix=matrix,
            num_examples=int(labels.numel()),
            loss=loss,
        )

    def metadata(self) -> Mapping[str, Any]:
        return {
            "seed": self._seed,
            "description": "Two-client non-IID linear classification proof task.",
        }
