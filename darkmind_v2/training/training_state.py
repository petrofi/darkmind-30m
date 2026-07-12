"""Serializable progress state for deterministic DarkMind v2 training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class TrainingState:
    step: int = 0
    tokens_seen: int = 0
    data_position: int = 0
    best_validation_loss: float | None = None
    last_validation_loss: float | None = None
    last_training_loss: float | None = None
    smoothed_training_loss: float | None = None
    best_checkpoint: str | None = None
    interrupted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainingState":
        return cls(**payload)
