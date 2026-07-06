from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from .v2_config import V2ExperimentConfig


def optimizer_tuning_grid(
    config: V2ExperimentConfig, optimizer_name: str
) -> list[dict[str, float]]:
    """Return validation-only tuning candidates for one optimizer."""

    momentums = config.tuning.sgd_momentum_values if optimizer_name == "SGD" else (0.0,)
    return [
        {
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "momentum": float(momentum),
        }
        for learning_rate in config.tuning.learning_rates
        for weight_decay in config.tuning.weight_decays
        for momentum in momentums
    ]


def load_locked_hyperparameters(path: str | Path) -> dict[str, dict[str, Any]]:
    locked_path = Path(path)
    with locked_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Locked hyperparameters file must contain an optimizer mapping")
    return payload


def apply_locked_hyperparameters(
    config: V2ExperimentConfig,
    optimizer_name: str,
    locked: dict[str, dict[str, Any]],
) -> tuple[V2ExperimentConfig, float]:
    """Apply validation-selected hyperparameters for one final evaluation run."""

    if optimizer_name not in locked:
        raise KeyError(f"Missing locked hyperparameters for optimizer: {optimizer_name}")
    row = locked[optimizer_name]
    if bool(row.get("uses_test_split", False)):
        raise ValueError("Locked hyperparameters must not be selected with the test split")

    learning_rate = float(row["learning_rate"])
    weight_decay = float(row["weight_decay"])
    momentum = float(row.get("momentum", row.get("sgd_momentum", 0.0)))
    if optimizer_name != "SGD":
        momentum = 0.0

    tuned_config = replace(
        config,
        training=replace(
            config.training,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        ),
    )
    return tuned_config, momentum
