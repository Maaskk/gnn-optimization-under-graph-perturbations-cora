from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    hidden_channels: int
    dropout: float
    learning_rate: float
    weight_decay: float
    device: str = "cpu"


@dataclass(frozen=True)
class TuningConfig:
    enabled: bool
    learning_rates: tuple[float, ...] = ()
    weight_decays: tuple[float, ...] = ()
    tuning_seeds: tuple[int, ...] = ()
    sgd_momentum: float = 0.9


@dataclass(frozen=True)
class V2ExperimentConfig:
    experiment_id: str
    version: str
    datasets: tuple[str, ...]
    protocol: str
    robustness_setting: str
    seeds: tuple[int, ...]
    optimizers: tuple[str, ...]
    perturbations: tuple[str, ...]
    severities: tuple[float, ...]
    training: TrainingConfig
    tuning: TuningConfig
    notes: str = ""


def _tuple(value: Any, cast=str) -> tuple:
    if value is None:
        return ()
    return tuple(cast(item) for item in value)


def load_v2_config(path: str | Path) -> V2ExperimentConfig:
    """Load a V2 YAML experiment config into typed immutable objects."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Config {config_path} must contain a YAML mapping")

    training_raw = raw.get("training", {})
    tuning_raw = raw.get("tuning", {})
    training = TrainingConfig(
        epochs=int(training_raw["epochs"]),
        hidden_channels=int(training_raw["hidden_channels"]),
        dropout=float(training_raw["dropout"]),
        learning_rate=float(training_raw["learning_rate"]),
        weight_decay=float(training_raw["weight_decay"]),
        device=str(training_raw.get("device", "cpu")),
    )
    tuning = TuningConfig(
        enabled=bool(tuning_raw.get("enabled", False)),
        learning_rates=_tuple(tuning_raw.get("learning_rates", ()), float),
        weight_decays=_tuple(tuning_raw.get("weight_decays", ()), float),
        tuning_seeds=_tuple(tuning_raw.get("tuning_seeds", ()), int),
        sgd_momentum=float(tuning_raw.get("sgd_momentum", 0.9)),
    )
    return V2ExperimentConfig(
        experiment_id=str(raw["experiment_id"]),
        version=str(raw.get("version", "v2")),
        datasets=_tuple(raw["datasets"], str),
        protocol=str(raw["protocol"]),
        robustness_setting=str(raw["robustness_setting"]),
        seeds=_tuple(raw["seeds"], int),
        optimizers=_tuple(raw["optimizers"], str),
        perturbations=_tuple(raw["perturbations"], str),
        severities=_tuple(raw.get("severities", ()), float),
        training=training,
        tuning=tuning,
        notes=str(raw.get("notes", "")),
    )


def resolved_seed(
    *,
    dataset: str,
    optimizer: str,
    base_seed: int,
    experiment_mode: str,
    perturbation_type: str,
    severity: float,
) -> int:
    """Derive a deterministic per-run seed from all experimental factors."""

    payload = "|".join(
        [
            dataset,
            optimizer,
            str(base_seed),
            experiment_mode,
            perturbation_type,
            f"{severity:.8f}",
        ]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % (2**31)
