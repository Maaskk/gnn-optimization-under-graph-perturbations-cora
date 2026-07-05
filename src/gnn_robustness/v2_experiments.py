from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .data import load_planetoid
from .metrics import accuracy_score, macro_f1_score
from .model import GCN
from .optimizers import make_optimizer
from .train import set_seed
from .v2_config import V2ExperimentConfig, resolved_seed
from .v2_hardware import capture_environment_metadata, current_git_commit
from .v2_perturbations import (
    add_undirected_fake_edges,
    mask_active_features,
    remove_undirected_edges,
)

V2_RAW_COLUMNS = [
    "experiment_id",
    "git_commit",
    "timestamp",
    "dataset",
    "protocol",
    "robustness_setting",
    "optimizer",
    "seed",
    "resolved_seed",
    "hyperparameter_config",
    "perturbation_type",
    "requested_severity",
    "actual_perturbation_rate",
    "actual_perturbation_count",
    "train_accuracy",
    "validation_accuracy",
    "test_accuracy",
    "macro_f1",
    "loss",
    "training_time_seconds",
    "best_validation_epoch",
    "final_epoch",
    "hardware_metadata_path",
]


def build_v2_raw_row(**values: Any) -> dict[str, Any]:
    missing = [column for column in V2_RAW_COLUMNS if column not in values]
    if missing:
        raise ValueError(f"Missing V2 raw result columns: {missing}")
    return {column: values[column] for column in V2_RAW_COLUMNS}


def _split_metrics(model: GCN, data, mask: torch.Tensor) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[mask], data.y[mask])
    return {
        "loss": float(loss.item()),
        "accuracy": accuracy_score(logits, data.y, mask),
        "macro_f1": macro_f1_score(logits, data.y, mask),
    }


def _apply_training_time_perturbation(data, perturbation_type: str, severity: float, seed: int):
    run_data = data.clone()
    metadata: dict[str, float | int] = {}
    if perturbation_type == "clean":
        return run_data, {"actual_perturbation_rate": 0.0, "actual_perturbation_count": 0}
    if perturbation_type == "feature_masking":
        result = mask_active_features(run_data.x, severity=severity, seed=seed)
        run_data.x = result.features
        metadata = {
            "actual_perturbation_rate": result.metadata["actual_masking_rate"],
            "actual_perturbation_count": result.metadata["masked_feature_entries"],
        }
    elif perturbation_type == "edge_removal":
        result = remove_undirected_edges(run_data.edge_index, severity=severity, seed=seed)
        run_data.edge_index = result.edge_index
        metadata = {
            "actual_perturbation_rate": result.metadata["actual_removed_rate"],
            "actual_perturbation_count": result.metadata["actual_removed_edges"],
        }
    elif perturbation_type == "fake_edge_addition":
        result = add_undirected_fake_edges(
            run_data.edge_index,
            num_nodes=int(run_data.num_nodes),
            severity=severity,
            seed=seed,
        )
        run_data.edge_index = result.edge_index
        metadata = {
            "actual_perturbation_rate": result.metadata["actual_inserted_rate"],
            "actual_perturbation_count": result.metadata["actual_inserted_edges"],
        }
    else:
        raise ValueError(f"Unsupported V2 perturbation type: {perturbation_type}")
    return run_data, metadata


def train_v2_model(
    data,
    *,
    num_features: int,
    num_classes: int,
    optimizer_name: str,
    config: V2ExperimentConfig,
    seed: int,
    momentum: float = 0.0,
) -> tuple[GCN, dict[str, float | int], float]:
    """Train a clean or pre-perturbed V2 GCN and track validation-only best epoch."""

    set_seed(seed)
    model = GCN(
        input_channels=num_features,
        hidden_channels=config.training.hidden_channels,
        output_channels=num_classes,
        dropout=config.training.dropout,
    ).to(config.training.device)
    run_data = data.to(config.training.device)
    optimizer = make_optimizer(
        optimizer_name,
        model.parameters(),
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        momentum=momentum if optimizer_name == "SGD" else 0.0,
    )
    best_val_accuracy = -1.0
    best_validation_epoch = 0
    final_loss = 0.0
    start = time.perf_counter()
    for epoch in range(1, config.training.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(run_data.x, run_data.edge_index)
        loss = F.cross_entropy(logits[run_data.train_mask], run_data.y[run_data.train_mask])
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
        val_accuracy = _split_metrics(model, run_data, run_data.val_mask)["accuracy"]
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_validation_epoch = epoch
    return (
        model,
        {"best_validation_epoch": best_validation_epoch, "loss": final_loss},
        time.perf_counter() - start,
    )


def run_v2_training_time_condition(
    *,
    dataset_name: str,
    data_root: str | Path,
    optimizer_name: str,
    base_seed: int,
    perturbation_type: str,
    severity: float,
    config: V2ExperimentConfig,
    hardware_metadata_path: str,
) -> dict[str, Any]:
    dataset, data = load_planetoid(dataset_name, data_root)
    seed = resolved_seed(
        dataset=dataset_name,
        optimizer=optimizer_name,
        base_seed=base_seed,
        experiment_mode=config.protocol,
        perturbation_type=perturbation_type,
        severity=severity,
    )
    run_data, perturbation_metadata = _apply_training_time_perturbation(
        data,
        perturbation_type=perturbation_type,
        severity=severity,
        seed=seed,
    )
    momentum = config.tuning.sgd_momentum if config.protocol == "tuned" else 0.0
    model, train_info, training_time = train_v2_model(
        run_data,
        num_features=dataset.num_node_features,
        num_classes=dataset.num_classes,
        optimizer_name=optimizer_name,
        config=config,
        seed=seed,
        momentum=momentum,
    )
    train_metrics = _split_metrics(model, run_data, run_data.train_mask)
    val_metrics = _split_metrics(model, run_data, run_data.val_mask)
    test_metrics = _split_metrics(model, run_data, run_data.test_mask)
    hyperparameters = {
        "hidden_channels": config.training.hidden_channels,
        "dropout": config.training.dropout,
        "learning_rate": config.training.learning_rate,
        "weight_decay": config.training.weight_decay,
        "momentum": momentum if optimizer_name == "SGD" else 0.0,
    }
    return build_v2_raw_row(
        experiment_id=config.experiment_id,
        git_commit=current_git_commit(),
        timestamp=datetime.now(UTC).isoformat(),
        dataset=dataset_name,
        protocol=config.protocol,
        robustness_setting=config.robustness_setting,
        optimizer=optimizer_name,
        seed=base_seed,
        resolved_seed=seed,
        hyperparameter_config=json.dumps(hyperparameters, sort_keys=True),
        perturbation_type=perturbation_type,
        requested_severity=severity,
        actual_perturbation_rate=float(perturbation_metadata["actual_perturbation_rate"]),
        actual_perturbation_count=int(perturbation_metadata["actual_perturbation_count"]),
        train_accuracy=train_metrics["accuracy"],
        validation_accuracy=val_metrics["accuracy"],
        test_accuracy=test_metrics["accuracy"],
        macro_f1=test_metrics["macro_f1"],
        loss=train_info["loss"],
        training_time_seconds=training_time,
        best_validation_epoch=train_info["best_validation_epoch"],
        final_epoch=config.training.epochs,
        hardware_metadata_path=hardware_metadata_path,
    )


def run_v2_inference_time_condition(
    *,
    dataset_name: str,
    data_root: str | Path,
    optimizer_name: str,
    base_seed: int,
    perturbation_type: str,
    severity: float,
    config: V2ExperimentConfig,
    hardware_metadata_path: str,
) -> dict[str, Any]:
    """Train on clean data, then evaluate fixed weights under a perturbed input."""

    dataset, clean_data = load_planetoid(dataset_name, data_root)
    seed = resolved_seed(
        dataset=dataset_name,
        optimizer=optimizer_name,
        base_seed=base_seed,
        experiment_mode=f"{config.protocol}_inference_train_clean",
        perturbation_type="clean",
        severity=0.0,
    )
    momentum = config.tuning.sgd_momentum if config.protocol == "tuned" else 0.0
    model, train_info, training_time = train_v2_model(
        clean_data.clone(),
        num_features=dataset.num_node_features,
        num_classes=dataset.num_classes,
        optimizer_name=optimizer_name,
        config=config,
        seed=seed,
        momentum=momentum,
    )
    eval_seed = resolved_seed(
        dataset=dataset_name,
        optimizer=optimizer_name,
        base_seed=base_seed,
        experiment_mode=config.robustness_setting,
        perturbation_type=perturbation_type,
        severity=severity,
    )
    eval_data, perturbation_metadata = _apply_training_time_perturbation(
        clean_data,
        perturbation_type=perturbation_type,
        severity=severity,
        seed=eval_seed,
    )
    eval_data = eval_data.to(config.training.device)
    train_metrics = _split_metrics(model, eval_data, eval_data.train_mask)
    val_metrics = _split_metrics(model, eval_data, eval_data.val_mask)
    test_metrics = _split_metrics(model, eval_data, eval_data.test_mask)
    hyperparameters = {
        "hidden_channels": config.training.hidden_channels,
        "dropout": config.training.dropout,
        "learning_rate": config.training.learning_rate,
        "weight_decay": config.training.weight_decay,
        "momentum": momentum if optimizer_name == "SGD" else 0.0,
        "trained_on": "clean",
    }
    return build_v2_raw_row(
        experiment_id=config.experiment_id,
        git_commit=current_git_commit(),
        timestamp=datetime.now(UTC).isoformat(),
        dataset=dataset_name,
        protocol=config.protocol,
        robustness_setting="inference_time",
        optimizer=optimizer_name,
        seed=base_seed,
        resolved_seed=eval_seed,
        hyperparameter_config=json.dumps(hyperparameters, sort_keys=True),
        perturbation_type=perturbation_type,
        requested_severity=severity,
        actual_perturbation_rate=float(perturbation_metadata["actual_perturbation_rate"]),
        actual_perturbation_count=int(perturbation_metadata["actual_perturbation_count"]),
        train_accuracy=train_metrics["accuracy"],
        validation_accuracy=val_metrics["accuracy"],
        test_accuracy=test_metrics["accuracy"],
        macro_f1=test_metrics["macro_f1"],
        loss=train_info["loss"],
        training_time_seconds=training_time,
        best_validation_epoch=train_info["best_validation_epoch"],
        final_epoch=config.training.epochs,
        hardware_metadata_path=hardware_metadata_path,
    )


def prepare_v2_output_dirs(output_root: str | Path) -> dict[str, Path]:
    root = Path(output_root)
    paths = {
        "root": root,
        "raw": root / "raw",
        "aggregated": root / "aggregated",
        "metadata": root / "metadata",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_hardware_metadata(output_root: str | Path) -> Path:
    paths = prepare_v2_output_dirs(output_root)
    return capture_environment_metadata(paths["metadata"] / "environment.json")
