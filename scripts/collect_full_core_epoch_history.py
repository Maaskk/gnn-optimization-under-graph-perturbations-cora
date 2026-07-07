from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

import sys

sys.path.insert(0, str(SRC))

from gnn_robustness.data import load_planetoid  # noqa: E402
from gnn_robustness.metrics import accuracy_score, macro_f1_score  # noqa: E402
from gnn_robustness.model import GCN  # noqa: E402
from gnn_robustness.optimizers import make_optimizer  # noqa: E402
from gnn_robustness.train import set_seed  # noqa: E402
from gnn_robustness.v2_config import load_v2_config, resolved_seed  # noqa: E402
from gnn_robustness.v2_experiments import (  # noqa: E402
    V2_RAW_COLUMNS,
    _apply_training_time_perturbation,
)
from gnn_robustness.v2_hardware import (  # noqa: E402
    capture_environment_metadata,
    current_git_commit,
)

EPOCH_COLUMNS = [
    "experiment_id",
    "git_commit",
    "timestamp",
    "dataset",
    "protocol",
    "robustness_setting",
    "optimizer",
    "seed",
    "resolved_seed",
    "perturbation_type",
    "requested_severity",
    "actual_perturbation_rate",
    "actual_perturbation_count",
    "epoch",
    "train_loss",
    "validation_loss",
    "validation_accuracy",
    "validation_macro_f1",
    "gradient_l2_norm",
    "elapsed_seconds",
]


KEY_COLUMNS = [
    "dataset",
    "optimizer",
    "seed",
    "perturbation_type",
    "requested_severity",
]


def core_conditions(perturbations: tuple[str, ...], severities: tuple[float, ...]):
    for perturbation in perturbations:
        if perturbation == "clean":
            yield perturbation, 0.0
        else:
            for severity in severities:
                yield perturbation, float(severity)


def gradient_l2_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = float(parameter.grad.detach().norm(2).item())
        total += value * value
    return math.sqrt(total)


def split_metrics(model: GCN, data, mask: torch.Tensor) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[mask], data.y[mask])
    return {
        "loss": float(loss.item()),
        "accuracy": accuracy_score(logits, data.y, mask),
        "macro_f1": macro_f1_score(logits, data.y, mask),
    }


def existing_completed_keys(epoch_path: Path, run_path: Path, epochs: int) -> set[tuple[Any, ...]]:
    if not epoch_path.exists() or not run_path.exists():
        return set()
    epoch_data = pd.read_csv(epoch_path, usecols=KEY_COLUMNS + ["epoch"])
    run_data = pd.read_csv(run_path, usecols=KEY_COLUMNS)
    epoch_data["requested_severity"] = epoch_data["requested_severity"].astype(float).round(8)
    run_data["requested_severity"] = run_data["requested_severity"].astype(float).round(8)
    epoch_counts = (
        epoch_data.groupby(KEY_COLUMNS, dropna=False)["epoch"]
        .nunique()
        .reset_index(name="n_epochs")
    )
    complete_epochs = epoch_counts[epoch_counts["n_epochs"] == epochs][KEY_COLUMNS]
    merged = complete_epochs.merge(run_data.drop_duplicates(), on=KEY_COLUMNS, how="inner")
    return {tuple(row) for row in merged.itertuples(index=False, name=None)}


def append_rows(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def run_condition(
    *,
    dataset_name: str,
    dataset,
    clean_data,
    config,
    optimizer_name: str,
    base_seed: int,
    perturbation_type: str,
    severity: float,
    hardware_metadata_path: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    condition_seed = resolved_seed(
        dataset=dataset_name,
        optimizer=optimizer_name,
        base_seed=base_seed,
        experiment_mode=config.protocol,
        perturbation_type=perturbation_type,
        severity=severity,
    )
    run_data, perturbation_metadata = _apply_training_time_perturbation(
        clean_data,
        perturbation_type=perturbation_type,
        severity=severity,
        seed=condition_seed,
    )
    set_seed(condition_seed)
    run_data = run_data.to(config.training.device)
    model = GCN(
        input_channels=dataset.num_node_features,
        hidden_channels=config.training.hidden_channels,
        output_channels=dataset.num_classes,
        dropout=config.training.dropout,
    ).to(config.training.device)
    optimizer = make_optimizer(
        optimizer_name,
        model.parameters(),
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        momentum=0.0,
    )

    timestamp = datetime.now(UTC).isoformat()
    git_commit = current_git_commit()
    epoch_rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    final_train_loss = 0.0
    best_validation_epoch = 0
    best_validation_accuracy = -1.0

    for epoch in range(1, config.training.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(run_data.x, run_data.edge_index)
        loss = F.cross_entropy(logits[run_data.train_mask], run_data.y[run_data.train_mask])
        loss.backward()
        grad_norm = gradient_l2_norm(model.parameters())
        optimizer.step()
        final_train_loss = float(loss.item())
        validation = split_metrics(model, run_data, run_data.val_mask)
        if validation["accuracy"] > best_validation_accuracy:
            best_validation_accuracy = validation["accuracy"]
            best_validation_epoch = epoch
        epoch_rows.append(
            {
                "experiment_id": f"{config.experiment_id}_full_epoch_history",
                "git_commit": git_commit,
                "timestamp": timestamp,
                "dataset": dataset_name,
                "protocol": config.protocol,
                "robustness_setting": config.robustness_setting,
                "optimizer": optimizer_name,
                "seed": base_seed,
                "resolved_seed": condition_seed,
                "perturbation_type": perturbation_type,
                "requested_severity": severity,
                "actual_perturbation_rate": float(
                    perturbation_metadata["actual_perturbation_rate"]
                ),
                "actual_perturbation_count": int(
                    perturbation_metadata["actual_perturbation_count"]
                ),
                "epoch": epoch,
                "train_loss": final_train_loss,
                "validation_loss": validation["loss"],
                "validation_accuracy": validation["accuracy"],
                "validation_macro_f1": validation["macro_f1"],
                "gradient_l2_norm": grad_norm,
                "elapsed_seconds": time.perf_counter() - start,
            }
        )

    training_time = time.perf_counter() - start
    train_metrics = split_metrics(model, run_data, run_data.train_mask)
    validation_metrics = split_metrics(model, run_data, run_data.val_mask)
    test_metrics = split_metrics(model, run_data, run_data.test_mask)
    hyperparameters = {
        "hidden_channels": config.training.hidden_channels,
        "dropout": config.training.dropout,
        "learning_rate": config.training.learning_rate,
        "weight_decay": config.training.weight_decay,
        "momentum": 0.0,
    }
    run_row = {
        "experiment_id": f"{config.experiment_id}_full_epoch_history",
        "git_commit": git_commit,
        "timestamp": timestamp,
        "dataset": dataset_name,
        "protocol": config.protocol,
        "robustness_setting": config.robustness_setting,
        "optimizer": optimizer_name,
        "seed": base_seed,
        "resolved_seed": condition_seed,
        "hyperparameter_config": json.dumps(hyperparameters, sort_keys=True),
        "perturbation_type": perturbation_type,
        "requested_severity": severity,
        "actual_perturbation_rate": float(perturbation_metadata["actual_perturbation_rate"]),
        "actual_perturbation_count": int(perturbation_metadata["actual_perturbation_count"]),
        "train_accuracy": train_metrics["accuracy"],
        "validation_accuracy": validation_metrics["accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "macro_f1": test_metrics["macro_f1"],
        "loss": final_train_loss,
        "training_time_seconds": training_time,
        "best_validation_epoch": best_validation_epoch,
        "final_epoch": config.training.epochs,
        "hardware_metadata_path": hardware_metadata_path,
    }
    return epoch_rows, run_row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect full per-epoch proof history for the 650-run core Cora protocol."
    )
    parser.add_argument("--config", default="configs/v2_fixed_cora.yaml")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="results/v2/proof")
    parser.add_argument(
        "--optimizers",
        default=None,
        help="Optional comma-separated optimizer subset, for example Adam,AdamW.",
    )
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(max(1, int(args.torch_threads)))

    config = load_v2_config(args.config)
    selected_optimizers = config.optimizers
    if args.optimizers:
        requested_optimizers = tuple(
            item.strip() for item in args.optimizers.split(",") if item.strip()
        )
        unknown = sorted(set(requested_optimizers) - set(config.optimizers))
        if unknown:
            raise ValueError(f"Unknown optimizers for config {args.config}: {unknown}")
        selected_optimizers = requested_optimizers
    output_dir = Path(args.output_dir)
    epoch_path = output_dir / "full_core_epoch_history.csv"
    run_path = output_dir / "full_core_run_results.csv"
    metadata_path = output_dir / "environment.json"

    if args.force:
        epoch_path.unlink(missing_ok=True)
        run_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)

    if metadata_path.exists():
        metadata_file = metadata_path
    else:
        metadata_file = capture_environment_metadata(metadata_path)
    completed = existing_completed_keys(epoch_path, run_path, config.training.epochs)

    total_expected = (
        len(config.datasets)
        * len(selected_optimizers)
        * len(config.seeds)
        * (1 + (len(config.perturbations) - 1) * len(config.severities))
    )
    completed_now = 0
    started_at = time.perf_counter()

    for dataset_name in config.datasets:
        dataset, clean_data = load_planetoid(dataset_name, args.data_root)
        for optimizer_name in selected_optimizers:
            for base_seed in config.seeds:
                for perturbation_type, severity in core_conditions(
                    config.perturbations, config.severities
                ):
                    key = (
                        dataset_name,
                        optimizer_name,
                        base_seed,
                        perturbation_type,
                        round(float(severity), 8),
                    )
                    if key in completed:
                        continue
                    if args.max_runs is not None and completed_now >= args.max_runs:
                        break
                    run_index = len(completed) + completed_now + 1
                    print(
                        f"[{run_index}/{total_expected}] dataset={dataset_name} "
                        f"optimizer={optimizer_name} seed={base_seed} "
                        f"perturbation={perturbation_type} severity={severity}",
                        flush=True,
                    )
                    epoch_rows, run_row = run_condition(
                        dataset_name=dataset_name,
                        dataset=dataset,
                        clean_data=clean_data,
                        config=config,
                        optimizer_name=optimizer_name,
                        base_seed=base_seed,
                        perturbation_type=perturbation_type,
                        severity=severity,
                        hardware_metadata_path=str(metadata_file),
                    )
                    append_rows(epoch_path, epoch_rows, EPOCH_COLUMNS)
                    append_rows(run_path, [run_row], V2_RAW_COLUMNS)
                    completed_now += 1
                if args.max_runs is not None and completed_now >= args.max_runs:
                    break
            if args.max_runs is not None and completed_now >= args.max_runs:
                break
        if args.max_runs is not None and completed_now >= args.max_runs:
            break

    if epoch_path.exists() and run_path.exists():
        epoch_data = pd.read_csv(epoch_path)
        run_data = pd.read_csv(run_path)
        proof_metadata = {
            "generated_at": datetime.now(UTC).isoformat(),
            "git_commit": current_git_commit(),
            "config": args.config,
            "epoch_history_path": str(epoch_path),
            "run_results_path": str(run_path),
            "expected_runs": total_expected,
            "observed_runs": int(len(run_data)),
            "expected_epoch_rows": total_expected * config.training.epochs,
            "observed_epoch_rows": int(len(epoch_data)),
            "epochs_per_run": config.training.epochs,
            "wall_time_seconds_this_invocation": time.perf_counter() - started_at,
        }
        (output_dir / "full_core_epoch_history_manifest.json").write_text(
            json.dumps(proof_metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(proof_metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
