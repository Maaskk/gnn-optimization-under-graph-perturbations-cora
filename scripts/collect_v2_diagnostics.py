#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import psutil
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gnn_robustness.data import load_planetoid  # noqa: E402
from gnn_robustness.metrics import accuracy_score, macro_f1_score  # noqa: E402
from gnn_robustness.model import GCN  # noqa: E402
from gnn_robustness.optimizers import make_optimizer  # noqa: E402
from gnn_robustness.train import set_seed  # noqa: E402
from gnn_robustness.v2_config import load_v2_config, resolved_seed  # noqa: E402
from gnn_robustness.v2_hardware import (  # noqa: E402
    capture_environment_metadata,
    current_git_commit,
)


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


def current_rss_mb(process: psutil.Process) -> float:
    return float(process.memory_info().rss / (1024**2))


def maybe_reset_gpu_peak(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def gpu_peak_mb(device: str) -> float:
    if device.startswith("cuda") and torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated(device) / (1024**2))
    return 0.0


def train_clean_diagnostic(
    *,
    dataset_name: str,
    data_root: str | Path,
    optimizer_name: str,
    base_seed: int,
    config,
    hardware_metadata_path: str,
) -> tuple[dict, list[dict]]:
    dataset, data = load_planetoid(dataset_name, data_root)
    seed = resolved_seed(
        dataset=dataset_name,
        optimizer=optimizer_name,
        base_seed=base_seed,
        experiment_mode=f"{config.protocol}_diagnostics_clean",
        perturbation_type="clean",
        severity=0.0,
    )
    set_seed(seed)
    device = config.training.device
    run_data = data.clone().to(device)
    model = GCN(
        input_channels=dataset.num_node_features,
        hidden_channels=config.training.hidden_channels,
        output_channels=dataset.num_classes,
        dropout=config.training.dropout,
    ).to(device)
    momentum = config.tuning.sgd_momentum if config.protocol == "tuned" else 0.0
    optimizer = make_optimizer(
        optimizer_name,
        model.parameters(),
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        momentum=momentum if optimizer_name == "SGD" else 0.0,
    )
    process = psutil.Process()
    rss_start_mb = current_rss_mb(process)
    rss_peak_mb = rss_start_mb
    maybe_reset_gpu_peak(device)
    best_validation_accuracy = -1.0
    best_validation_epoch = 0
    history_rows: list[dict] = []
    final_loss = 0.0
    final_gradient = 0.0
    start = time.perf_counter()
    timestamp = datetime.now(UTC).isoformat()
    git_commit = current_git_commit()
    hyperparameters = {
        "hidden_channels": config.training.hidden_channels,
        "dropout": config.training.dropout,
        "learning_rate": config.training.learning_rate,
        "weight_decay": config.training.weight_decay,
        "momentum": momentum if optimizer_name == "SGD" else 0.0,
    }
    for epoch in range(1, config.training.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(run_data.x, run_data.edge_index)
        loss = F.cross_entropy(logits[run_data.train_mask], run_data.y[run_data.train_mask])
        loss.backward()
        final_gradient = gradient_l2_norm(model.parameters())
        optimizer.step()
        final_loss = float(loss.item())
        validation_metrics = split_metrics(model, run_data, run_data.val_mask)
        if validation_metrics["accuracy"] > best_validation_accuracy:
            best_validation_accuracy = validation_metrics["accuracy"]
            best_validation_epoch = epoch
        rss_mb = current_rss_mb(process)
        rss_peak_mb = max(rss_peak_mb, rss_mb)
        history_rows.append(
            {
                "experiment_id": f"{config.experiment_id}_clean_diagnostics",
                "git_commit": git_commit,
                "timestamp": timestamp,
                "dataset": dataset_name,
                "protocol": config.protocol,
                "robustness_setting": "clean_training_diagnostics",
                "optimizer": optimizer_name,
                "seed": base_seed,
                "resolved_seed": seed,
                "epoch": epoch,
                "train_loss": final_loss,
                "validation_loss": validation_metrics["loss"],
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_macro_f1": validation_metrics["macro_f1"],
                "gradient_l2_norm": final_gradient,
                "rss_mb": rss_mb,
                "rss_delta_mb": max(0.0, rss_mb - rss_start_mb),
            }
        )
    training_time = time.perf_counter() - start
    train_metrics = split_metrics(model, run_data, run_data.train_mask)
    val_metrics = split_metrics(model, run_data, run_data.val_mask)
    test_metrics = split_metrics(model, run_data, run_data.test_mask)
    run_row = {
        "experiment_id": f"{config.experiment_id}_clean_diagnostics",
        "git_commit": git_commit,
        "timestamp": timestamp,
        "dataset": dataset_name,
        "protocol": config.protocol,
        "robustness_setting": "clean_training_diagnostics",
        "optimizer": optimizer_name,
        "seed": base_seed,
        "resolved_seed": seed,
        "hyperparameter_config": json.dumps(hyperparameters, sort_keys=True),
        "train_accuracy": train_metrics["accuracy"],
        "validation_accuracy": val_metrics["accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "macro_f1": test_metrics["macro_f1"],
        "loss": final_loss,
        "training_time_seconds": training_time,
        "best_validation_epoch": best_validation_epoch,
        "final_epoch": config.training.epochs,
        "final_gradient_l2_norm": final_gradient,
        "mean_gradient_l2_norm": float(pd.DataFrame(history_rows)["gradient_l2_norm"].mean()),
        "peak_rss_mb": rss_peak_mb,
        "cpu_memory_delta_mb": max(0.0, rss_peak_mb - rss_start_mb),
        "gpu_peak_memory_mb": gpu_peak_mb(device),
        "memory_device": "cuda"
        if device.startswith("cuda") and torch.cuda.is_available()
        else "cpu",
        "hardware_metadata_path": hardware_metadata_path,
    }
    return run_row, history_rows


def summarize_outputs(
    runs: pd.DataFrame, history: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_summary = (
        runs.groupby("optimizer", dropna=False)
        .agg(
            n_seeds=("seed", "nunique"),
            mean_clean_test_accuracy=("test_accuracy", "mean"),
            std_clean_test_accuracy=("test_accuracy", "std"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_training_time_seconds=("training_time_seconds", "mean"),
            mean_best_validation_epoch=("best_validation_epoch", "mean"),
            mean_final_gradient_l2_norm=("final_gradient_l2_norm", "mean"),
            mean_gradient_l2_norm=("mean_gradient_l2_norm", "mean"),
            mean_peak_rss_mb=("peak_rss_mb", "mean"),
            mean_cpu_memory_delta_mb=("cpu_memory_delta_mb", "mean"),
            max_peak_rss_mb=("peak_rss_mb", "max"),
            max_gpu_peak_memory_mb=("gpu_peak_memory_mb", "max"),
        )
        .reset_index()
    )
    gradient_summary = (
        history.groupby(["optimizer", "epoch"], dropna=False)
        .agg(
            n_seeds=("seed", "nunique"),
            mean_gradient_l2_norm=("gradient_l2_norm", "mean"),
            std_gradient_l2_norm=("gradient_l2_norm", "std"),
            mean_train_loss=("train_loss", "mean"),
            mean_validation_loss=("validation_loss", "mean"),
            mean_validation_accuracy=("validation_accuracy", "mean"),
        )
        .reset_index()
    )
    numeric_columns = run_summary.select_dtypes(include=["number"]).columns
    run_summary[numeric_columns] = run_summary[numeric_columns].round(6)
    numeric_columns = gradient_summary.select_dtypes(include=["number"]).columns
    gradient_summary[numeric_columns] = gradient_summary[numeric_columns].round(6)
    return run_summary, gradient_summary


def parse_csv_values(raw: str, caster=str) -> tuple:
    return tuple(caster(value.strip()) for value in raw.split(",") if value.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect V2 clean optimizer diagnostics.")
    parser.add_argument("--config", default="configs/v2_fixed_cora.yaml")
    parser.add_argument("--output-root", default="results/v2/diagnostics")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--seeds", default="", help="Optional comma-separated seed override")
    parser.add_argument(
        "--optimizers", default="", help="Optional comma-separated optimizer override"
    )
    parser.add_argument("--epochs", type=int, default=0, help="Optional epoch override")
    parser.add_argument("--max-runs", type=int, default=0, help="0 means no limit")
    parser.add_argument("--label", default="clean", help="Output filename suffix")
    args = parser.parse_args()

    config = load_v2_config(args.config)
    if args.seeds:
        object.__setattr__(config, "seeds", parse_csv_values(args.seeds, int))
    if args.optimizers:
        object.__setattr__(config, "optimizers", parse_csv_values(args.optimizers, str))
    if args.epochs:
        object.__setattr__(config.training, "epochs", args.epochs)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    metadata_dir = output_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    hardware_path = capture_environment_metadata(metadata_dir / "environment.json")
    all_runs: list[dict] = []
    all_history: list[dict] = []
    planned = [
        (dataset, seed, optimizer)
        for dataset in config.datasets
        for seed in config.seeds
        for optimizer in config.optimizers
    ]
    for index, (dataset, seed, optimizer) in enumerate(planned):
        if args.max_runs and index >= args.max_runs:
            continue
        print(f"[v2-diagnostics] dataset={dataset} seed={seed} optimizer={optimizer}", flush=True)
        run_row, history_rows = train_clean_diagnostic(
            dataset_name=dataset,
            data_root=args.data_root,
            optimizer_name=optimizer,
            base_seed=seed,
            config=config,
            hardware_metadata_path=str(hardware_path),
        )
        all_runs.append(run_row)
        all_history.extend(history_rows)

    suffix = f"_{args.label}" if args.label else ""
    runs = pd.DataFrame(all_runs)
    history = pd.DataFrame(all_history)
    summary, gradient_summary = summarize_outputs(runs, history)
    runs_path = output_root / f"v2_optimizer_diagnostics_runs{suffix}.csv"
    history_path = output_root / f"v2_gradient_history{suffix}.csv"
    gradient_summary_path = output_root / f"v2_gradient_summary{suffix}.csv"
    summary_path = output_root / f"v2_optimizer_diagnostics_summary{suffix}.csv"
    runs.to_csv(runs_path, index=False)
    history.to_csv(history_path, index=False)
    gradient_summary.to_csv(gradient_summary_path, index=False)
    summary.to_csv(summary_path, index=False)
    manifest = {
        "config": args.config,
        "completed_runs": len(all_runs),
        "planned_runs": len(planned),
        "epochs": config.training.epochs,
        "outputs": {
            "runs": str(runs_path),
            "history": str(history_path),
            "gradient_summary": str(gradient_summary_path),
            "optimizer_summary": str(summary_path),
        },
    }
    manifest_path = output_root / f"v2_optimizer_diagnostics_manifest{suffix}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote diagnostics runs: {runs_path}")
    print(f"Wrote gradient history: {history_path}")
    print(f"Wrote gradient summary: {gradient_summary_path}")
    print(f"Wrote diagnostics summary: {summary_path}")
    print(f"Wrote diagnostics manifest: {manifest_path}")


if __name__ == "__main__":
    main()
