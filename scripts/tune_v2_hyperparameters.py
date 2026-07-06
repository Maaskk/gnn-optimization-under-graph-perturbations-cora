#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gnn_robustness.data import load_planetoid  # noqa: E402
from gnn_robustness.v2_config import load_v2_config, resolved_seed  # noqa: E402
from gnn_robustness.v2_experiments import _split_metrics, train_v2_model  # noqa: E402
from gnn_robustness.v2_tuning import optimizer_tuning_grid  # noqa: E402


def parse_csv_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only V2 hyperparameter tuning.")
    parser.add_argument("--config", default="configs/v2_tuned_cora.yaml")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-root", default="results/v2")
    parser.add_argument("--epochs", type=int, default=0, help="Optional smoke override")
    parser.add_argument(
        "--optimizers", default="", help="Optional comma-separated optimizer override"
    )
    parser.add_argument(
        "--max-configs", type=int, default=0, help="Grid limit per optimizer; 0 means full grid"
    )
    parser.add_argument("--label", default="local")
    args = parser.parse_args()

    config = load_v2_config(args.config)
    if not config.tuning.enabled:
        raise SystemExit("The selected config does not enable tuning.")
    if args.epochs:
        object.__setattr__(config, "training", replace(config.training, epochs=args.epochs))
    optimizers = parse_csv_list(args.optimizers) if args.optimizers else config.optimizers

    dataset_name = config.datasets[0]
    dataset, data = load_planetoid(dataset_name, args.data_root)

    trial_rows: list[dict] = []
    locked: dict[str, dict] = {}
    for optimizer_name in optimizers:
        best_row = None
        optimizer_grid = optimizer_tuning_grid(config, optimizer_name)
        if args.max_configs:
            optimizer_grid = optimizer_grid[: args.max_configs]
        for trial in optimizer_grid:
            learning_rate = trial["learning_rate"]
            weight_decay = trial["weight_decay"]
            momentum = trial["momentum"]
            trial_config = replace(
                config,
                training=replace(
                    config.training,
                    learning_rate=float(learning_rate),
                    weight_decay=float(weight_decay),
                ),
            )
            validation_scores = []
            for base_seed in config.tuning.tuning_seeds:
                seed = resolved_seed(
                    dataset=dataset_name,
                    optimizer=optimizer_name,
                    base_seed=base_seed,
                    experiment_mode="validation_only_tuning",
                    perturbation_type="clean",
                    severity=0.0,
                )
                model, train_info, seconds = train_v2_model(
                    data.clone(),
                    num_features=dataset.num_node_features,
                    num_classes=dataset.num_classes,
                    optimizer_name=optimizer_name,
                    config=trial_config,
                    seed=seed,
                    momentum=momentum,
                )
                validation = _split_metrics(model, data, data.val_mask)
                validation_scores.append(float(validation["accuracy"]))
                trial_rows.append(
                    {
                        "optimizer": optimizer_name,
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                        "momentum": momentum if optimizer_name == "SGD" else 0.0,
                        "seed": base_seed,
                        "resolved_seed": seed,
                        "validation_accuracy": validation["accuracy"],
                        "validation_loss": validation["loss"],
                        "best_validation_epoch": train_info["best_validation_epoch"],
                        "training_time_seconds": seconds,
                        "uses_test_split": False,
                    }
                )
            mean_validation_accuracy = sum(validation_scores) / len(validation_scores)
            row = {
                "optimizer": optimizer_name,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "momentum": momentum if optimizer_name == "SGD" else 0.0,
                "mean_validation_accuracy": mean_validation_accuracy,
            }
            if best_row is None or mean_validation_accuracy > best_row["mean_validation_accuracy"]:
                best_row = row
        if best_row is None:
            continue
        locked[optimizer_name] = {
            "learning_rate": best_row["learning_rate"],
            "weight_decay": best_row["weight_decay"],
            "momentum": best_row["momentum"],
            "mean_validation_accuracy": best_row["mean_validation_accuracy"],
            "selection_metric": "mean_validation_accuracy",
            "tuning_seeds": list(config.tuning.tuning_seeds),
            "uses_test_split": False,
            "sgd_momentum": best_row["momentum"] if optimizer_name == "SGD" else 0.0,
        }

    output_root = Path(args.output_root)
    tuned_dir = output_root / "tuned"
    tuned_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trial_rows).to_csv(tuned_dir / f"tuning_trials_{args.label}.csv", index=False)
    locked_path = tuned_dir / f"locked_hyperparameters_{args.label}.json"
    locked_path.write_text(json.dumps(locked, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote validation-only tuning trials: {tuned_dir / f'tuning_trials_{args.label}.csv'}")
    print(f"Wrote locked hyperparameters: {locked_path}")


if __name__ == "__main__":
    main()
