#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gnn_robustness.data import load_planetoid  # noqa: E402
from gnn_robustness.v2_config import load_v2_config, resolved_seed  # noqa: E402
from gnn_robustness.v2_experiments import train_v2_model  # noqa: E402
from gnn_robustness.v2_hardware import capture_environment_metadata  # noqa: E402


def iqr(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    quantiles = statistics.quantiles(values, n=4, method="inclusive")
    return float(quantiles[2] - quantiles[0])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark V2 training runtime on the local machine."
    )
    parser.add_argument("--config", default="configs/v2_fixed_cora.yaml")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-root", default="results/v2")
    parser.add_argument("--dataset", default="Cora")
    parser.add_argument("--optimizer", default="Adam")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--label", default="local")
    args = parser.parse_args()

    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")

    config = load_v2_config(args.config)
    object.__setattr__(config, "training", replace(config.training, epochs=args.epochs))
    dataset, data = load_planetoid(args.dataset, args.data_root)

    output_root = Path(args.output_root)
    benchmark_dir = output_root / "benchmarks"
    metadata_dir = output_root / "metadata"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = capture_environment_metadata(
        metadata_dir / "runtime_benchmark_environment.json"
    )

    warmup_seed = resolved_seed(
        dataset=args.dataset,
        optimizer=args.optimizer,
        base_seed=0,
        experiment_mode="runtime_benchmark_warmup",
        perturbation_type="clean",
        severity=0.0,
    )
    train_v2_model(
        data.clone(),
        num_features=dataset.num_node_features,
        num_classes=dataset.num_classes,
        optimizer_name=args.optimizer,
        config=config,
        seed=warmup_seed,
    )

    timings: list[float] = []
    for repeat in range(args.repeats):
        run_seed = resolved_seed(
            dataset=args.dataset,
            optimizer=args.optimizer,
            base_seed=repeat + 1,
            experiment_mode="runtime_benchmark",
            perturbation_type="clean",
            severity=0.0,
        )
        _, _, seconds = train_v2_model(
            data.clone(),
            num_features=dataset.num_node_features,
            num_classes=dataset.num_classes,
            optimizer_name=args.optimizer,
            config=config,
            seed=run_seed,
        )
        timings.append(float(seconds))

    summary = {
        "dataset": args.dataset,
        "optimizer": args.optimizer,
        "epochs": args.epochs,
        "repeats": args.repeats,
        "warmup_excluded": True,
        "local_machine_cpu_dependent": True,
        "mean_seconds": statistics.mean(timings),
        "median_seconds": statistics.median(timings),
        "std_seconds": statistics.stdev(timings) if len(timings) > 1 else 0.0,
        "iqr_seconds": iqr(timings),
        "metadata_path": str(metadata_path),
        "raw_timings_json": json.dumps(timings),
    }
    output_path = benchmark_dir / f"runtime_benchmark_{args.label}.csv"
    pd.DataFrame([summary]).to_csv(output_path, index=False)
    print(f"Wrote runtime benchmark: {output_path}")


if __name__ == "__main__":
    main()
