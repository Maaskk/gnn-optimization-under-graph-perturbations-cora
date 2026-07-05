#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gnn_robustness.v2_config import load_v2_config  # noqa: E402
from gnn_robustness.v2_experiments import (  # noqa: E402
    V2_RAW_COLUMNS,
    prepare_v2_output_dirs,
    run_v2_inference_time_condition,
    run_v2_training_time_condition,
    write_hardware_metadata,
)


def planned_conditions(config):
    for dataset in config.datasets:
        for seed in config.seeds:
            for optimizer in config.optimizers:
                for perturbation in config.perturbations:
                    severities = (0.0,) if perturbation == "clean" else config.severities
                    for severity in severities:
                        yield dataset, seed, optimizer, perturbation, float(severity)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V2 GNN robustness experiments.")
    parser.add_argument("--config", default="configs/v2_fixed_cora.yaml")
    parser.add_argument("--output-root", default="results/v2")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--max-runs", type=int, default=0, help="0 means no limit")
    parser.add_argument("--epochs", type=int, default=0, help="Override epochs for smoke runs")
    parser.add_argument("--seeds", default="", help="Optional comma-separated seed override")
    parser.add_argument(
        "--optimizers", default="", help="Optional comma-separated optimizer override"
    )
    parser.add_argument(
        "--perturbations", default="", help="Optional comma-separated perturbation override"
    )
    parser.add_argument(
        "--severities", default="", help="Optional comma-separated severity override"
    )
    parser.add_argument("--label", default="", help="Optional output filename suffix")
    args = parser.parse_args()

    config = load_v2_config(args.config)
    if args.epochs:
        object.__setattr__(config.training, "epochs", args.epochs)
    if args.seeds:
        object.__setattr__(
            config, "seeds", tuple(int(value) for value in args.seeds.split(",") if value)
        )
    if args.optimizers:
        object.__setattr__(
            config,
            "optimizers",
            tuple(value.strip() for value in args.optimizers.split(",") if value.strip()),
        )
    if args.perturbations:
        object.__setattr__(
            config,
            "perturbations",
            tuple(value.strip() for value in args.perturbations.split(",") if value.strip()),
        )
    if args.severities:
        object.__setattr__(
            config,
            "severities",
            tuple(float(value) for value in args.severities.split(",") if value),
        )

    paths = prepare_v2_output_dirs(args.output_root)
    hardware_path = write_hardware_metadata(args.output_root)
    rows = []
    pending = []
    all_conditions = list(planned_conditions(config))
    for index, (dataset, seed, optimizer, perturbation, severity) in enumerate(all_conditions):
        if args.max_runs and index >= args.max_runs:
            pending.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "optimizer": optimizer,
                    "perturbation_type": perturbation,
                    "requested_severity": severity,
                    "reason": "not_run_due_to_max_runs_limit",
                }
            )
            continue
        print(
            f"[v2] {config.robustness_setting} dataset={dataset} seed={seed} "
            f"optimizer={optimizer} perturbation={perturbation} severity={severity}",
            flush=True,
        )
        if config.robustness_setting == "inference_time":
            row = run_v2_inference_time_condition(
                dataset_name=dataset,
                data_root=args.data_root,
                optimizer_name=optimizer,
                base_seed=seed,
                perturbation_type=perturbation,
                severity=severity,
                config=config,
                hardware_metadata_path=str(hardware_path),
            )
        else:
            row = run_v2_training_time_condition(
                dataset_name=dataset,
                data_root=args.data_root,
                optimizer_name=optimizer,
                base_seed=seed,
                perturbation_type=perturbation,
                severity=severity,
                config=config,
                hardware_metadata_path=str(hardware_path),
            )
        rows.append(row)

    suffix = f"_{args.label}" if args.label else ""
    raw_path = paths["raw"] / f"{config.experiment_id}{suffix}.csv"
    pd.DataFrame(rows, columns=V2_RAW_COLUMNS).to_csv(raw_path, index=False)
    manifest = {
        "config": args.config,
        "output": str(raw_path),
        "completed_runs": len(rows),
        "planned_runs": len(all_conditions),
        "pending_runs": pending,
    }
    manifest_path = paths["metadata"] / f"{config.experiment_id}{suffix}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote raw results: {raw_path}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
