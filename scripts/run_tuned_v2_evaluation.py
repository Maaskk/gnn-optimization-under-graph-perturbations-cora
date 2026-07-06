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

from gnn_robustness.v2_config import load_v2_config  # noqa: E402
from gnn_robustness.v2_experiments import (  # noqa: E402
    V2_RAW_COLUMNS,
    prepare_v2_output_dirs,
    run_v2_training_time_condition,
    write_hardware_metadata,
)
from gnn_robustness.v2_tuning import (  # noqa: E402
    apply_locked_hyperparameters,
    load_locked_hyperparameters,
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
    parser = argparse.ArgumentParser(
        description="Run final test evaluation with validation-locked V2 hyperparameters."
    )
    parser.add_argument("--config", default="configs/v2_tuned_cora.yaml")
    parser.add_argument("--locked", default="results/v2/tuned/locked_hyperparameters_final.json")
    parser.add_argument("--output-root", default="results/v2")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    config = load_v2_config(args.config)
    if not config.tuning.enabled:
        raise SystemExit("The tuned evaluation config must enable tuning.")
    locked = load_locked_hyperparameters(args.locked)
    paths = prepare_v2_output_dirs(args.output_root)
    hardware_path = write_hardware_metadata(args.output_root)
    rows = []
    all_conditions = list(planned_conditions(config))
    for dataset, seed, optimizer, perturbation, severity in all_conditions:
        tuned_config, momentum = apply_locked_hyperparameters(config, optimizer, locked)
        tuned_config = replace(
            tuned_config,
            tuning=replace(tuned_config.tuning, sgd_momentum=momentum),
        )
        print(
            f"[tuned] dataset={dataset} seed={seed} optimizer={optimizer} "
            f"perturbation={perturbation} severity={severity}",
            flush=True,
        )
        rows.append(
            run_v2_training_time_condition(
                dataset_name=dataset,
                data_root=args.data_root,
                optimizer_name=optimizer,
                base_seed=seed,
                perturbation_type=perturbation,
                severity=severity,
                config=tuned_config,
                hardware_metadata_path=str(hardware_path),
            )
        )

    suffix = f"_{args.label}" if args.label else ""
    raw_path = paths["raw"] / f"{config.experiment_id}{suffix}.csv"
    pd.DataFrame(rows, columns=V2_RAW_COLUMNS).to_csv(raw_path, index=False)
    manifest = {
        "config": args.config,
        "locked_hyperparameters": args.locked,
        "output": str(raw_path),
        "completed_runs": len(rows),
        "planned_runs": len(all_conditions),
        "pending_runs": [],
        "uses_test_split_for_tuning": False,
    }
    manifest_path = paths["metadata"] / f"{config.experiment_id}{suffix}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote tuned raw results: {raw_path}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
