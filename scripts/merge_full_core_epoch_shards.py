from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

OPTIMIZER_ORDER = ["Adam", "AdamW", "RMSProp", "AdaGrad", "SGD"]
KEY_COLUMNS = ["optimizer", "seed", "perturbation_type", "requested_severity"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge per-optimizer full epoch-history shards.")
    parser.add_argument("--shard-root", default="results/v2/proof_shards")
    parser.add_argument("--output-dir", default="results/v2/proof")
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()

    shard_root = Path(args.shard_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epoch_frames = []
    run_frames = []
    shard_summary = []

    for optimizer in OPTIMIZER_ORDER:
        shard_dir = shard_root / optimizer
        epoch_path = shard_dir / "full_core_epoch_history.csv"
        run_path = shard_dir / "full_core_run_results.csv"
        if not epoch_path.exists() or not run_path.exists():
            raise FileNotFoundError(f"Missing shard output for {optimizer}: {shard_dir}")
        epoch_data = pd.read_csv(epoch_path)
        run_data = pd.read_csv(run_path)
        if len(run_data) != 130:
            raise AssertionError(f"{optimizer} shard has {len(run_data)} runs, expected 130")
        if len(epoch_data) != 130 * args.epochs:
            raise AssertionError(
                f"{optimizer} shard has {len(epoch_data)} epoch rows, expected {130 * args.epochs}"
            )
        per_run_epochs = epoch_data.groupby(KEY_COLUMNS, dropna=False)["epoch"].nunique()
        if int(per_run_epochs.min()) != args.epochs or int(per_run_epochs.max()) != args.epochs:
            raise AssertionError(f"{optimizer} shard does not have {args.epochs} epochs per run")
        epoch_frames.append(epoch_data)
        run_frames.append(run_data)
        shard_summary.append(
            {
                "optimizer": optimizer,
                "runs": int(len(run_data)),
                "epoch_rows": int(len(epoch_data)),
                "min_epochs_per_run": int(per_run_epochs.min()),
                "max_epochs_per_run": int(per_run_epochs.max()),
            }
        )

    epoch_all = pd.concat(epoch_frames, ignore_index=True)
    run_all = pd.concat(run_frames, ignore_index=True)
    epoch_all["optimizer_order"] = epoch_all["optimizer"].map(
        {name: index for index, name in enumerate(OPTIMIZER_ORDER)}
    )
    run_all["optimizer_order"] = run_all["optimizer"].map(
        {name: index for index, name in enumerate(OPTIMIZER_ORDER)}
    )
    epoch_all = (
        epoch_all.sort_values(
            ["optimizer_order", "seed", "perturbation_type", "requested_severity", "epoch"]
        )
        .drop(columns=["optimizer_order"])
        .reset_index(drop=True)
    )
    run_all = (
        run_all.sort_values(["optimizer_order", "seed", "perturbation_type", "requested_severity"])
        .drop(columns=["optimizer_order"])
        .reset_index(drop=True)
    )

    if len(run_all) != 650:
        raise AssertionError(f"Merged run file has {len(run_all)} runs, expected 650")
    if len(epoch_all) != 650 * args.epochs:
        raise AssertionError(
            f"Merged epoch file has {len(epoch_all)} rows, expected {650 * args.epochs}"
        )
    per_run_epochs = epoch_all.groupby(KEY_COLUMNS, dropna=False)["epoch"].nunique()
    if int(per_run_epochs.min()) != args.epochs or int(per_run_epochs.max()) != args.epochs:
        raise AssertionError("Merged proof file does not have exactly 200 epochs per run")

    epoch_output = output_dir / "full_core_epoch_history.csv"
    run_output = output_dir / "full_core_run_results.csv"
    epoch_all.to_csv(epoch_output, index=False)
    run_all.to_csv(run_output, index=False)

    first_environment = shard_root / OPTIMIZER_ORDER[0] / "environment.json"
    if first_environment.exists():
        shutil.copy2(first_environment, output_dir / "environment.json")

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source_shard_root": str(shard_root),
        "output_dir": str(output_dir),
        "optimizers": OPTIMIZER_ORDER,
        "expected_runs": 650,
        "observed_runs": int(len(run_all)),
        "epochs_per_run": args.epochs,
        "expected_epoch_rows": 650 * args.epochs,
        "observed_epoch_rows": int(len(epoch_all)),
        "shards": shard_summary,
    }
    manifest_path = output_dir / "full_core_epoch_history_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
