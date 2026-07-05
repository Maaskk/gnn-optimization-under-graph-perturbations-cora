#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gnn_robustness.v2_results import aggregate_raw_results  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate V2 raw experiment results.")
    parser.add_argument("--raw-dir", default="results/v2/raw")
    parser.add_argument("--output-dir", default="results/v2/aggregated")
    args = parser.parse_args()

    raw_paths = sorted(Path(args.raw_dir).glob("*.csv"))
    if not raw_paths:
        raise FileNotFoundError(f"No V2 raw CSV files found in {args.raw_dir}")
    raw = pd.concat([pd.read_csv(path) for path in raw_paths], ignore_index=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / "v2_raw_combined.csv"
    aggregate_path = output_dir / "v2_aggregated_summary.csv"
    raw.to_csv(combined_path, index=False)
    aggregate_raw_results(raw).to_csv(aggregate_path, index=False)
    print(f"Wrote combined raw results: {combined_path}")
    print(f"Wrote aggregate summary: {aggregate_path}")


if __name__ == "__main__":
    main()
