#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gnn_robustness.v2_statistics import build_optimizer_comparison_table  # noqa: E402


def parse_csv_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate matched-seed optimizer statistics.")
    parser.add_argument("--raw", default="results/v2/raw/v2_fixed_cora.csv")
    parser.add_argument(
        "--output", default="results/v2/statistics/optimizer_paired_comparisons.csv"
    )
    parser.add_argument("--dataset", default="Cora")
    parser.add_argument("--protocol", default="fixed")
    parser.add_argument("--robustness-setting", default="training_time")
    parser.add_argument("--optimizers", default="Adam,AdamW,RMSProp,AdaGrad,SGD")
    parser.add_argument("--metrics", default="test_accuracy,macro_f1")
    parser.add_argument("--n-boot", type=int, default=5000)
    args = parser.parse_args()

    raw = pd.read_csv(args.raw)
    table = build_optimizer_comparison_table(
        raw,
        optimizers=parse_csv_list(args.optimizers),
        metrics=parse_csv_list(args.metrics),
        dataset=args.dataset,
        protocol=args.protocol,
        robustness_setting=args.robustness_setting,
        n_boot=args.n_boot,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(f"Wrote paired statistics: {output}")


if __name__ == "__main__":
    main()
