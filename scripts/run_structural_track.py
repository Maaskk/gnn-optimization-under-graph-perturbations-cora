#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gnn_robustness.config import ExperimentConfig
from gnn_robustness.structural_experiments import run_structural_track


def parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run structural Cora GCN robustness experiments."
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seeds", type=parse_int_list, default=(42,))
    parser.add_argument("--severities", type=parse_float_list, default=(0.05, 0.10, 0.20, 0.30))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--include-amsgrad", action="store_true")
    args = parser.parse_args()

    config = ExperimentConfig(
        epochs=args.epochs,
        hidden_channels=args.hidden_channels,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seeds=args.seeds,
        severities=args.severities,
        device=args.device,
    )
    outputs = run_structural_track(
        output_dir=args.output_dir,
        data_root=args.data_root,
        config=config,
        include_amsgrad=args.include_amsgrad,
    )
    print("Generated outputs:")
    for key, value in outputs.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()

