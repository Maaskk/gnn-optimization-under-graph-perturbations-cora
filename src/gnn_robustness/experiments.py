from pathlib import Path

import pandas as pd

from .config import ExperimentConfig
from .data import cora_summary, load_cora
from .optimizers import BONUS_OPTIMIZERS, DEFAULT_OPTIMIZERS
from .plots import write_all_plots
from .train import run_training


def optimizer_names(include_amsgrad: bool = False) -> list[str]:
    names = list(DEFAULT_OPTIMIZERS)
    if include_amsgrad:
        names.extend(BONUS_OPTIMIZERS)
    return names


def run_ossama_track(
    output_dir: str | Path = "results",
    data_root: str | Path = "data",
    config: ExperimentConfig | None = None,
    include_amsgrad: bool = False,
) -> dict[str, Path | dict]:
    """Run clean Cora and feature-noise robustness experiments."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    config = config or ExperimentConfig()

    dataset, data = load_cora(data_root)
    clean_rows: list[dict] = []
    noise_rows: list[dict] = []
    history_rows: list[dict] = []

    for seed in config.seeds:
        for optimizer_name in optimizer_names(include_amsgrad=include_amsgrad):
            print(f"[clean] seed={seed} optimizer={optimizer_name}", flush=True)
            clean_outcome = run_training(
                data=data,
                num_features=dataset.num_node_features,
                num_classes=dataset.num_classes,
                optimizer_name=optimizer_name,
                config=config,
                perturbation_type="clean",
                severity=0.0,
                seed=seed,
            )
            clean_rows.append(clean_outcome.result_row)
            history_rows.extend(clean_outcome.history_rows)

            for severity in config.severities:
                print(
                    f"[feature_noise] seed={seed} optimizer={optimizer_name} "
                    f"severity={severity}",
                    flush=True,
                )
                noise_outcome = run_training(
                    data=data,
                    num_features=dataset.num_node_features,
                    num_classes=dataset.num_classes,
                    optimizer_name=optimizer_name,
                    config=config,
                    perturbation_type="feature_noise",
                    severity=float(severity),
                    seed=seed,
                )
                noise_rows.append(noise_outcome.result_row)
                history_rows.extend(noise_outcome.history_rows)

    clean_results = pd.DataFrame(clean_rows)
    noise_results = pd.DataFrame(noise_rows)
    history = pd.DataFrame(history_rows)

    clean_path = output_path / "ossama_clean_results.csv"
    noise_path = output_path / "ossama_feature_noise_results.csv"
    history_path = output_path / "ossama_loss_history.csv"
    summary_path = output_path / "cora_dataset_summary.csv"

    clean_results.to_csv(clean_path, index=False)
    noise_results.to_csv(noise_path, index=False)
    history.to_csv(history_path, index=False)
    pd.DataFrame([cora_summary(dataset, data)]).to_csv(summary_path, index=False)
    figure_paths = write_all_plots(clean_results, noise_results, history, output_path)

    return {
        "clean_results": clean_path,
        "feature_noise_results": noise_path,
        "history": history_path,
        "dataset_summary": summary_path,
        "figures": figure_paths,
        "dataset": cora_summary(dataset, data),
    }
