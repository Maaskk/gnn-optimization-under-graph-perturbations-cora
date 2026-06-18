from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import ExperimentConfig
from .data import load_cora
from .experiments import optimizer_names
from .structural_perturbations import add_fake_edges, remove_edges
from .train import run_training


def _plot_structural_metric(
    results: pd.DataFrame,
    metric: str,
    title: str,
    output_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    for optimizer, group in results.groupby("optimizer"):
        grouped = group.sort_values("severity")
        ax.plot(grouped["severity"], grouped[metric], marker="o", label=optimizer)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Perturbation severity")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def run_structural_track(
    output_dir: str | Path = "results",
    data_root: str | Path = "data",
    config: ExperimentConfig | None = None,
    include_amsgrad: bool = False,
) -> dict[str, Path]:
    """Run teammate structural robustness experiments with shared hyperparameters."""

    output_path = Path(output_dir)
    figures_dir = output_path / "figures"
    output_path.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    config = config or ExperimentConfig()

    dataset, data = load_cora(data_root)
    edge_removal_rows: list[dict] = []
    fake_edge_rows: list[dict] = []

    for seed in config.seeds:
        for severity in config.severities:
            removed_edge_index = remove_edges(data.edge_index, severity=severity, seed=seed)
            fake_edge_index = add_fake_edges(
                data.edge_index,
                num_nodes=int(data.num_nodes),
                severity=severity,
                seed=seed,
            )
            for optimizer_name in optimizer_names(include_amsgrad=include_amsgrad):
                print(
                    f"[edge_removal] seed={seed} optimizer={optimizer_name} severity={severity}",
                    flush=True,
                )
                edge_removal_rows.append(
                    run_training(
                        data=data,
                        num_features=dataset.num_node_features,
                        num_classes=dataset.num_classes,
                        optimizer_name=optimizer_name,
                        config=config,
                        perturbation_type="edge_removal",
                        severity=float(severity),
                        seed=seed,
                        edge_index_override=removed_edge_index,
                        track="teammate",
                    ).result_row
                )

                print(
                    f"[fake_edge_addition] seed={seed} optimizer={optimizer_name} "
                    f"severity={severity}",
                    flush=True,
                )
                fake_edge_rows.append(
                    run_training(
                        data=data,
                        num_features=dataset.num_node_features,
                        num_classes=dataset.num_classes,
                        optimizer_name=optimizer_name,
                        config=config,
                        perturbation_type="fake_edge_addition",
                        severity=float(severity),
                        seed=seed,
                        edge_index_override=fake_edge_index,
                        track="teammate",
                    ).result_row
                )

    edge_removal_results = pd.DataFrame(edge_removal_rows)
    fake_edge_results = pd.DataFrame(fake_edge_rows)

    edge_removal_path = output_path / "teammate_edge_removal_results.csv"
    fake_edge_path = output_path / "teammate_fake_edge_results.csv"
    edge_removal_results.to_csv(edge_removal_path, index=False)
    fake_edge_results.to_csv(fake_edge_path, index=False)

    figure_paths = [
        _plot_structural_metric(
            edge_removal_results,
            "test_accuracy",
            "Edge Removal Robustness: Test Accuracy",
            figures_dir / "edge_removal_accuracy.png",
        ),
        _plot_structural_metric(
            edge_removal_results,
            "macro_f1",
            "Edge Removal Robustness: Macro F1",
            figures_dir / "edge_removal_f1.png",
        ),
        _plot_structural_metric(
            fake_edge_results,
            "test_accuracy",
            "Fake Edge Addition Robustness: Test Accuracy",
            figures_dir / "fake_edge_accuracy.png",
        ),
        _plot_structural_metric(
            fake_edge_results,
            "macro_f1",
            "Fake Edge Addition Robustness: Macro F1",
            figures_dir / "fake_edge_f1.png",
        ),
    ]

    return {
        "edge_removal_results": edge_removal_path,
        "fake_edge_results": fake_edge_path,
        "figures": figure_paths,
    }

