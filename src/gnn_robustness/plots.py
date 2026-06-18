from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _prepare_output_dir(output_dir: Path) -> Path:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def plot_loss_convergence(history: pd.DataFrame, output_dir: Path) -> Path:
    figures_dir = _prepare_output_dir(output_dir)
    clean_history = history[history["perturbation_type"] == "clean"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for optimizer, group in clean_history.groupby("optimizer"):
        grouped = group.groupby("epoch", as_index=False)["train_loss"].mean()
        ax.plot(grouped["epoch"], grouped["train_loss"], label=optimizer)
    ax.set_title("Clean Cora GCN Loss Convergence")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.legend()
    ax.grid(alpha=0.25)
    path = figures_dir / "ossama_clean_loss_convergence.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_clean_bars(clean_results: pd.DataFrame, output_dir: Path) -> Path:
    figures_dir = _prepare_output_dir(output_dir)
    grouped = clean_results.groupby("optimizer", as_index=False)[["test_accuracy", "macro_f1"]].mean()
    x = range(len(grouped))
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.35
    ax.bar([i - width / 2 for i in x], grouped["test_accuracy"], width, label="Accuracy")
    ax.bar([i + width / 2 for i in x], grouped["macro_f1"], width, label="Macro F1")
    ax.set_xticks(list(x))
    ax.set_xticklabels(grouped["optimizer"], rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Clean Cora Test Performance by Optimizer")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    path = figures_dir / "ossama_clean_accuracy_macro_f1.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_feature_noise_lines(noise_results: pd.DataFrame, output_dir: Path, metric: str) -> Path:
    figures_dir = _prepare_output_dir(output_dir)
    fig, ax = plt.subplots(figsize=(9, 5))
    for optimizer, group in noise_results.groupby("optimizer"):
        grouped = group.groupby("severity", as_index=False)[metric].mean().sort_values("severity")
        ax.plot(grouped["severity"], grouped[metric], marker="o", label=optimizer)
    ax.set_ylim(0, 1)
    ax.set_title(f"Feature Noise Robustness: {metric.replace('_', ' ').title()}")
    ax.set_xlabel("Gaussian noise severity")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.legend()
    ax.grid(alpha=0.25)
    path = figures_dir / f"ossama_feature_noise_{metric}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_all_plots(
    clean_results: pd.DataFrame,
    noise_results: pd.DataFrame,
    history: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    return [
        plot_loss_convergence(history, output_dir),
        plot_clean_bars(clean_results, output_dir),
        plot_feature_noise_lines(noise_results, output_dir, "test_accuracy"),
        plot_feature_noise_lines(noise_results, output_dir, "macro_f1"),
    ]
