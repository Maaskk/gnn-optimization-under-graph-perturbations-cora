from pathlib import Path

import pandas as pd

from .results import RESULT_COLUMNS


INPUT_RESULT_FILES = [
    "clean_optimizer_results.csv",
    "feature_noise_results.csv",
    "edge_removal_results.csv",
    "fake_edge_addition_results.csv",
]


def _format_score(value: float) -> str:
    return f"{value:.3f}"


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    rows = frame.astype(str).values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def combine_results(results_dir: str | Path = "results") -> pd.DataFrame:
    results_path = Path(results_dir)
    frames = []
    for filename in INPUT_RESULT_FILES:
        path = results_path / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing required result file: {path}")
        frame = pd.read_csv(path)
        if list(frame.columns) != RESULT_COLUMNS:
            raise ValueError(f"{path} does not match the shared result schema")
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[RESULT_COLUMNS]
    combined.to_csv(results_path / "all_results.csv", index=False)
    return combined


def build_optimizer_summary(combined: pd.DataFrame, results_dir: str | Path = "results") -> pd.DataFrame:
    summary = (
        combined.groupby(["optimizer", "perturbation_type"], as_index=False)
        .agg(
            mean_test_accuracy=("test_accuracy", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_training_time_seconds=("training_time_seconds", "mean"),
        )
        .sort_values(["perturbation_type", "mean_test_accuracy"], ascending=[True, False])
    )
    summary.to_csv(Path(results_dir) / "final_optimizer_summary.csv", index=False)
    return summary


def _best_optimizer(summary: pd.DataFrame, perturbation_type: str, metric: str) -> tuple[str, float]:
    subset = summary[summary["perturbation_type"] == perturbation_type]
    best = subset.sort_values(metric, ascending=False).iloc[0]
    return str(best["optimizer"]), float(best[metric])


def write_final_report(
    combined: pd.DataFrame,
    summary: pd.DataFrame,
    results_dir: str | Path = "results",
    reports_dir: str | Path = "reports",
) -> Path:
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(results_dir) / "cora_dataset_summary.csv"
    dataset = pd.read_csv(dataset_path).iloc[0].to_dict()

    clean_best, clean_acc = _best_optimizer(summary, "clean", "mean_test_accuracy")
    feature_best, feature_acc = _best_optimizer(summary, "feature_noise", "mean_test_accuracy")
    removal_best, removal_acc = _best_optimizer(summary, "edge_removal", "mean_test_accuracy")
    fake_best, fake_acc = _best_optimizer(summary, "fake_edge_addition", "mean_test_accuracy")

    overall = (
        combined.groupby("optimizer", as_index=False)
        .agg(
            mean_test_accuracy=("test_accuracy", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
        )
        .sort_values(["mean_test_accuracy", "mean_macro_f1"], ascending=False)
    )
    overall_best = overall.iloc[0]

    clean_table = combined[combined["perturbation_type"] == "clean"][
        ["optimizer", "test_accuracy", "macro_f1", "final_loss", "training_time_seconds"]
    ].sort_values("test_accuracy", ascending=False)
    for column in ["test_accuracy", "macro_f1", "final_loss", "training_time_seconds"]:
        clean_table[column] = clean_table[column].map(_format_score)

    summary_table = summary.copy()
    for column in ["mean_test_accuracy", "mean_macro_f1", "mean_training_time_seconds"]:
        summary_table[column] = summary_table[column].map(_format_score)

    report = f"""# Final Report - GNN Optimizer Robustness on Cora

## Project

**Title:** Optimization of Graph Neural Networks under Graph Perturbations: Adam vs AdamW vs RMSProp vs AdaGrad vs SGD on the Cora Citation Network

**Option:** Option 4 - Robustesse des GNN face au bruit dans le graphe

**Dataset:** Cora Citation Network

## Dataset And Model

The experiments use the Cora citation network from PyTorch Geometric's Planetoid dataset. Cora contains {int(dataset['num_nodes'])} nodes, {int(dataset['num_edges'])} directed citation edges, {int(dataset['num_features'])} node features, and {int(dataset['num_classes'])} classes. The standard split contains {int(dataset['train_nodes'])} training nodes, {int(dataset['validation_nodes'])} validation nodes, and {int(dataset['test_nodes'])} test nodes.

All experiments use a 2-layer Graph Convolutional Network with 16 hidden channels, dropout 0.5, learning rate 0.01, weight decay 0.0005, seed 42, and 200 epochs. Keeping this protocol fixed makes the optimizer comparison fair across clean training, feature noise, edge removal, and fake edge addition.

## Perturbations

- **Clean graph:** original Cora graph and features.
- **Feature noise:** Gaussian noise added to node features at severities 0.05, 0.10, 0.20, and 0.30.
- **Edge removal:** random removal of a fraction of graph edges at the same severities.
- **Fake edge addition:** random non-existing directed edges added at the same severities.

## Clean Graph Results

{_markdown_table(clean_table)}

Adam and AdamW achieve the strongest clean accuracy in this run, both reaching {_format_score(clean_acc)} test accuracy. RMSProp is very close, while AdaGrad and especially SGD are weaker under the fixed hyperparameters.

## Robustness Summary

{_markdown_table(summary_table)}

Best mean test accuracy by setting:

- Clean graph: **{clean_best}** with {_format_score(clean_acc)}.
- Feature noise: **{feature_best}** with {_format_score(feature_acc)}.
- Edge removal: **{removal_best}** with {_format_score(removal_acc)}.
- Fake edge addition: **{fake_best}** with {_format_score(fake_acc)}.

## Final Conclusion

Across all result rows, the best overall optimizer by mean test accuracy is **{overall_best['optimizer']}** with {_format_score(float(overall_best['mean_test_accuracy']))} mean accuracy and {_format_score(float(overall_best['mean_macro_f1']))} mean macro F1.

The main pattern is that adaptive optimizers are much stronger than SGD for this Cora GCN setup. Adam, AdamW, and RMSProp are the most competitive optimizers. Perturbations reduce performance compared with the clean graph, but the adaptive methods remain more stable than SGD.

## Reproducibility Package

The repository contains the experiment scripts, source modules, generated tables, figures, and tests needed to reproduce and inspect the study. Key generated files:

- `results/all_results.csv`
- `results/final_optimizer_summary.csv`
- `reports/final_report.md`
- `reports/baseline_feature_noise_analysis.md`
- figures in `results/figures/`
"""
    path = reports_path / "final_report.md"
    path.write_text(report, encoding="utf-8")
    return path


def finalize_project(
    results_dir: str | Path = "results",
    reports_dir: str | Path = "reports",
) -> dict[str, Path]:
    combined = combine_results(results_dir)
    summary = build_optimizer_summary(combined, results_dir)
    report_path = write_final_report(combined, summary, results_dir, reports_dir)
    return {
        "all_results": Path(results_dir) / "all_results.csv",
        "summary": Path(results_dir) / "final_optimizer_summary.csv",
        "report": report_path,
    }
