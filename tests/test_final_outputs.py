from pathlib import Path

import pandas as pd

from gnn_robustness.results import RESULT_COLUMNS


def test_all_results_contains_all_project_perturbations():
    path = Path("results/all_results.csv")
    assert path.exists()

    df = pd.read_csv(path)

    assert list(df.columns) == RESULT_COLUMNS
    assert set(df["optimizer"]) == {"Adam", "AdamW", "RMSProp", "AdaGrad", "SGD"}
    assert set(df["perturbation_type"]) == {
        "clean",
        "feature_noise",
        "edge_removal",
        "fake_edge_addition",
    }
    assert len(df[df["perturbation_type"] == "clean"]) == 5
    assert len(df[df["perturbation_type"] == "feature_noise"]) == 20
    assert len(df[df["perturbation_type"] == "edge_removal"]) == 20
    assert len(df[df["perturbation_type"] == "fake_edge_addition"]) == 20
    assert set(df["hidden_channels"]) == {16}


def test_final_report_and_summary_exist():
    assert Path("reports/final_report.md").exists()
    assert Path("results/final_optimizer_summary.csv").exists()
