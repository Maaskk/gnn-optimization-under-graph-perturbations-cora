import pandas as pd

from gnn_robustness.v2_results import aggregate_raw_results


def test_v2_aggregation_reports_mean_std_ci_and_clean_drop():
    raw = pd.DataFrame(
        [
            {
                "dataset": "Cora",
                "protocol": "fixed",
                "robustness_setting": "training_time",
                "optimizer": "Adam",
                "seed": 42,
                "perturbation_type": "clean",
                "requested_severity": 0.0,
                "test_accuracy": 0.80,
                "macro_f1": 0.79,
            },
            {
                "dataset": "Cora",
                "protocol": "fixed",
                "robustness_setting": "training_time",
                "optimizer": "Adam",
                "seed": 43,
                "perturbation_type": "clean",
                "requested_severity": 0.0,
                "test_accuracy": 0.82,
                "macro_f1": 0.80,
            },
            {
                "dataset": "Cora",
                "protocol": "fixed",
                "robustness_setting": "training_time",
                "optimizer": "Adam",
                "seed": 42,
                "perturbation_type": "feature_masking",
                "requested_severity": 0.2,
                "test_accuracy": 0.70,
                "macro_f1": 0.68,
            },
            {
                "dataset": "Cora",
                "protocol": "fixed",
                "robustness_setting": "training_time",
                "optimizer": "Adam",
                "seed": 43,
                "perturbation_type": "feature_masking",
                "requested_severity": 0.2,
                "test_accuracy": 0.74,
                "macro_f1": 0.71,
            },
        ]
    )

    aggregated = aggregate_raw_results(raw)
    perturbed = aggregated[
        (aggregated["perturbation_type"] == "feature_masking")
        & (aggregated["requested_severity"] == 0.2)
    ].iloc[0]

    assert perturbed["n_seeds"] == 2
    assert perturbed["mean_test_accuracy"] == 0.72
    assert perturbed["std_test_accuracy"] > 0
    assert perturbed["ci95_test_accuracy_half_width"] > 0
    assert perturbed["clean_to_perturbed_accuracy_drop"] == 0.09
    assert "robustness_score_auc" in aggregated.columns
    assert 0.0 < perturbed["robustness_score_auc"] <= 1.0
