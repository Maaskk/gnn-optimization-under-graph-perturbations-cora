import math

from gnn_robustness.v2_statistics import (
    build_optimizer_comparison_table,
    holm_correction,
    paired_bootstrap_ci,
    wilcoxon_signed_rank,
)


def test_paired_bootstrap_ci_is_deterministic_and_matched_by_seed():
    adam = [0.81, 0.80, 0.79, 0.82]
    rmsprop = [0.78, 0.77, 0.79, 0.80]

    first = paired_bootstrap_ci(adam, rmsprop, n_boot=200, seed=123)
    second = paired_bootstrap_ci(adam, rmsprop, n_boot=200, seed=123)

    assert first == second
    assert first["n_pairs"] == 4
    assert math.isclose(first["mean_difference"], 0.02, abs_tol=1e-12)
    assert first["ci_low"] <= first["mean_difference"] <= first["ci_high"]


def test_wilcoxon_signed_rank_returns_bounded_p_value_without_scipy_dependency():
    result = wilcoxon_signed_rank([0.81, 0.82, 0.83, 0.84], [0.75, 0.77, 0.78, 0.79])

    assert result["n_pairs"] == 4
    assert result["statistic"] >= 0
    assert 0.0 <= result["p_value"] <= 1.0
    assert result["method"] == "normal_approximation"


def test_holm_correction_adjusts_monotonically_and_tracks_rejections():
    adjusted = holm_correction(
        {
            "Adam_vs_SGD": 0.001,
            "Adam_vs_AdamW": 0.04,
            "Adam_vs_RMSProp": 0.02,
        },
        alpha=0.05,
    )

    by_label = {row["comparison"]: row for row in adjusted}
    assert (
        by_label["Adam_vs_SGD"]["adjusted_p_value"]
        <= by_label["Adam_vs_RMSProp"]["adjusted_p_value"]
    )
    assert (
        by_label["Adam_vs_RMSProp"]["adjusted_p_value"]
        <= by_label["Adam_vs_AdamW"]["adjusted_p_value"]
    )
    assert by_label["Adam_vs_SGD"]["reject_null"] is True


def test_build_optimizer_comparison_table_uses_matched_seed_means():
    import pandas as pd

    raw = pd.DataFrame(
        [
            {
                "dataset": "Cora",
                "protocol": "fixed",
                "robustness_setting": "training_time",
                "optimizer": optimizer,
                "seed": seed,
                "test_accuracy": base + offset,
                "macro_f1": base + offset - 0.02,
            }
            for optimizer, offset in [("Adam", 0.04), ("RMSProp", 0.02), ("SGD", -0.08)]
            for seed, base in [(42, 0.70), (43, 0.72), (44, 0.74)]
        ]
    )

    table = build_optimizer_comparison_table(
        raw,
        optimizers=("Adam", "RMSProp", "SGD"),
        metrics=("test_accuracy",),
        n_boot=100,
    )

    assert set(table["Metric"]) == {"test_accuracy"}
    assert {"Optimizer A", "Optimizer B", "Mean difference", "CI95", "Adjusted p-value"}.issubset(
        table.columns
    )
    adam_sgd = table[(table["Optimizer A"] == "Adam") & (table["Optimizer B"] == "SGD")].iloc[0]
    assert adam_sgd["Matched seeds"] == 3
    assert adam_sgd["Mean difference"] > 0
    assert "under this protocol" in adam_sgd["Interpretation"]
