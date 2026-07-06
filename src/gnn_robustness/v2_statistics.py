from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from itertools import combinations

import pandas as pd


def _paired_differences(
    first: Sequence[float],
    second: Sequence[float],
) -> list[float]:
    if len(first) != len(second):
        raise ValueError("Paired comparisons require the same number of matched seeds")
    if not first:
        raise ValueError("At least one matched pair is required")
    return [float(a) - float(b) for a, b in zip(first, second, strict=True)]


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute percentile of an empty sample")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(sorted_values[lower])
    weight = rank - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


def paired_bootstrap_ci(
    first: Sequence[float],
    second: Sequence[float],
    *,
    n_boot: int = 5000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float | int | str]:
    """Bootstrap a matched-seed optimizer difference confidence interval."""

    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    diffs = _paired_differences(first, second)
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        estimates.append(sum(sample) / len(sample))
    estimates.sort()
    return {
        "n_pairs": len(diffs),
        "mean_difference": sum(diffs) / len(diffs),
        "ci_low": _percentile(estimates, alpha / 2),
        "ci_high": _percentile(estimates, 1 - alpha / 2),
        "method": "paired_bootstrap",
    }


def _average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and math.isclose(ordered[end][1], ordered[index][1]):
            end += 1
        average_rank = (index + 1 + end) / 2
        for original_index, _ in ordered[index:end]:
            ranks[original_index] = average_rank
        index = end
    return ranks


def wilcoxon_signed_rank(
    first: Sequence[float],
    second: Sequence[float],
) -> dict[str, float | int | str]:
    """Wilcoxon signed-rank normal approximation for matched optimizer rows."""

    diffs = [value for value in _paired_differences(first, second) if not math.isclose(value, 0.0)]
    n = len(diffs)
    if n == 0:
        return {
            "n_pairs": 0,
            "statistic": 0.0,
            "p_value": 1.0,
            "method": "all_zero_differences",
        }

    ranks = _average_ranks([abs(value) for value in diffs])
    w_plus = sum(rank for rank, diff in zip(ranks, diffs, strict=True) if diff > 0)
    w_minus = sum(rank for rank, diff in zip(ranks, diffs, strict=True) if diff < 0)
    statistic = min(w_plus, w_minus)
    expected = n * (n + 1) / 4
    variance = n * (n + 1) * (2 * n + 1) / 24
    if variance <= 0:
        p_value = 1.0
    else:
        z_score = (statistic - expected) / math.sqrt(variance)
        p_value = math.erfc(abs(z_score) / math.sqrt(2))
    return {
        "n_pairs": n,
        "statistic": float(statistic),
        "p_value": max(0.0, min(1.0, float(p_value))),
        "method": "normal_approximation",
    }


def holm_correction(
    p_values: Mapping[str, float],
    *,
    alpha: float = 0.05,
) -> list[dict[str, float | bool | str | int]]:
    """Apply Holm-Bonferroni correction and return rows in comparison order."""

    total = len(p_values)
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    rows: list[dict[str, float | bool | str | int]] = []
    running_max = 0.0
    for rank, (comparison, p_value) in enumerate(ordered, start=1):
        adjusted = min(1.0, float(p_value) * (total - rank + 1))
        running_max = max(running_max, adjusted)
        rows.append(
            {
                "comparison": comparison,
                "rank": rank,
                "p_value": float(p_value),
                "adjusted_p_value": running_max,
                "reject_null": running_max <= alpha,
                "method": "Holm-Bonferroni",
            }
        )
    return rows


def _optimizer_seed_means(
    raw: pd.DataFrame,
    *,
    optimizer: str,
    metric: str,
    dataset: str,
    protocol: str,
    robustness_setting: str,
) -> pd.Series:
    filtered = raw[
        (raw["dataset"] == dataset)
        & (raw["protocol"] == protocol)
        & (raw["robustness_setting"] == robustness_setting)
        & (raw["optimizer"] == optimizer)
    ]
    return filtered.groupby("seed")[metric].mean().sort_index()


def _interpret_difference(ci_low: float, ci_high: float, adjusted_p_value: float) -> str:
    if ci_low <= 0 <= ci_high:
        return "Evidence is insufficient to distinguish clearly under this protocol."
    if adjusted_p_value <= 0.05:
        return "Difference is statistically distinguishable under this protocol."
    return "Mean is higher under this protocol, but adjusted evidence remains cautious."


def build_optimizer_comparison_table(
    raw: pd.DataFrame,
    *,
    optimizers: Sequence[str],
    metrics: Sequence[str],
    dataset: str = "Cora",
    protocol: str = "fixed",
    robustness_setting: str = "training_time",
    n_boot: int = 5000,
) -> pd.DataFrame:
    """Build matched-seed optimizer comparison rows with bootstrap and Holm tests."""

    rows: list[dict[str, object]] = []
    p_values: dict[str, float] = {}
    row_indices: dict[str, int] = {}
    for metric in metrics:
        for optimizer_a, optimizer_b in combinations(optimizers, 2):
            first = _optimizer_seed_means(
                raw,
                optimizer=optimizer_a,
                metric=metric,
                dataset=dataset,
                protocol=protocol,
                robustness_setting=robustness_setting,
            )
            second = _optimizer_seed_means(
                raw,
                optimizer=optimizer_b,
                metric=metric,
                dataset=dataset,
                protocol=protocol,
                robustness_setting=robustness_setting,
            )
            matched = first.to_frame("first").join(second.to_frame("second"), how="inner")
            if matched.empty:
                continue
            boot = paired_bootstrap_ci(
                matched["first"].tolist(),
                matched["second"].tolist(),
                n_boot=n_boot,
                seed=42 + len(rows),
            )
            wilcoxon = wilcoxon_signed_rank(
                matched["first"].tolist(),
                matched["second"].tolist(),
            )
            comparison = f"{metric}:{optimizer_a}_vs_{optimizer_b}"
            p_values[comparison] = float(wilcoxon["p_value"])
            row_indices[comparison] = len(rows)
            rows.append(
                {
                    "Optimizer A": optimizer_a,
                    "Optimizer B": optimizer_b,
                    "Metric": metric,
                    "Matched seeds": int(boot["n_pairs"]),
                    "Mean difference": round(float(boot["mean_difference"]), 6),
                    "CI95": f"[{float(boot['ci_low']):.6f}, {float(boot['ci_high']):.6f}]",
                    "CI95 low": round(float(boot["ci_low"]), 6),
                    "CI95 high": round(float(boot["ci_high"]), 6),
                    "Wilcoxon p-value": round(float(wilcoxon["p_value"]), 6),
                    "Adjusted p-value": 1.0,
                    "Interpretation": "Pending Holm correction.",
                }
            )

    for adjusted in holm_correction(p_values):
        row = rows[row_indices[str(adjusted["comparison"])]]
        adjusted_p_value = float(adjusted["adjusted_p_value"])
        row["Adjusted p-value"] = round(adjusted_p_value, 6)
        row["Interpretation"] = _interpret_difference(
            float(row["CI95 low"]),
            float(row["CI95 high"]),
            adjusted_p_value,
        )
    return pd.DataFrame(rows)
