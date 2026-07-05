from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence


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
