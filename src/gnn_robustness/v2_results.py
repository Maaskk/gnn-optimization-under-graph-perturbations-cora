from __future__ import annotations

import math

import pandas as pd

GROUP_COLUMNS = [
    "dataset",
    "protocol",
    "robustness_setting",
    "optimizer",
    "perturbation_type",
    "requested_severity",
]


def _ci95_half_width(values: pd.Series) -> float:
    count = int(values.count())
    if count <= 1:
        return 0.0
    std = float(values.std(ddof=1))
    return float(1.96 * std / math.sqrt(count))


def _normalized_auc(points: pd.DataFrame) -> float:
    ordered = points.sort_values("requested_severity")
    severities = [float(value) for value in ordered["requested_severity"]]
    values = [float(value) for value in ordered["mean_test_accuracy"]]
    if not values:
        return 0.0
    if len(values) == 1 or math.isclose(max(severities), min(severities)):
        return values[0]
    area = 0.0
    for left in range(len(values) - 1):
        width = severities[left + 1] - severities[left]
        area += width * (values[left] + values[left + 1]) / 2
    return area / (max(severities) - min(severities))


def _add_robustness_auc(aggregated: pd.DataFrame) -> pd.DataFrame:
    auc_columns = [
        "dataset",
        "protocol",
        "robustness_setting",
        "optimizer",
        "perturbation_type",
    ]
    scores: dict[tuple, float] = {}
    for key, group in aggregated.groupby(auc_columns, dropna=False):
        scores[key] = round(_normalized_auc(group), 6)
    aggregated["robustness_score_auc"] = [
        scores[
            (
                row.dataset,
                row.protocol,
                row.robustness_setting,
                row.optimizer,
                row.perturbation_type,
            )
        ]
        for row in aggregated.itertuples()
    ]
    return aggregated


def aggregate_raw_results(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate V2 raw rows with honest seed counts and uncertainty."""

    grouped = raw.groupby(GROUP_COLUMNS, dropna=False)
    rows: list[dict] = []
    for key, group in grouped:
        row = dict(zip(GROUP_COLUMNS, key, strict=False))
        row["n_seeds"] = int(group["seed"].nunique())
        for metric in ("test_accuracy", "macro_f1"):
            row[f"mean_{metric}"] = round(float(group[metric].mean()), 6)
            row[f"std_{metric}"] = round(
                float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0, 6
            )
            row[f"ci95_{metric}_half_width"] = round(_ci95_half_width(group[metric]), 6)
        rows.append(row)

    aggregated = pd.DataFrame(rows)
    clean_lookup = {
        (
            row.dataset,
            row.protocol,
            row.robustness_setting,
            row.optimizer,
        ): row.mean_test_accuracy
        for row in aggregated.itertuples()
        if row.perturbation_type == "clean"
    }
    drops: list[float] = []
    for row in aggregated.itertuples():
        clean_value = clean_lookup.get(
            (row.dataset, row.protocol, row.robustness_setting, row.optimizer)
        )
        if clean_value is None or row.perturbation_type == "clean":
            drops.append(0.0)
        else:
            drops.append(round(float(clean_value - row.mean_test_accuracy), 6))
    aggregated["clean_to_perturbed_accuracy_drop"] = drops
    aggregated = _add_robustness_auc(aggregated)
    return aggregated.sort_values(GROUP_COLUMNS).reset_index(drop=True)
