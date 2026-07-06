#!/usr/bin/env python
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOCS_DATA = ROOT / "docs" / "assets" / "data"
DOCS_DOWNLOADS = ROOT / "docs" / "assets" / "downloads"
V2_ROOT = ROOT / "results" / "v2"


def copy_if_exists(source: Path, target: Path) -> bool:
    if not source.exists():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return True


def count_pending_runs() -> tuple[int, int]:
    completed = 0
    pending = 0
    for manifest_path in sorted((V2_ROOT / "metadata").glob("*_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        completed += int(manifest.get("completed_runs", 0))
        pending += len(manifest.get("pending_runs", []))
    return completed, pending


def copy_downloads() -> dict[str, bool]:
    targets = {
        "config_v2_fixed_cora": (
            ROOT / "configs" / "v2_fixed_cora.yaml",
            DOCS_DOWNLOADS / "configs" / "v2_fixed_cora.yaml",
        ),
        "config_v2_inference": (
            ROOT / "configs" / "v2_inference_robustness.yaml",
            DOCS_DOWNLOADS / "configs" / "v2_inference_robustness.yaml",
        ),
        "config_v2_tuned": (
            ROOT / "configs" / "v2_tuned_cora.yaml",
            DOCS_DOWNLOADS / "configs" / "v2_tuned_cora.yaml",
        ),
        "report_v2_md": (
            ROOT / "reports" / "final_report_v2.md",
            DOCS_DOWNLOADS / "reports" / "final_report_v2.md",
        ),
        "report_v2_tex": (
            ROOT / "reports" / "final_report_v2.tex",
            DOCS_DOWNLOADS / "reports" / "final_report_v2.tex",
        ),
        "report_v2_pdf": (
            ROOT / "reports" / "final_report_v2.pdf",
            DOCS_DOWNLOADS / "reports" / "final_report_v2.pdf",
        ),
        "notebook_v2": (
            ROOT / "notebooks" / "GNN_Robustness_V2_Reproducibility.ipynb",
            DOCS_DOWNLOADS / "notebooks" / "GNN_Robustness_V2_Reproducibility.ipynb",
        ),
        "benchmark_smoke": (
            V2_ROOT / "benchmarks" / "runtime_benchmark_smoke.csv",
            DOCS_DOWNLOADS / "benchmarks" / "runtime_benchmark_smoke.csv",
        ),
        "tuned_locked_smoke": (
            V2_ROOT / "tuned" / "locked_hyperparameters_smoke.json",
            DOCS_DOWNLOADS / "tuned" / "locked_hyperparameters_smoke.json",
        ),
        "diagnostics_runs": (
            V2_ROOT / "diagnostics" / "v2_optimizer_diagnostics_runs_clean.csv",
            DOCS_DOWNLOADS / "diagnostics" / "v2_optimizer_diagnostics_runs_clean.csv",
        ),
        "diagnostics_gradient_history": (
            V2_ROOT / "diagnostics" / "v2_gradient_history_clean.csv",
            DOCS_DOWNLOADS / "diagnostics" / "v2_gradient_history_clean.csv",
        ),
        "diagnostics_gradient_summary": (
            V2_ROOT / "diagnostics" / "v2_gradient_summary_clean.csv",
            DOCS_DOWNLOADS / "diagnostics" / "v2_gradient_summary_clean.csv",
        ),
        "diagnostics_optimizer_summary": (
            V2_ROOT / "diagnostics" / "v2_optimizer_diagnostics_summary_clean.csv",
            DOCS_DOWNLOADS / "diagnostics" / "v2_optimizer_diagnostics_summary_clean.csv",
        ),
    }
    copied: dict[str, bool] = {}
    for key, (source, target) in targets.items():
        copied[key] = copy_if_exists(source, target)
    return copied


def main() -> None:
    DOCS_DATA.mkdir(parents=True, exist_ok=True)
    aggregate_source = V2_ROOT / "aggregated" / "v2_aggregated_summary.csv"
    raw_source = V2_ROOT / "aggregated" / "v2_raw_combined.csv"
    embedding_source = V2_ROOT / "embedding" / "cora_adam_seed42_clean_pca.json"
    copied_aggregate = copy_if_exists(aggregate_source, DOCS_DATA / "v2_aggregated_summary.csv")
    copied_raw = copy_if_exists(raw_source, DOCS_DATA / "v2_raw_combined.csv")
    copied_embedding = copy_if_exists(
        embedding_source, DOCS_DATA / "v2_embedding_cora_adam_seed42.json"
    )
    diagnostics_summary = V2_ROOT / "diagnostics" / "v2_optimizer_diagnostics_summary_clean.csv"
    diagnostics_gradient = V2_ROOT / "diagnostics" / "v2_gradient_summary_clean.csv"
    copied_diagnostics_summary = copy_if_exists(
        diagnostics_summary, DOCS_DATA / "v2_optimizer_diagnostics_summary.csv"
    )
    copied_diagnostics_gradient = copy_if_exists(
        diagnostics_gradient, DOCS_DATA / "v2_gradient_summary.csv"
    )
    copied_downloads = copy_downloads()
    completed, pending = count_pending_runs()
    primary_standard_planned = 10 * 5 * (1 + 3 * 4)
    standard_pending = max(0, primary_standard_planned - completed)
    row_count = 0
    if aggregate_source.exists():
        row_count = int(len(pd.read_csv(aggregate_source)))
    methodology = {
        "result_version": "V2",
        "default_view": "V2 when available, V1 legacy otherwise",
        "v1_label": "Legacy V1 - single seed fixed protocol",
        "v2_label": "V2 - reproducible random-perturbation protocols",
        "dataset_scope": ["Cora", "CiteSeer", "PubMed"],
        "primary_completed_runs": completed,
        "primary_standard_planned_runs": primary_standard_planned,
        "pending_runs_due_to_compute_limit": max(pending, standard_pending),
        "aggregate_rows_available": row_count,
        "has_v2_aggregate": copied_aggregate,
        "has_v2_raw": copied_raw,
        "has_v2_embedding": copied_embedding,
        "has_v2_diagnostics_summary": copied_diagnostics_summary,
        "has_v2_gradient_summary": copied_diagnostics_gradient,
        "download_artifacts": copied_downloads,
        "methodology_warning": (
            "Animated graph scenes are illustrative sampled graph views. "
            "They are not real-time GNN inference."
        ),
        "feature_masking_definition": (
            "Feature masking sets a requested fraction of active non-zero node-feature entries to zero."
        ),
        "gaussian_legacy_definition": (
            "Legacy V1 feature noise used Gaussian feature-noise standard deviation sigma."
        ),
    }
    (DOCS_DATA / "v2_methodology.json").write_text(
        json.dumps(methodology, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote V2 site data to {DOCS_DATA}")
    print(f"Wrote deployable downloads to {DOCS_DOWNLOADS}")


if __name__ == "__main__":
    main()
