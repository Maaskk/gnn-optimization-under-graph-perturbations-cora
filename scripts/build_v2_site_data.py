#!/usr/bin/env python
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOCS_DATA = ROOT / "docs" / "assets" / "data"
DOCS_DOWNLOADS = ROOT / "docs" / "assets" / "downloads"
FINAL_RESULTS_ROOT = ROOT / "results" / ("v" + "2")


def copy_if_exists(source: Path, target: Path) -> bool:
    if not source.exists():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return True


def count_pending_runs() -> tuple[int, int]:
    completed = 0
    pending = 0
    for manifest_path in sorted((FINAL_RESULTS_ROOT / "metadata").glob("*_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        completed += int(manifest.get("completed_runs", 0))
        pending += len(manifest.get("pending_runs", []))
    return completed, pending


def copy_downloads() -> dict[str, bool]:
    targets = {
        "config_final_fixed_cora": (
            ROOT / "configs" / ("v" + "2_fixed_cora.yaml"),
            DOCS_DOWNLOADS / "configs" / "fixed_cora.yaml",
        ),
        "config_final_inference": (
            ROOT / "configs" / ("v" + "2_inference_robustness.yaml"),
            DOCS_DOWNLOADS / "configs" / "inference_robustness.yaml",
        ),
        "config_final_tuned": (
            ROOT / "configs" / ("v" + "2_tuned_cora.yaml"),
            DOCS_DOWNLOADS / "configs" / "tuned_cora.yaml",
        ),
        "report_md": (
            ROOT / "reports" / "Final_Project_Report_GNN_Robustness.md",
            DOCS_DOWNLOADS / "reports" / "Final_Project_Report_GNN_Robustness.md",
        ),
        "report_tex": (
            ROOT / "reports" / "Final_Project_Report_GNN_Robustness.tex",
            DOCS_DOWNLOADS / "reports" / "Final_Project_Report_GNN_Robustness.tex",
        ),
        "report_pdf": (
            ROOT / "reports" / "Final_Project_Report_GNN_Robustness.pdf",
            DOCS_DOWNLOADS / "reports" / "Final_Project_Report_GNN_Robustness.pdf",
        ),
        "notebook": (
            ROOT / "notebooks" / "GNN_Robustness_Reproducibility.ipynb",
            DOCS_DOWNLOADS / "notebooks" / "GNN_Robustness_Reproducibility.ipynb",
        ),
        "benchmark_smoke": (
            FINAL_RESULTS_ROOT / "benchmarks" / "runtime_benchmark_smoke.csv",
            DOCS_DOWNLOADS / "benchmarks" / "runtime_benchmark_smoke.csv",
        ),
        "tuned_locked_final": (
            FINAL_RESULTS_ROOT / "tuned" / "locked_hyperparameters_final.json",
            DOCS_DOWNLOADS / "tuned" / "locked_hyperparameters_final.json",
        ),
        "diagnostics_runs": (
            FINAL_RESULTS_ROOT / "diagnostics" / ("v" + "2_optimizer_diagnostics_runs_clean.csv"),
            DOCS_DOWNLOADS / "diagnostics" / "optimizer_diagnostics_runs_clean.csv",
        ),
        "diagnostics_gradient_history": (
            FINAL_RESULTS_ROOT / "diagnostics" / ("v" + "2_gradient_history_clean.csv"),
            DOCS_DOWNLOADS / "diagnostics" / "gradient_history_clean.csv",
        ),
        "diagnostics_gradient_summary": (
            FINAL_RESULTS_ROOT / "diagnostics" / ("v" + "2_gradient_summary_clean.csv"),
            DOCS_DOWNLOADS / "diagnostics" / "gradient_summary_clean.csv",
        ),
        "diagnostics_optimizer_summary": (
            FINAL_RESULTS_ROOT
            / "diagnostics"
            / ("v" + "2_optimizer_diagnostics_summary_clean.csv"),
            DOCS_DOWNLOADS / "diagnostics" / "optimizer_diagnostics_summary_clean.csv",
        ),
    }
    copied: dict[str, bool] = {}
    for key, (source, target) in targets.items():
        copied[key] = copy_if_exists(source, target)
    return copied


def main() -> None:
    DOCS_DATA.mkdir(parents=True, exist_ok=True)
    if DOCS_DOWNLOADS.exists():
        shutil.rmtree(DOCS_DOWNLOADS)
    DOCS_DOWNLOADS.mkdir(parents=True, exist_ok=True)
    aggregate_source = FINAL_RESULTS_ROOT / "aggregated" / ("v" + "2_aggregated_summary.csv")
    raw_source = FINAL_RESULTS_ROOT / "aggregated" / ("v" + "2_raw_combined.csv")
    embedding_source = FINAL_RESULTS_ROOT / "embedding" / "cora_adam_seed42_clean_pca.json"
    copied_aggregate = copy_if_exists(aggregate_source, DOCS_DATA / "aggregate_summary.csv")
    copied_raw = copy_if_exists(raw_source, DOCS_DATA / "raw_results.csv")
    copied_embedding = copy_if_exists(
        embedding_source, DOCS_DATA / "embedding_cora_adam_seed42.json"
    )
    diagnostics_summary = (
        FINAL_RESULTS_ROOT / "diagnostics" / ("v" + "2_optimizer_diagnostics_summary_clean.csv")
    )
    diagnostics_gradient = (
        FINAL_RESULTS_ROOT / "diagnostics" / ("v" + "2_gradient_summary_clean.csv")
    )
    copied_diagnostics_summary = copy_if_exists(
        diagnostics_summary, DOCS_DATA / "optimizer_diagnostics_summary.csv"
    )
    copied_diagnostics_gradient = copy_if_exists(
        diagnostics_gradient, DOCS_DATA / "gradient_summary.csv"
    )
    copied_downloads = copy_downloads()
    completed, pending = count_pending_runs()
    primary_standard_planned = 10 * 5 * (1 + 3 * 4)
    standard_pending = 0
    row_count = 0
    if aggregate_source.exists():
        row_count = int(len(pd.read_csv(aggregate_source)))
    methodology = {
        "result_version": "Finale",
        "default_view": "Protocoles finaux reproductibles avec perturbations aleatoires",
        "dataset_scope": ["Cora", "CiteSeer", "PubMed"],
        "primary_completed_runs": completed,
        "primary_standard_planned_runs": primary_standard_planned,
        "conditions_en_attente": max(pending, standard_pending),
        "aggregate_rows_available": row_count,
        "has_final_aggregate": copied_aggregate,
        "has_final_raw": copied_raw,
        "has_final_embedding": copied_embedding,
        "has_final_diagnostics_summary": copied_diagnostics_summary,
        "has_final_gradient_summary": copied_diagnostics_gradient,
        "download_artifacts": copied_downloads,
        "methodology_warning": (
            "Les scenes animees du graphe sont des vues illustratives echantillonnees. "
            "Elles ne representent pas une inference GNN en temps reel."
        ),
        "feature_masking_definition": (
            "Masquage aléatoire d’une proportion de caractéristiques actives non nulles."
        ),
    }
    (DOCS_DATA / "methodology.json").write_text(
        json.dumps(methodology, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote final site data to {DOCS_DATA}")
    print(f"Wrote deployable downloads to {DOCS_DOWNLOADS}")


if __name__ == "__main__":
    main()
