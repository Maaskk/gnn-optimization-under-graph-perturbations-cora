import json
from pathlib import Path


def test_v2_required_project_artifacts_exist():
    required = [
        "configs/v1_legacy_fixed_cora.yaml",
        "configs/v2_fixed_cora.yaml",
        "configs/v2_tuned_cora.yaml",
        "configs/v2_cross_dataset.yaml",
        "configs/v2_inference_robustness.yaml",
        "scripts/run_tuned_v2_evaluation.py",
        "scripts/compare_v2_statistics.py",
        "scripts/build_final_documents.py",
        "reports/Final_Project_Report_GNN_Robustness.md",
        "reports/Final_Project_Report_GNN_Robustness.tex",
        "reports/Final_Project_Report_GNN_Robustness.pdf",
        "notebooks/GNN_Robustness_Reproducibility.ipynb",
        "Makefile",
        "docs/ci/github-actions-ci.yml",
        "LICENSE",
        "CITATION.cff",
        "CHANGELOG.md",
        "docs/REPRODUCIBILITY.md",
        "scripts/benchmark_v2_runtime.py",
        "scripts/collect_v2_diagnostics.py",
        "scripts/tune_v2_hyperparameters.py",
        "scripts/build_v2_pdf_report.py",
        "src/gnn_robustness/v2_statistics.py",
    ]

    for path in required:
        assert Path(path).exists(), path


def test_v2_notebook_contains_required_reproducibility_sections():
    notebook = json.loads(Path("notebooks/GNN_Robustness_Reproducibility.ipynb").read_text())
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "environment" in text.lower()
    assert "Dataset Loading" in text
    assert "GCN Architecture" in text
    assert "Perturbation Demonstration" in text
    assert "aggregate_raw_results" in text
    assert "Confidence Interval" in text
    assert "PCA" in text
    assert "model.encode" in text


def test_v2_docs_do_not_overstate_legacy_results():
    readme = Path("README.md").read_text(encoding="utf-8")
    report = Path("reports/Final_Project_Report_GNN_Robustness.md").read_text(encoding="utf-8")

    forbidden = ["universally best", "proves adversarial robustness", "dominates globally"]
    for phrase in forbidden:
        assert phrase not in readme.lower()
        assert phrase not in report.lower()

    assert "perturbations aléatoires" in readme.lower()
    assert "650 runs réels" in readme


def test_makefile_exposes_benchmark_and_tuning_entry_points():
    makefile = Path("Makefile").read_text(encoding="utf-8")

    for target in [
        "make setup",
        "test:",
        "lint:",
        "format-check:",
        "smoke:",
        "experiment-cora:",
        "experiment-cross-dataset:",
        "experiment-tuned:",
        "experiment-inference:",
        "aggregate:",
        "build-site:",
        "reproduce-final:",
    ]:
        assert target.replace("make ", "") in makefile
