from pathlib import Path


def production_csvs():
    return [
        path
        for path in Path("results").rglob("*.csv")
        if "ci_smoke" not in path.parts
    ]


def test_final_results_are_present():
    files = production_csvs()
    assert files, "No production CSV results found under results/"

    assert any(
        "raw" in path.parts or "raw" in path.name.lower()
        for path in files
    ), "No raw-result CSV found"

    assert any(
        "aggregated" in path.parts
        or "aggregate" in path.name.lower()
        or "summary" in path.name.lower()
        for path in files
    ), "No aggregated-result CSV found"


def test_final_report_exists():
    assert Path("reports/Final_Project_Report_GNN_Robustness.md").exists()
    assert Path("reports/Final_Project_Report_GNN_Robustness.pdf").exists()


def test_public_dashboard_assets_exist():
    docs = Path("docs")
    data_dir = docs / "assets" / "data"

    assert (docs / "index.html").exists()
    assert (docs / "assets" / "styles.css").exists()
    assert (docs / "assets" / "app.js").exists()
    assert data_dir.exists()

    data_csvs = list(data_dir.glob("*.csv"))
    data_jsons = list(data_dir.glob("*.json"))

    assert data_csvs, "Dashboard has no CSV data files"
    assert data_jsons, "Dashboard has no JSON data files"


def test_public_asset_names_are_neutral():
    data_dir = Path("docs") / "assets" / "data"
    forbidden = ("legacy", "v1_", "v2_", "old_", "single_seed")

    for path in data_dir.iterdir():
        assert not any(token in path.name.lower() for token in forbidden), (
            f"Non-neutral public asset name: {path.name}"
        )
