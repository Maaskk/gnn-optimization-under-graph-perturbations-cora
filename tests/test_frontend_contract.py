from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_frontend_file(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_frontend_uses_light_red_visual_system():
    css = read_frontend_file("docs/assets/styles.css")

    assert "color-scheme: light" in css
    assert "--accent-red" in css
    assert "--paper" in css
    assert "#ffffff" in css.lower()


def test_frontend_has_mouse_reactive_layers():
    html = read_frontend_file("docs/index.html")
    js = read_frontend_file("docs/assets/app.js")

    assert "heroCanvas" in html
    assert "perturbationCanvas" in html
    assert "--mouse-x" in js
    assert "--mouse-y" in js
    assert "pointermove" in js


def test_frontend_explains_accuracy_benchmark_and_adds_visual_depth():
    html = read_frontend_file("docs/index.html")
    js = read_frontend_file("docs/assets/app.js")

    assert "Accuracy benchmark" in html
    assert "GCN reference" in html
    assert "dropMatrixChart" in html
    assert "drawDropMatrixChart" in js
