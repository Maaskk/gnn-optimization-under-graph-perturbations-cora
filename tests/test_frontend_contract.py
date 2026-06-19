from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_frontend_file(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_frontend_uses_light_burgundy_teal_visual_system():
    css = read_frontend_file("docs/assets/styles.css")

    assert "color-scheme: light" in css
    assert "--accent-red" in css
    assert "--accent-teal" in css
    assert "#8f1d2c" in css.lower()
    assert "#0f7c80" in css.lower()
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
    assert "applyTangleField" in js
    assert "drawTangleField" in js
    assert "quadraticCurveTo" in js


def test_frontend_explains_accuracy_benchmark_and_adds_visual_depth():
    html = read_frontend_file("docs/index.html")
    js = read_frontend_file("docs/assets/app.js")

    assert 'lang="fr"' in html
    assert "Benchmark de précision" in html
    assert "Référence GCN" in html
    assert "dropMatrixChart" in html
    assert "drawDropMatrixChart" in js


def test_frontend_uses_professional_graph_mark_not_letter_badge():
    html = read_frontend_file("docs/index.html")

    assert '<svg class="brand-mark"' in html
    assert '<span class="brand-mark">G</span>' not in html
