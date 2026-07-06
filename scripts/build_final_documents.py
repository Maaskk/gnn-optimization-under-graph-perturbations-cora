#!/usr/bin/env python
from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import qrcode
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "v2"
REPORTS = ROOT / "reports"
DELIVERABLES = ROOT / "deliverables"
FIGURES = REPORTS / "assets" / "final"
PROJECT_TITLE = "Robustesse des Réseaux de Neurones de Graphes sous Perturbations Aléatoires"
PROJECT_TITLE_EN = "Robustness of Graph Convolutional Networks under Random Graph Perturbations"
GITHUB_URL = "https://github.com/Maaskk/gnn-optimization-under-graph-perturbations-cora"
PAGES_URL = "https://maaskk.github.io/gnn-optimization-under-graph-perturbations-cora/"
OPTIMIZERS = ["Adam", "AdamW", "RMSProp", "AdaGrad", "SGD"]
PERTURBATION_LABELS = {
    "clean": "Graphe propre",
    "feature_masking": "Masquage attributs",
    "edge_removal": "Suppression d'aretes",
    "fake_edge_addition": "Ajout de fausses aretes",
}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def pct(value: float, digits: int = 1) -> str:
    if pd.isna(value):
        return "-"
    return f"{100 * float(value):.{digits}f}%"


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def current_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def manifest_counts() -> dict[str, int]:
    counts = {
        "core": 0,
        "cross_dataset": 0,
        "tuned": 0,
        "inference": 0,
        "pending": 0,
    }
    for path in sorted((RESULTS / "metadata").glob("*manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        completed = int(payload.get("completed_runs", 0))
        pending = len(payload.get("pending_runs", []))
        name = path.name
        if "fixed_cora" in name:
            counts["core"] += completed
        elif "cross_dataset" in name:
            counts["cross_dataset"] += completed
        elif "tuned_cora" in name:
            counts["tuned"] += completed
        elif "inference" in name:
            counts["inference"] += completed
        counts["pending"] += pending
    return counts


def core_aggregate(agg: pd.DataFrame) -> pd.DataFrame:
    if agg.empty:
        return agg
    return agg[
        (agg["dataset"] == "Cora")
        & (agg["protocol"] == "fixed")
        & (agg["robustness_setting"] == "training_time")
    ].copy()


def completed_subset(
    agg: pd.DataFrame, *, protocol: str | None = None, robustness_setting: str | None = None
) -> pd.DataFrame:
    if agg.empty:
        return agg
    subset = agg.copy()
    if protocol is not None:
        subset = subset[subset["protocol"] == protocol]
    if robustness_setting is not None:
        subset = subset[subset["robustness_setting"] == robustness_setting]
    return subset.copy()


def table_to_markdown(df: pd.DataFrame, columns: list[str], max_rows: int = 12) -> str:
    if df.empty:
        return "_Aucune sortie complete disponible pour cette section._"
    clipped = df.loc[:, columns].head(max_rows).copy()
    headers = [str(column) for column in clipped.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in clipped.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in clipped.columns) + " |")
    return "\n".join(lines)


def make_figures(agg: pd.DataFrame, raw: pd.DataFrame, diagnostics: pd.DataFrame) -> list[Path]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    made: list[Path] = []
    core = core_aggregate(agg)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    clean = core[core["perturbation_type"] == "clean"].sort_values(
        "mean_test_accuracy", ascending=False
    )
    if not clean.empty:
        fig, ax = plt.subplots(figsize=(7.4, 4.2))
        ax.bar(clean["optimizer"], clean["mean_test_accuracy"], color="#8f1d2c")
        ax.errorbar(
            clean["optimizer"],
            clean["mean_test_accuracy"],
            yerr=clean["ci95_test_accuracy_half_width"],
            fmt="none",
            ecolor="#14213d",
            capsize=4,
        )
        ax.set_ylim(0, max(0.85, clean["mean_test_accuracy"].max() + 0.05))
        ax.set_ylabel("Test accuracy")
        ax.set_title("Performance sur graphe propre avec IC95")
        path = FIGURES / "clean_accuracy_ci.png"
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        made.append(path)

    if not raw.empty:
        ranking = (
            raw[
                (raw["dataset"] == "Cora")
                & (raw["protocol"] == "fixed")
                & (raw["robustness_setting"] == "training_time")
            ]
            .groupby("optimizer")["test_accuracy"]
            .mean()
            .reindex(OPTIMIZERS)
            .dropna()
            .sort_values(ascending=False)
        )
        if not ranking.empty:
            fig, ax = plt.subplots(figsize=(7.4, 4.2))
            ax.bar(ranking.index, ranking.values, color="#0f7c80")
            ax.set_ylim(0, max(0.85, ranking.max() + 0.05))
            ax.set_ylabel("Accuracy moyenne")
            ax.set_title("Robustesse agregee sur les 13 conditions Cora")
            path = FIGURES / "aggregate_robustness_accuracy.png"
            fig.tight_layout()
            fig.savefig(path, dpi=180)
            plt.close(fig)
            made.append(path)

    for perturbation, filename, title in [
        (
            "feature_masking",
            "feature_masking_accuracy_ci.png",
            "Robustesse au masquage d'attributs",
        ),
        ("edge_removal", "edge_removal_accuracy_ci.png", "Robustesse a la suppression d'aretes"),
        ("fake_edge_addition", "fake_edge_accuracy_ci.png", "Robustesse aux fausses aretes"),
    ]:
        subset = core[core["perturbation_type"] == perturbation]
        if subset.empty:
            continue
        fig, ax = plt.subplots(figsize=(7.6, 4.4))
        for optimizer in OPTIMIZERS:
            rows = subset[subset["optimizer"] == optimizer].sort_values("requested_severity")
            if rows.empty:
                continue
            ax.errorbar(
                rows["requested_severity"] * 100,
                rows["mean_test_accuracy"],
                yerr=rows["ci95_test_accuracy_half_width"],
                marker="o",
                linewidth=1.8,
                capsize=3,
                label=optimizer,
            )
        ax.set_xlabel("Severite demandee (%)")
        ax.set_ylabel("Test accuracy")
        ax.set_ylim(0, 0.9)
        ax.set_title(title)
        ax.legend(ncol=3, fontsize=8)
        path = FIGURES / filename
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        made.append(path)

    if not diagnostics.empty:
        fig, ax = plt.subplots(figsize=(7.6, 4.4))
        for optimizer in OPTIMIZERS:
            rows = diagnostics[diagnostics["optimizer"] == optimizer].sort_values("epoch")
            if rows.empty:
                continue
            ax.plot(rows["epoch"], rows["mean_gradient_l2_norm"], label=optimizer, linewidth=1.7)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Norme L2 moyenne du gradient")
        ax.set_title("Diagnostics de gradients sur graphe propre")
        ax.legend(ncol=3, fontsize=8)
        path = FIGURES / "gradient_l2_norms.png"
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        made.append(path)
    return made


def make_qr_assets() -> None:
    DELIVERABLES.mkdir(parents=True, exist_ok=True)
    for name, url in [("qr_github.png", GITHUB_URL), ("qr_website.png", PAGES_URL)]:
        image = qrcode.make(url)
        image.save(DELIVERABLES / name)


def public_summary_tables(
    agg: pd.DataFrame, raw: pd.DataFrame, stats: pd.DataFrame, diagnostics: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    core = core_aggregate(agg)
    clean = core[core["perturbation_type"] == "clean"].copy()
    clean["Accuracy moyenne"] = clean["mean_test_accuracy"].map(pct)
    clean["IC95"] = clean["ci95_test_accuracy_half_width"].map(pct)
    clean["Macro F1"] = clean["mean_macro_f1"].map(pct)
    clean = clean.sort_values("mean_test_accuracy", ascending=False)

    if raw.empty:
        ranking = pd.DataFrame()
    else:
        rows = raw[
            (raw["dataset"] == "Cora")
            & (raw["protocol"] == "fixed")
            & (raw["robustness_setting"] == "training_time")
        ]
        ranking = rows.groupby("optimizer", as_index=False).agg(
            mean_accuracy=("test_accuracy", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_train_val_gap=("train_accuracy", lambda values: 0.0),
        )
        val_gap = (
            rows.assign(gap=rows["train_accuracy"] - rows["validation_accuracy"])
            .groupby("optimizer")["gap"]
            .mean()
        )
        ranking["Gap train-validation"] = ranking["optimizer"].map(val_gap).round(3)
        ranking["Accuracy moyenne"] = ranking["mean_accuracy"].map(pct)
        ranking["Macro F1 moyen"] = ranking["mean_macro_f1"].map(pct)
        ranking = ranking.sort_values("mean_accuracy", ascending=False)

    cross = completed_subset(agg, protocol="fixed", robustness_setting="training_time")
    cross = cross[cross["dataset"].isin(["Cora", "CiteSeer", "PubMed"])].copy()
    cross = cross[
        (cross["perturbation_type"] == "clean")
        | ((cross["requested_severity"] == 0.20) & (cross["perturbation_type"] != "clean"))
    ]
    cross["Accuracy"] = cross["mean_test_accuracy"].map(pct)
    cross["Macro F1"] = cross["mean_macro_f1"].map(pct)
    cross["Perturbation"] = cross["perturbation_type"].map(PERTURBATION_LABELS)
    cross["Severite"] = cross["requested_severity"].map(pct)
    cross["Dataset"] = cross["dataset"]
    cross["Optimiseur"] = cross["optimizer"]

    tuned = completed_subset(agg, protocol="tuned", robustness_setting="training_time").copy()
    if not tuned.empty:
        tuned["Accuracy"] = tuned["mean_test_accuracy"].map(pct)
        tuned["Perturbation"] = tuned["perturbation_type"].map(PERTURBATION_LABELS)
        tuned["Severite"] = tuned["requested_severity"].map(pct)
        tuned["Dataset"] = tuned["dataset"]
        tuned["Optimiseur"] = tuned["optimizer"]

    inference = completed_subset(agg, robustness_setting="inference_time").copy()
    if not inference.empty:
        inference["Accuracy"] = inference["mean_test_accuracy"].map(pct)
        inference["Perturbation"] = inference["perturbation_type"].map(PERTURBATION_LABELS)
        inference["Severite"] = inference["requested_severity"].map(pct)
        inference["Dataset"] = inference["dataset"]
        inference["Optimiseur"] = inference["optimizer"]

    stats_fr = stats.copy()
    if not stats_fr.empty:
        stats_fr["Metrique"] = stats_fr["Metric"].replace(
            {"test_accuracy": "Accuracy test", "macro_f1": "Macro F1"}
        )
        stats_fr["Interpretation FR"] = stats_fr["Interpretation"].map(
            lambda text: "Preuve insuffisante pour distinguer clairement les deux optimiseurs."
            if "insufficient" in str(text).lower()
            else "Moyenne plus elevee sous ce protocole, avec une interpretation statistique prudente."
        )
        stats_fr = stats_fr.rename(
            columns={
                "Optimizer A": "Optimiseur A",
                "Optimizer B": "Optimiseur B",
                "Mean difference": "Difference moyenne",
                "Adjusted p-value": "p ajustee",
            }
        )

    diag = diagnostics.groupby("optimizer", as_index=False).agg(
        Gradient_moyen=("mean_gradient_l2_norm", "mean"),
        Perte_validation_moyenne=("mean_validation_loss", "mean"),
    )
    if not diag.empty:
        diag["Gradient moyen"] = diag["Gradient_moyen"].map(lambda value: f"{value:.4f}")
        diag["Perte validation moyenne"] = diag["Perte_validation_moyenne"].map(
            lambda value: f"{value:.4f}"
        )

    return {
        "clean": clean,
        "ranking": ranking,
        "cross": cross,
        "tuned": tuned,
        "inference": inference,
        "stats": stats_fr,
        "diagnostics": diag,
    }


def build_report_markdown(
    agg: pd.DataFrame,
    raw: pd.DataFrame,
    stats: pd.DataFrame,
    diagnostics: pd.DataFrame,
    figures: list[Path],
) -> str:
    counts = manifest_counts()
    tables = public_summary_tables(agg, raw, stats, diagnostics)
    best = "-"
    if not tables["ranking"].empty:
        row = tables["ranking"].iloc[0]
        best = f"{row['optimizer']} ({row['Accuracy moyenne']})"
    figure_lines = "\n".join(
        f"![{path.stem}](assets/final/{path.name})" for path in figures if path.exists()
    )
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    return f"""# {PROJECT_TITLE}

**Titre anglais:** {PROJECT_TITLE_EN}

**Projet 13 - Option 4.** Dataset Cora Citation Network. Modele GCN a deux couches. Optimiseurs: Adam, AdamW, RMSProp, AdaGrad et SGD.

**Generation:** {generated}.

## Résumé

Ce rapport etudie comment le choix de l'optimiseur influence la stabilite d'un GCN a deux couches lorsque le graphe Cora subit des perturbations aleatoires. Le coeur de l'etude contient {counts["core"]} entrainements reels: 5 optimiseurs x 13 conditions x 10 graines. Les resultats sont rapportes avec moyenne, ecart-type et intervalle de confiance 95%. Les conclusions restent limitees au dataset, a l'architecture, aux hyperparametres et aux perturbations aleatoires etudies.

## Introduction

Les reseaux de neurones de graphes propagent l'information le long des aretes. Une suppression d'aretes, l'ajout de fausses aretes ou le masquage des attributs actifs peut donc modifier directement le signal disponible. L'objectif est de comparer les optimiseurs sous un protocole fixe et reproductible, pas de chercher le meilleur optimiseur universel.

## Question de recherche

Pour un GCN a deux couches sur les reseaux de citations Planetoid, comment Adam, AdamW, RMSProp, AdaGrad et SGD se comparent-ils lorsque des perturbations aleatoires sont appliquees au graphe ou aux caracteristiques?

## Travaux connexes

Kipf et Welling ont introduit le Graph Convolutional Network utilise ici comme reference architecturale. Kingma et Ba ont propose Adam, Loshchilov et Hutter ont separe la regularisation dans AdamW, et les optimiseurs RMSProp, AdaGrad et SGD representent des familles classiques d'optimisation stochastique. Cette etude reste experimentale et controlee.

## Jeu de données et architecture GCN

Cora contient 2708 documents, 1433 attributs binaires, 7 classes et 10556 aretes dirigees dans la representation Planetoid chargee par PyTorch Geometric. Le modele utilise deux couches GCNConv, 16 canaux caches, ReLU, dropout 0.5 et une perte cross-entropy sur le masque d'entrainement.

## Optimiseurs

Les cinq optimiseurs sont executes avec le meme budget de 200 epoques dans le protocole fixe. Le protocole tune utilise uniquement la validation pour choisir les hyperparametres, puis verrouille ces choix avant l'evaluation test.

## Protocole expérimental

- Graines principales: 42 a 51.
- 200 epoques par entrainement.
- Conditions Cora principales: graphe propre, masquage d'attributs 5%, 10%, 20%, 30%, edge removal 5%, 10%, 20%, 30%, fake edge addition 5%, 10%, 20%, 30%.
- Matrice principale: 5 optimiseurs x 13 conditions x 10 graines = {counts["core"]} runs reels.
- Cross-dataset complete: {counts["cross_dataset"]} runs reels.
- Evaluation tunee complete: {counts["tuned"]} runs reels.
- Robustesse a l'inference complete: {counts["inference"]} runs reels.

## Définitions des perturbations aléatoires

Masquage aléatoire d’une proportion de caractéristiques actives non nulles. Edge removal supprime une fraction de connexions non orientees uniques et conserve une representation symetrique. Fake edge addition ajoute uniquement des paires de noeuds non connectees auparavant, sans self-loops ni duplicats.

## Méthodologie statistique

Les tableaux agreges donnent moyenne, ecart-type et IC95. Les comparaisons entre optimiseurs utilisent des graines appariees, un intervalle bootstrap apparie, un test de Wilcoxon quand il est valide, puis une correction de Holm pour comparaisons multiples.

## Résultats principaux sur Cora

Meilleur score moyen agrege sous ce protocole: **{best}**. Cette phrase ne signifie pas que l'optimiseur est universellement meilleur.

{table_to_markdown(tables["clean"], ["optimizer", "Accuracy moyenne", "IC95", "Macro F1", "n_seeds"])}

## Classement global de robustesse

{table_to_markdown(tables["ranking"], ["optimizer", "Accuracy moyenne", "Macro F1 moyen", "Gap train-validation"])}

## Analyse statistique

{table_to_markdown(tables["stats"], ["Optimiseur A", "Optimiseur B", "Metrique", "Difference moyenne", "CI95", "p ajustee", "Interpretation FR"], max_rows=20)}

## Résultats multi-datasets

{table_to_markdown(tables["cross"], ["Dataset", "Optimiseur", "Perturbation", "Severite", "Accuracy", "Macro F1", "n_seeds"], max_rows=18)}

## Résultats du protocole tuné

{table_to_markdown(tables["tuned"], ["Dataset", "Optimiseur", "Perturbation", "Severite", "Accuracy", "n_seeds"], max_rows=18)}

## Robustesse à l'inférence

{table_to_markdown(tables["inference"], ["Dataset", "Optimiseur", "Perturbation", "Severite", "Accuracy", "n_seeds"], max_rows=18)}

## Diagnostics gradients et ressources

{table_to_markdown(tables["diagnostics"], ["optimizer", "Gradient moyen", "Perte validation moyenne"])}

Les diagnostics propres comportent 5 optimiseurs x 10 graines x 200 epoques = 10000 lignes de normes de gradients. La consommation memoire est enregistree dans les sorties de diagnostics par optimiseur et par graine.

## Visualisations

{figure_lines}

## Discussion

Adam et RMSProp obtiennent des moyennes elevees sous le protocole fixe Cora. Les differences entre optimiseurs adaptatifs proches doivent etre lues avec les intervalles et les tests appariees. SGD sous-performe dans ce protocole fixe, notamment parce que le taux d'apprentissage et le momentum ne sont pas optimises pour lui dans la comparaison principale.

## Limites

Les perturbations sont aleatoires et non des attaques optimisees. Le benchmark reste controle sur des datasets de citation classiques. Les temps CPU dependent du materiel local. Les conclusions ne doivent pas etre generalisees a tous les GNN, tous les graphes ou tous les regimes d'hyperparametres.

## Reproductibilité

Commandes principales:

```bash
make setup
make test
make lint
make format-check
make smoke
make experiment-cora
make experiment-cross-dataset
make experiment-tuned
make experiment-inference
make aggregate
make build-site
make reproduce-final
```

Les resultats bruts, agregats, diagnostics, configurations, notebook, rapport et site sont stockes dans le depot. L'identifiant de version est conserve dans les lignes brutes et dans les metadonnees d'environnement.

## Références

- Kipf, T. N., & Welling, M. Semi-Supervised Classification with Graph Convolutional Networks.
- Kingma, D. P., & Ba, J. Adam: A Method for Stochastic Optimization.
- Loshchilov, I., & Hutter, F. Decoupled Weight Decay Regularization.
- Duchi, J., Hazan, E., & Singer, Y. Adaptive Subgradient Methods for Online Learning and Stochastic Optimization.
- PyTorch Geometric documentation for Planetoid datasets and GCNConv.

## Annexe

Les definitions exactes de graines resolues, taux demandes, taux effectivement appliques, identifiants de version et metadonnees materiel sont dans les CSV bruts et les JSON de metadonnees generes par le pipeline.
"""


def markdown_to_tex(markdown: str) -> str:
    body = html.escape(markdown)
    body = body.replace("&amp;", "&")
    body = re.sub(r"^# (.+)$", r"\\section*{\1}", body, flags=re.MULTILINE)
    body = re.sub(r"^## (.+)$", r"\\section{\1}", body, flags=re.MULTILINE)
    body = re.sub(r"^- (.+)$", r"\\noindent-- \1\\\\", body, flags=re.MULTILINE)
    body = body.replace("\n\n", "\n\\par\n")
    return (
        "\\documentclass[11pt,a4paper]{article}\n\\usepackage[margin=2cm]{geometry}\n\\usepackage{graphicx}\n\\usepackage{hyperref}\n\\begin{document}\n\\tableofcontents\n\\newpage\n"
        + body
        + "\n\\end{document}\n"
    )


def clean_inline_markdown(text: str) -> str:
    escaped = html.escape(text.replace("**", ""), quote=False)
    escaped = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", escaped)
    return escaped


def markdown_headings(markdown: str) -> list[str]:
    return [line[3:].strip() for line in markdown.splitlines() if line.startswith("## ")]


def markdown_to_flowables(markdown: str, styles, *, skip_h1: bool = False) -> list:
    flowables: list = []
    table_buffer: list[list[str]] = []

    def flush_table() -> None:
        nonlocal table_buffer
        if not table_buffer:
            return
        rows = [
            row for row in table_buffer if not all(set(cell.strip()) <= {"-", ":"} for cell in row)
        ]
        if rows:
            table = Table(
                [
                    [
                        Paragraph(
                            clean_inline_markdown(cell),
                            styles["TableHeader" if i == 0 else "TableCell"],
                        )
                        for cell in row
                    ]
                    for i, row in enumerate(rows)
                ],
                repeatRows=1,
            )
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#8f1d2c")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d7dde3")),
                        ("FONTSIZE", (0, 0), (-1, -1), 7.2),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ]
                )
            )
            flowables.append(table)
            flowables.append(Spacer(1, 0.18 * cm))
        table_buffer = []

    in_code = False
    code_lines: list[str] = []
    for raw in markdown.splitlines():
        line = raw.rstrip()
        if line.startswith("```"):
            flush_table()
            if in_code:
                flowables.append(Paragraph("<br/>".join(code_lines), styles["Code"]))
                code_lines = []
            in_code = not in_code
            continue
        if in_code:
            code_lines.append(html.escape(line))
            continue
        if line.startswith("# "):
            flush_table()
            if skip_h1:
                continue
            flowables.append(Paragraph(clean_inline_markdown(line[2:]), styles["Title"]))
            continue
        if line.startswith("## "):
            flush_table()
            flowables.append(Paragraph(clean_inline_markdown(line[3:]), styles["Heading"]))
            continue
        if line.startswith("|"):
            table_buffer.append([cell.strip() for cell in line.strip("|").split("|")])
            continue
        if line.startswith("!["):
            flush_table()
            image_match = re.search(r"\(([^)]+)\)", line)
            if image_match:
                path = REPORTS / image_match.group(1)
                if path.exists():
                    flowables.append(
                        Image(str(path), width=15.5 * cm, height=8.5 * cm, kind="proportional")
                    )
                    flowables.append(Spacer(1, 0.18 * cm))
            continue
        if not line.strip():
            flush_table()
            flowables.append(Spacer(1, 0.08 * cm))
            continue
        if line.startswith("- "):
            flush_table()
            flowables.append(Paragraph("- " + clean_inline_markdown(line[2:]), styles["Body"]))
            continue
        flush_table()
        flowables.append(Paragraph(clean_inline_markdown(line), styles["Body"]))
    flush_table()
    return flowables


def build_pdf(
    markdown: str,
    output: Path,
    *,
    title: str,
    cover: bool = False,
) -> None:
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="CoverTitle",
            parent=styles["Title"],
            fontSize=24,
            leading=29,
            alignment=1,
            textColor=colors.HexColor("#14213d"),
            spaceAfter=18,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CoverSubtitle",
            parent=styles["BodyText"],
            fontSize=12,
            leading=16,
            alignment=1,
            textColor=colors.HexColor("#243043"),
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TocTitle",
            parent=styles["Heading1"],
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#8f1d2c"),
            spaceAfter=14,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Heading",
            parent=styles["Heading1"],
            fontSize=15,
            leading=18,
            textColor=colors.HexColor("#8f1d2c"),
        )
    )
    styles.add(ParagraphStyle(name="Body", parent=styles["BodyText"], fontSize=9.5, leading=13))
    styles.add(ParagraphStyle(name="TableCell", parent=styles["Body"], fontSize=7.2, leading=8.5))
    styles.add(
        ParagraphStyle(name="TableHeader", parent=styles["TableCell"], textColor=colors.white)
    )
    doc = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=1.7 * cm,
        rightMargin=1.7 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
        title=title,
    )
    story = []
    if cover:
        story.extend(
            [
                Spacer(1, 2.8 * cm),
                Paragraph(PROJECT_TITLE, styles["CoverTitle"]),
                Paragraph(PROJECT_TITLE_EN, styles["CoverSubtitle"]),
                Spacer(1, 0.6 * cm),
                Paragraph("Projet 13 - Option 4", styles["CoverSubtitle"]),
                Paragraph("Dataset: Cora Citation Network", styles["CoverSubtitle"]),
                Paragraph(
                    "Adam vs AdamW vs RMSProp vs AdaGrad vs SGD",
                    styles["CoverSubtitle"],
                ),
                Spacer(1, 1.0 * cm),
                Paragraph(
                    "Rapport scientifique final - perturbations aleatoires du graphe",
                    styles["CoverSubtitle"],
                ),
                Paragraph(
                    "Équipe: Ossama Ashad, Mouhcine Ayar, Mohamed Amine Kar-any, "
                    "Hamza Elhaddaji, Iliass Ouchida",
                    styles["CoverSubtitle"],
                ),
                Paragraph(
                    f"Generation: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
                    styles["CoverSubtitle"],
                ),
                PageBreak(),
                Paragraph("Table des matieres", styles["TocTitle"]),
            ]
        )
        for index, heading in enumerate(markdown_headings(markdown), start=1):
            story.append(Paragraph(f"{index}. {clean_inline_markdown(heading)}", styles["Body"]))
            story.append(Spacer(1, 0.05 * cm))
        story.append(PageBreak())
    story.extend(markdown_to_flowables(markdown, styles, skip_h1=cover))
    output.parent.mkdir(parents=True, exist_ok=True)
    doc.build(story)


def defense_script_markdown() -> str:
    return """# Script de soutenance - GNN Cora

## 1. Mohamed Amine Kar-any - Slides 1-2 - environ 2 minutes

Bonjour. Aujourd'hui, nous presentons notre etude sur la robustesse des GNN sous perturbations aleatoires. La question est simple: si le graphe Cora perd des aretes, gagne de fausses aretes, ou perd une partie de ses attributs actifs, quel optimiseur garde l'entrainement le plus stable?

Transition: Hamza va maintenant presenter le dataset, le modele et les optimiseurs compares.

Mots a accentuer: perturbations aleatoires; comparaison controlee; Cora.

A ne pas dire: robustesse adversariale; meilleur optimiseur universel.

## 2. Hamza Elhaddaji - Slides 3-4 - environ 2.5 minutes

Cora est un reseau de citations: les noeuds sont des documents, les aretes representent des citations, et les attributs de noeuds sont des caracteristiques textuelles binaires. Nous utilisons un GCN a deux couches avec 16 canaux caches, dropout 0.5 et une perte cross-entropy.

Nous comparons Adam, AdamW, RMSProp, AdaGrad et SGD. Le but n'est pas de favoriser un optimiseur, mais de les placer dans le meme cadre experimental.

Transition: Ossama va detailler le protocole scientifique.

Mots a accentuer: meme architecture; meme dataset; cinq optimiseurs.

A ne pas dire: SGD est mauvais en general.

## 3. Ossama Ashad - Slides 5-7 - environ 3 minutes

Nous avons utilise 200 epoques par entrainement. Le budget fixe de 200 epoques assure une comparaison controlee. Le coeur de l'etude comporte 650 entrainements reels: 5 optimiseurs x 13 conditions x 10 graines.

Les graines permettent d'obtenir moyenne, ecart-type et IC95%. Le masquage d'attributs suit la definition: masquage aleatoire d'une proportion de caracteristiques actives non nulles. Pour la structure, nous supprimons des aretes ou nous ajoutons de fausses aretes sans self-loops et sans duplicats. Il s'agit de perturbations aleatoires, pas d'attaques adversariales.

Les conclusions sont limitees au dataset, a l'architecture et au protocole etudies.

Transition: Iliass va maintenant interpreter les resultats principaux.

Mots a accentuer: 650 entrainements reels; 200 epoques; IC95%; perturbations aleatoires.

A ne pas dire: resultats inventes; test utilise pour choisir les hyperparametres.

## 4. Iliass Ouchida - Slides 8-11 - environ 3 minutes

Sur le graphe propre, les optimiseurs adaptatifs obtiennent les meilleurs scores moyens. Sur les perturbations, Adam et RMSProp restent proches dans plusieurs conditions. Quand les intervalles se recouvrent, nous evitons de parler de dominance.

Le masquage d'attributs teste la perte d'information dans les attributs. Les perturbations structurelles testent la sensibilite aux connexions du graphe. SGD sous-performe ici dans le protocole fixe, mais cela ne veut pas dire que SGD est faible dans tous les contextes.

Transition: Mouhcine va presenter les validations complementaires, les limites et la demonstration.

Mots a accentuer: intervalles; prudence; protocole fixe.

A ne pas dire: Adam domine globalement.

## 5. Mouhcine Ayar - Slides 12-14 - environ 3 minutes

Nous avons ajoute trois validations: cross-dataset, protocole tune avec selection validation uniquement, et perturbation a l'inference apres entrainement propre. Ces validations aident a separer ce qui depend de Cora, de l'optimiseur, et du moment ou la perturbation est appliquee.

Les limites sont importantes: ce ne sont pas des attaques adversariales, les temps CPU dependent du materiel, et les conclusions restent liees au GCN a deux couches.

Je termine avec la demonstration du dashboard. Le site montre les resultats, les fichiers bruts, les diagnostics et les artefacts reproductibles.

Mots a accentuer: validations complementaires; limites honnetes; dashboard.

A ne pas dire: preuve universelle; inference live dans l'animation.
"""


def jury_qa_markdown() -> str:
    pairs = [
        (
            "Pourquoi 200 epoques ?",
            "Pour donner le meme budget d'entrainement a chaque optimiseur et eviter qu'un optimiseur profite d'un temps different.",
        ),
        (
            "Pourquoi 10 graines ?",
            "Pour estimer la variabilite aleatoire et calculer moyenne, ecart-type et IC95.",
        ),
        (
            "Pourquoi 650 runs ?",
            "Parce que le coeur de l'etude combine 5 optimiseurs, 13 conditions et 10 graines.",
        ),
        (
            "Pourquoi Cora ?",
            "Cora est un benchmark classique de classification de noeuds, assez petit pour une matrice reproductible complete.",
        ),
        (
            "Pourquoi un GCN a deux couches ?",
            "C'est l'architecture de reference la plus simple pour isoler l'effet de l'optimiseur.",
        ),
        (
            "Pourquoi ces optimiseurs ?",
            "Ils couvrent Adam, AdamW, RMSProp, AdaGrad et SGD, donc des familles adaptatives et non adaptatives.",
        ),
        (
            "Pourquoi SGD est faible dans ce protocole ?",
            "Sous ce taux d'apprentissage fixe et sans tuning principal, SGD converge moins bien. Ce n'est pas une critique generale de SGD.",
        ),
        (
            "Pourquoi le masquage d'attributs ?",
            "Masquage aléatoire d’une proportion de caractéristiques actives non nulles. Cette definition correspond au protocole final et garde les valeurs deja nulles intactes.",
        ),
        (
            "Est-ce de la robustesse adversariale ?",
            "Non. Les perturbations sont aleatoires, pas optimisees contre le modele.",
        ),
        (
            "Pourquoi suppression et fausses aretes ?",
            "Elles representent deux erreurs structurelles opposees: manque de citations et connexions artificielles.",
        ),
        (
            "Perturbation a l'entrainement ou a l'inference ?",
            "A l'entrainement, le modele apprend sur les donnees perturbees. A l'inference, il apprend proprement puis on perturbe seulement l'evaluation.",
        ),
        (
            "Pourquoi les resultats changent selon les datasets ?",
            "La densite, les attributs et les classes different, donc le signal de propagation change.",
        ),
        (
            "Comment garantir la reproductibilite ?",
            "Graines fixes, configs versionnees, identifiant de version, metadonnees materiel, resultats bruts et tests automatises.",
        ),
        (
            "Ou sont les resultats bruts ?",
            "Dans le depot, separes des agregats, avec une ligne par entrainement reel.",
        ),
        (
            "Pourquoi ne pas dire qu'Adam est universellement meilleur ?",
            "Parce que l'etude teste un dataset, une architecture et un protocole precis.",
        ),
        (
            "Pourquoi les intervalles de confiance ?",
            "Ils montrent l'incertitude due aux graines et evitent de surinterpreter de petites differences.",
        ),
        (
            "Le projet a-t-il ete entraine sur Google Colab ?",
            "Les experiences principales ont ete lancees localement sur CPU; la reproductibilite compte plus que la plateforme cloud.",
        ),
        (
            "Prochaines etapes concretes ?",
            "Tester d'autres architectures, ajouter des attaques adversariales reelles, et elargir les datasets.",
        ),
    ]
    lines = ["# Questions - Reponses Jury GNN", ""]
    for question, answer in pairs:
        lines.append(f"## {question}")
        lines.append("")
        lines.append(answer)
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build final academic report and defense documents."
    )
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()
    agg = read_csv(RESULTS / "aggregated" / "v2_aggregated_summary.csv")
    raw = read_csv(RESULTS / "aggregated" / "v2_raw_combined.csv")
    stats = read_csv(RESULTS / "statistics" / "optimizer_paired_comparisons.csv")
    diagnostics = read_csv(RESULTS / "diagnostics" / "v2_gradient_summary_clean.csv")

    figures = [] if args.skip_figures else make_figures(agg, raw, diagnostics)
    make_qr_assets()
    report_md = build_report_markdown(agg, raw, stats, diagnostics, figures)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "Final_Project_Report_GNN_Robustness.md").write_text(report_md, encoding="utf-8")
    (REPORTS / "Final_Project_Report_GNN_Robustness.tex").write_text(
        markdown_to_tex(report_md), encoding="utf-8"
    )
    build_pdf(
        report_md,
        REPORTS / "Final_Project_Report_GNN_Robustness.pdf",
        title=PROJECT_TITLE,
        cover=True,
    )

    DELIVERABLES.mkdir(parents=True, exist_ok=True)
    script_md = defense_script_markdown()
    qa_md = jury_qa_markdown()
    (DELIVERABLES / "Script_Soutenance_GNN_FR.md").write_text(script_md, encoding="utf-8")
    (DELIVERABLES / "Questions_Reponses_Jury_GNN_FR.md").write_text(qa_md, encoding="utf-8")
    build_pdf(script_md, DELIVERABLES / "Script_Soutenance_GNN_FR.pdf", title="Script Soutenance")
    build_pdf(
        qa_md,
        DELIVERABLES / "Questions_Reponses_Jury_GNN_FR.pdf",
        title="Questions Reponses Jury",
    )
    print("Wrote final report and defense documents.")


if __name__ == "__main__":
    main()
