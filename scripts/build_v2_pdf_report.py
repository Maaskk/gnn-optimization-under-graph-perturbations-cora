#!/usr/bin/env python
from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PROJECT_TITLE = "Robustesse des GNN face aux perturbations du graphe"
PROJECT_SUBTITLE = "Projet 13 - Option 4 - Cora Citation Network"


def read_markdown_sections(path: Path) -> list[tuple[str, list[str]]]:
    sections: list[tuple[str, list[str]]] = []
    current_title = ""
    current_lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            continue
        if line.startswith("## "):
            if current_title:
                sections.append((current_title, current_lines))
            current_title = line.removeprefix("## ").strip()
            current_lines = []
        elif current_title:
            current_lines.append(line)
    if current_title:
        sections.append((current_title, current_lines))
    return sections


def clean_inline_markdown(text: str) -> str:
    escaped = html.escape(text.replace("**", ""), quote=False)
    return re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", escaped)


def clean_table_cell(text: str) -> str:
    return html.escape(text.replace("**", "").replace("`", ""), quote=False)


def markdown_lines_to_flowables(lines: list[str], styles) -> list:
    flowables: list = []
    bullet_buffer: list[str] = []
    table_buffer: list[list[str]] = []

    def flush_bullets() -> None:
        nonlocal bullet_buffer
        for item in bullet_buffer:
            flowables.append(Paragraph(f"- {clean_inline_markdown(item)}", styles["BulletBody"]))
        if bullet_buffer:
            flowables.append(Spacer(1, 0.12 * cm))
        bullet_buffer = []

    def flush_table() -> None:
        nonlocal table_buffer
        if not table_buffer:
            return
        filtered = [row for row in table_buffer if not all(set(cell) <= {"-", ":"} for cell in row)]
        if filtered:
            formatted = [
                [
                    Paragraph(cell, styles["TableHeader" if row_index == 0 else "TableCell"])
                    for cell in row
                ]
                for row_index, row in enumerate(filtered)
            ]
            table = Table(formatted, repeatRows=1)
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#8f1d2c")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d7dde3")),
                        (
                            "ROWBACKGROUNDS",
                            (0, 1),
                            (-1, -1),
                            [colors.white, colors.HexColor("#f7f8fa")],
                        ),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 5),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ]
                )
            )
            flowables.append(table)
            flowables.append(Spacer(1, 0.22 * cm))
        table_buffer = []

    in_code = False
    code_lines: list[str] = []
    for raw in lines:
        line = raw.rstrip()
        if line.startswith("```"):
            flush_bullets()
            flush_table()
            if in_code:
                flowables.append(Paragraph("<br/>".join(code_lines), styles["CodeBlock"]))
                flowables.append(Spacer(1, 0.18 * cm))
                code_lines = []
            in_code = not in_code
            continue
        if in_code:
            code_lines.append(html.escape(line))
            continue
        if not line.strip():
            flush_bullets()
            flush_table()
            continue
        if line.startswith("|"):
            row = [clean_table_cell(cell.strip()) for cell in line.strip("|").split("|")]
            table_buffer.append(row)
            continue
        if line.startswith("- "):
            flush_table()
            bullet_buffer.append(line.removeprefix("- ").strip())
            continue
        if line.startswith("### "):
            flush_bullets()
            flush_table()
            flowables.append(
                Paragraph(
                    clean_inline_markdown(line.removeprefix("### ").strip()), styles["Subheading"]
                )
            )
            continue
        flush_bullets()
        flush_table()
        flowables.append(Paragraph(clean_inline_markdown(line), styles["Body"]))
        flowables.append(Spacer(1, 0.12 * cm))
    flush_bullets()
    flush_table()
    return flowables


def page_decorator(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#617083"))
    canvas.drawString(2 * cm, 1.2 * cm, "Rapport V2 - perturbations aléatoires")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf(markdown_path: Path, output_path: Path) -> None:
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="CoverTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=26,
            leading=31,
            textColor=colors.HexColor("#14213d"),
            spaceAfter=20,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionHeading",
            parent=styles["Heading1"],
            fontSize=16,
            leading=20,
            textColor=colors.HexColor("#8f1d2c"),
            spaceBefore=12,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Body",
            parent=styles["BodyText"],
            fontSize=10.5,
            leading=15,
            textColor=colors.HexColor("#243043"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="Subheading",
            parent=styles["Heading2"],
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#14213d"),
            spaceBefore=8,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BulletBody",
            parent=styles["Body"],
            leftIndent=12,
            firstLineIndent=-8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableCell",
            parent=styles["Body"],
            fontSize=8,
            leading=10,
            textColor=colors.black,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableHeader",
            parent=styles["TableCell"],
            fontName="Helvetica-Bold",
            textColor=colors.white,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CodeBlock",
            parent=styles["Code"],
            fontName="Courier",
            fontSize=8.5,
            leading=11,
            backColor=colors.HexColor("#f7f8fa"),
            borderColor=colors.HexColor("#d7dde3"),
            borderWidth=0.4,
            borderPadding=6,
        )
    )

    sections = read_markdown_sections(markdown_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
        title=PROJECT_TITLE,
        author="Project 13 - Option 4",
    )

    story = [
        Spacer(1, 3.2 * cm),
        Paragraph(PROJECT_TITLE, styles["CoverTitle"]),
        Paragraph(PROJECT_SUBTITLE, styles["Heading2"]),
        Spacer(1, 1.2 * cm),
        Paragraph("Rapport scientifique V2", styles["Heading3"]),
        Paragraph(
            "Perturbations aléatoires, configurations reproductibles et provenance V1/V2 claire.",
            styles["Body"],
        ),
        PageBreak(),
        Paragraph("Table des matières", styles["SectionHeading"]),
    ]
    for title, _ in sections:
        story.append(Paragraph(clean_inline_markdown(title), styles["Body"]))
    story.append(PageBreak())

    for title, lines in sections:
        story.append(Paragraph(clean_inline_markdown(title), styles["SectionHeading"]))
        story.extend(markdown_lines_to_flowables(lines, styles))

    doc.build(story, onFirstPage=page_decorator, onLaterPages=page_decorator)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the V2 PDF report from Markdown.")
    parser.add_argument("--source", default="reports/final_report_v2.md")
    parser.add_argument("--output", default="reports/final_report_v2.pdf")
    args = parser.parse_args()
    build_pdf(Path(args.source), Path(args.output))
    print(f"Wrote PDF report: {args.output}")


if __name__ == "__main__":
    main()
