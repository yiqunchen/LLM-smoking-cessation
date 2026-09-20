#!/usr/bin/env python3
"""Create a three-domain Word-ready Reviewer 3 ablation-results document."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "revision" / "figures" / "reviewer_ablations_dt10"
OUTPUT = ROOT / "revision" / "Reviewer_3_prompt_ablation_bootstrap_results.docx"
MODEL_ORDER = ["GPT-4o-mini", "GPT-5", "DeepSeek-R1", "Grok-4.3", "Gemini-2.5-Pro"]
CONDITION_ORDER = [
    "PP + history + CBT/ACT",
    "PP + history (no CBT/ACT)",
    "History + ratings only",
    "History text only",
]
DOMAIN_ORDER = ["Content", "Coping", "Quitting"]
STANDARD_METRICS = ["accuracy", "macro_f1", "qwk"]
DIRECTIONAL_METRICS = ["directional_accuracy", "directional_macro_f1"]


def shade(cell, fill: str) -> None:
    properties = cell._tc.get_or_add_tcPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), fill)
    properties.append(shading)


def set_cell_text(cell, text: str, bold: bool = False, color: str | None = None) -> None:
    cell.text = ""
    run = cell.paragraphs[0].add_run(text)
    run.bold = bold
    run.font.size = Pt(7.8)
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def add_table(doc: Document, columns: list[str], rows: list[list[str]]) -> None:
    table = doc.add_table(rows=1, cols=len(columns))
    table.style = "Table Grid"
    for cell, title in zip(table.rows[0].cells, columns):
        set_cell_text(cell, title, bold=True, color="FFFFFF")
        shade(cell, "1F4E79")
    for row in rows:
        cells = table.add_row().cells
        for cell, value in zip(cells, row):
            set_cell_text(cell, value)
    doc.add_paragraph()


def format_ci(values: pd.Series, value: str) -> str:
    if value == "difference":
        return f"{values[value]:+.3f} [{values['ci_low']:+.3f}, {values['ci_high']:+.3f}]"
    return f"{values[value]:.3f} [{values['ci_low']:.3f}, {values['ci_high']:.3f}]"


def add_caption(doc: Document, text: str) -> None:
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run(text)
    run.italic = True
    run.font.size = Pt(9)


def make_table_rows(frame: pd.DataFrame, index: list[str], metrics: list[str], value: str) -> list[list[str]]:
    pivot = frame.pivot(index=index, columns="metric", values=[value, "ci_low", "ci_high"])
    pivot = pivot.reindex(columns=pd.MultiIndex.from_product([[value, "ci_low", "ci_high"], metrics]))
    rows: list[list[str]] = []
    for keys, values in pivot.iterrows():
        cells = [str(key) for key in (keys if isinstance(keys, tuple) else (keys,))]
        for metric in metrics:
            row = pd.Series({
                value: values[(value, metric)],
                "ci_low": values[("ci_low", metric)],
                "ci_high": values[("ci_high", metric)],
            })
            cells.append(format_ci(row, value))
        rows.append(cells)
    return rows


def main() -> None:
    metrics_path = OUTDIR / "reviewer_ablation_bootstrap_metrics_dt10.csv"
    deltas_path = OUTDIR / "reviewer_ablation_bootstrap_deltas_vs_pp_cbtact_dt10.csv"
    for path in (metrics_path, deltas_path):
        if not path.exists():
            raise SystemExit(f"Missing bootstrap source table: {path.name}")
    metrics = pd.read_csv(metrics_path)
    deltas = pd.read_csv(deltas_path)
    completed = [model for model in MODEL_ORDER if model in set(metrics["model"])]
    for frame, condition_column in ((metrics, "condition"), (deltas, "comparison")):
        frame["model"] = pd.Categorical(frame["model"], categories=completed, ordered=True)
        frame["domain"] = pd.Categorical(frame["domain"], categories=DOMAIN_ORDER, ordered=True)
        frame[condition_column] = pd.Categorical(frame[condition_column], categories=CONDITION_ORDER, ordered=True)

    document = Document()
    section = document.sections[0]
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width, section.page_height = section.page_height, section.page_width
    section.top_margin = Inches(.55)
    section.bottom_margin = Inches(.55)
    section.left_margin = Inches(.55)
    section.right_margin = Inches(.55)
    normal = document.styles["Normal"]
    normal.font.name = "Helvetica"
    normal.font.size = Pt(9)

    title = document.add_heading("Reviewer 3: Prompt-Component Sensitivity Analysis", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    document.add_paragraph(
        "All results use the canonical dt10-k7 held-out test set (898 messages from 301 participants). "
        "The evaluated outcomes are Content, Coping, and Quitting, each reported separately."
    )
    document.add_paragraph(
        "Intervals are 95% percentile intervals from 2,000 participant-clustered bootstrap replicates. "
        "Directional metrics use low (1–2), neutral (3), and high (4–5)."
    )

    document.add_heading("Table 1. Domain-specific exact and ordinal performance", level=1)
    standard = metrics[metrics["metric"].isin(STANDARD_METRICS)].sort_values(["domain", "model", "condition"])
    add_table(document, ["Domain", "Model", "Configuration", "Exact accuracy (95% CI)", "Macro-F1 (95% CI)", "QWK (95% CI)"],
              make_table_rows(standard, ["domain", "model", "condition"], STANDARD_METRICS, "estimate"))

    document.add_picture(str(OUTDIR / "reviewer_ablation_all_models_accuracy_by_domain_dt10.png"), width=Inches(9.8))
    add_caption(document, "Figure 1. Exact accuracy by tested rating domain. Lines connect configurations evaluated on the same canonical test messages.")

    document.add_heading("Table 2. Domain-specific directional performance", level=1)
    directional = metrics[metrics["metric"].isin(DIRECTIONAL_METRICS)].sort_values(["domain", "model", "condition"])
    add_table(document, ["Domain", "Model", "Configuration", "Directional accuracy (95% CI)", "Directional macro-F1 (95% CI)"],
              make_table_rows(directional, ["domain", "model", "condition"], DIRECTIONAL_METRICS, "estimate"))

    document.add_picture(str(OUTDIR / "reviewer_ablation_all_models_directional_accuracy_by_domain_dt10.png"), width=Inches(9.8))
    add_caption(document, "Figure 2. Directional accuracy by tested rating domain. Direction is low (1–2), neutral (3), or high (4–5).")

    document.add_heading("Table 3. Paired domain-specific changes from PP + history + CBT/ACT", level=1)
    standard_delta = deltas[deltas["metric"].isin(STANDARD_METRICS)].sort_values(["domain", "model", "comparison"])
    add_table(document, ["Domain", "Model", "Compared configuration", "Δ accuracy (95% CI)", "Δ macro-F1 (95% CI)", "Δ QWK (95% CI)"],
              make_table_rows(standard_delta, ["domain", "model", "comparison"], STANDARD_METRICS, "difference"))
    document.add_paragraph("Positive differences favor the comparison configuration. Every interval is paired within the same participant-bootstrap replicate.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document.save(OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
