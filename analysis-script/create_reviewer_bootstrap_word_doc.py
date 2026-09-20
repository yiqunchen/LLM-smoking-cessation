#!/usr/bin/env python3
"""Create a Word-ready Reviewer 3 ablation-results document from cached tables."""
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
METRIC_ORDER = ["accuracy", "macro_f1", "qwk"]
METRIC_LABELS = {"accuracy": "Exact accuracy", "macro_f1": "Macro-F1", "qwk": "QWK"}
DIRECTIONAL_METRIC_ORDER = ["directional_accuracy", "directional_macro_f1"]
DIRECTIONAL_METRIC_LABELS = {
    "directional_accuracy": "Directional accuracy",
    "directional_macro_f1": "Directional macro-F1",
}


def shade(cell, fill: str) -> None:
    properties = cell._tc.get_or_add_tcPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), fill)
    properties.append(shading)


def set_cell_text(cell, text: str, bold: bool = False, color: str | None = None) -> None:
    cell.text = ""
    run = cell.paragraphs[0].add_run(text)
    run.bold = bold
    run.font.size = Pt(8.5)
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def add_table(doc: Document, columns: list[str], rows: list[list[str]]) -> None:
    table = doc.add_table(rows=1, cols=len(columns))
    table.style = "Table Grid"
    table.autofit = True
    for cell, title in zip(table.rows[0].cells, columns):
        set_cell_text(cell, title, bold=True, color="FFFFFF")
        shade(cell, "1F4E79")
    for row in rows:
        cells = table.add_row().cells
        for cell, value in zip(cells, row):
            set_cell_text(cell, value)
    doc.add_paragraph()


def format_ci(row: pd.Series, value: str) -> str:
    return f"{row[value]:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}]"


def add_caption(doc: Document, text: str) -> None:
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run(text)
    run.italic = True
    run.font.size = Pt(9)


def main() -> None:
    metrics_path = OUTDIR / "reviewer_ablation_bootstrap_metrics_dt10.csv"
    deltas_path = OUTDIR / "reviewer_ablation_bootstrap_deltas_vs_pp_cbtact_dt10.csv"
    status_path = OUTDIR / "reviewer_ablation_bootstrap_status_dt10.csv"
    for path in (metrics_path, deltas_path, status_path):
        if not path.exists():
            raise SystemExit(f"Missing bootstrap source table: {path.name}")
    metrics = pd.read_csv(metrics_path)
    deltas = pd.read_csv(deltas_path)
    status = pd.read_csv(status_path)
    completed = [model for model in MODEL_ORDER if model in set(metrics["model"])]
    pending = status.loc[status["status"] == "pending", "model"].tolist()

    document = Document()
    section = document.sections[0]
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width, section.page_height = section.page_height, section.page_width
    section.top_margin = Inches(0.55)
    section.bottom_margin = Inches(0.55)
    section.left_margin = Inches(0.55)
    section.right_margin = Inches(0.55)
    normal = document.styles["Normal"]
    normal.font.name = "Arial"
    normal.font.size = Pt(9.5)
    normal.paragraph_format.space_after = Pt(5)

    title = document.add_heading("Reviewer 3: Prompt-Component Sensitivity Analysis", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run("Confidence-interval results on the canonical dt10-k7 held-out test set")
    run.italic = True
    run.font.size = Pt(10)

    document.add_heading("Analysis summary", level=1)
    model_text = ", ".join(completed)
    document.add_paragraph(
        f"This analysis evaluates {len(completed)} completed model families ({model_text}) "
        "on the same 898 held-out messages from 301 participants. Each model was evaluated under four configurations: "
        "PP + history + CBT/ACT; PP + history without CBT/ACT; history + ratings only; and history text only."
    )
    document.add_paragraph(
        "Confidence intervals are 95% percentile intervals from 2,000 participant-clustered bootstrap replicates. "
        "Each replicate resamples participants and retains all of their held-out messages, accounting for repeated messages within participants. "
        "Quadratic-weighted kappa (QWK) uses the fixed ordinal 1–5 rating scale; directional metrics use low (1–2), neutral (3), and high (4–5)."
    )
    if pending:
        pending_note = document.add_paragraph()
        pending_note.add_run("DeepSeek-R1 status: ").bold = True
        pending_note.add_run(
            "its four condition runs are still incomplete and it is intentionally excluded from every result and interval in this document. "
            "The cached pipeline will add only DeepSeek-R1 after it reaches all 898 rows per condition."
        )

    document.add_heading("Table 1. Overall performance across four rating dimensions", level=1)
    overall = metrics[(metrics["domain"] == "Mean (4 domains)")].copy()
    overall["model"] = pd.Categorical(overall["model"], categories=completed, ordered=True)
    overall["condition"] = pd.Categorical(overall["condition"], categories=CONDITION_ORDER, ordered=True)
    overall = overall.sort_values(["model", "condition"])
    metric_pivot = overall.pivot(index=["model", "condition"], columns="metric", values=["estimate", "ci_low", "ci_high"])
    metric_pivot = metric_pivot.reindex(columns=pd.MultiIndex.from_product([["estimate", "ci_low", "ci_high"], METRIC_ORDER]))
    table_rows: list[list[str]] = []
    for (model, condition), values in metric_pivot.iterrows():
        cells = [str(model), str(condition)]
        for metric in METRIC_ORDER:
            cells.append(f"{values[('estimate', metric)]:.3f} [{values[('ci_low', metric)]:.3f}, {values[('ci_high', metric)]:.3f}]")
        table_rows.append(cells)
    add_table(document, ["Model", "Configuration", "Exact accuracy (95% CI)", "Macro-F1 (95% CI)", "QWK (95% CI)"], table_rows)

    all_models_ready = len(completed) == len(MODEL_ORDER)
    overall_figure = OUTDIR / ("reviewer_ablation_all_models_accuracy_dt10.png" if all_models_ready else "reviewer_ablation_completed_models_overall_dt10.png")
    document.add_picture(str(overall_figure), width=Inches(9.6))
    if all_models_ready:
        add_caption(document, "Figure 1. Mean exact accuracy across the four rating dimensions for all five model families. Lines connect configurations evaluated on the same canonical test messages.")
    else:
        add_caption(document, "Figure 1. Mean exact accuracy, macro-F1, and QWK across the four rating dimensions. Lines connect configurations evaluated on the same canonical test messages.")

    document.add_heading("Table 2. Directional performance across four rating dimensions", level=1)
    directional_pivot = overall.pivot(index=["model", "condition"], columns="metric", values=["estimate", "ci_low", "ci_high"])
    directional_pivot = directional_pivot.reindex(
        columns=pd.MultiIndex.from_product([["estimate", "ci_low", "ci_high"], DIRECTIONAL_METRIC_ORDER])
    )
    directional_rows: list[list[str]] = []
    for (model, condition), values in directional_pivot.iterrows():
        cells = [str(model), str(condition)]
        for metric in DIRECTIONAL_METRIC_ORDER:
            cells.append(f"{values[('estimate', metric)]:.3f} [{values[('ci_low', metric)]:.3f}, {values[('ci_high', metric)]:.3f}]")
        directional_rows.append(cells)
    add_table(
        document,
        ["Model", "Configuration", "Directional accuracy (95% CI)", "Directional macro-F1 (95% CI)"],
        directional_rows,
    )

    directional_figure = OUTDIR / "reviewer_ablation_all_models_directional_accuracy_dt10.png"
    document.add_picture(str(directional_figure), width=Inches(9.6))
    add_caption(document, "Figure 2. Mean directional accuracy across the four rating dimensions. Direction is low (ratings 1–2), neutral (3), or high (4–5).")

    document.add_heading("Table 3. Paired changes from PP + history + CBT/ACT", level=1)
    paired = deltas[deltas["domain"] == "Mean (4 domains)"].copy()
    paired["model"] = pd.Categorical(paired["model"], categories=completed, ordered=True)
    paired["comparison"] = pd.Categorical(paired["comparison"], categories=CONDITION_ORDER[1:], ordered=True)
    paired = paired.sort_values(["model", "comparison", "metric"])
    paired_pivot = paired.pivot(index=["model", "comparison"], columns="metric", values=["difference", "ci_low", "ci_high"])
    paired_pivot = paired_pivot.reindex(columns=pd.MultiIndex.from_product([["difference", "ci_low", "ci_high"], METRIC_ORDER]))
    paired_rows: list[list[str]] = []
    for (model, comparison), values in paired_pivot.iterrows():
        cells = [str(model), str(comparison)]
        for metric in METRIC_ORDER:
            cells.append(f"{values[('difference', metric)]:+.3f} [{values[('ci_low', metric)]:+.3f}, {values[('ci_high', metric)]:+.3f}]")
        paired_rows.append(cells)
    add_table(document, ["Model", "Compared configuration", "Δ accuracy (95% CI)", "Δ macro-F1 (95% CI)", "Δ QWK (95% CI)"], paired_rows)
    document.add_paragraph(
        "Positive differences favor the comparison configuration. Every interval is paired: the two configurations use the same participant-bootstrap replicate."
    )

    document.add_heading("Table 4. Paired directional changes from PP + history + CBT/ACT", level=1)
    paired_directional_pivot = paired.pivot(index=["model", "comparison"], columns="metric", values=["difference", "ci_low", "ci_high"])
    paired_directional_pivot = paired_directional_pivot.reindex(
        columns=pd.MultiIndex.from_product([["difference", "ci_low", "ci_high"], DIRECTIONAL_METRIC_ORDER])
    )
    paired_directional_rows: list[list[str]] = []
    for (model, comparison), values in paired_directional_pivot.iterrows():
        cells = [str(model), str(comparison)]
        for metric in DIRECTIONAL_METRIC_ORDER:
            cells.append(
                f"{values[('difference', metric)]:+.3f} "
                f"[{values[('ci_low', metric)]:+.3f}, {values[('ci_high', metric)]:+.3f}]"
            )
        paired_directional_rows.append(cells)
    add_table(
        document,
        ["Model", "Compared configuration", "Δ directional accuracy (95% CI)", "Δ directional macro-F1 (95% CI)"],
        paired_directional_rows,
    )

    domain_figure = OUTDIR / ("reviewer_ablation_all_models_accuracy_by_domain_dt10.png" if all_models_ready else "reviewer_ablation_completed_models_accuracy_by_domain_dt10.png")
    document.add_picture(str(domain_figure), width=Inches(8.8))
    add_caption(document, "Figure 3. Exact accuracy by rating dimension. The content, design, coping, and quitting panels use the same configuration order and model colors as Table 1.")

    directional_domain_figure = OUTDIR / "reviewer_ablation_all_models_directional_accuracy_by_domain_dt10.png"
    document.add_picture(str(directional_domain_figure), width=Inches(8.8))
    add_caption(document, "Figure 4. Directional accuracy by rating dimension. Direction is defined consistently as low (1–2), neutral (3), or high (4–5).")

    document.add_heading("Supplementary distribution checks", level=1)
    document.add_paragraph(
        "Separate observed-versus-predicted 1–5 score-frequency histograms were generated for Content, Design, Coping, and Quitting. "
        "They are retained as companion figures because pooling domains would mix different rating-wording scales. Each displayed distribution sums to 898 canonical test messages."
    )
    document.add_paragraph(
        "Source-data tables contain the complete domain-level exact and directional metrics, paired differences, and histogram counts."
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document.save(OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
