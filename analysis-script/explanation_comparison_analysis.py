"""
Analyze paired explanation text differences between digital-twin and zero-shot runs.

The generic zero-shot runs were evaluated on the participant split, while the
digital-twin runs were evaluated on the digital-twin split. This script limits
comparison to the exact overlapping (response_id, input_message) pairs.

Outputs:
  - revision/figures/explanation_pairwise_comparison.csv
  - revision/figures/explanation_pairwise_summary.csv
  - revision/figures/explanation_keyword_summary.csv
  - revision/figures/explanation_pair_examples.csv
  - revision/figures/explanation_comparison_report.md

Usage:
  uv run python analysis-script/explanation_comparison_analysis.py
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from revision_utils import (
    MODEL_CONFIGS,
    apply_repo_plot_style,
    figures_path,
    save_figure,
)


DIGITAL_TWIN_FILE = "digital_twin_4_cbtact_7030.json"
BASELINE_FILE = "generic_llm_2_zero_shot_select.json"
BASELINE_LABEL = "Zero-shot (select)"
DIGITAL_TWIN_LABEL = "Digital Twin"
PREDICTION_COLS = [
    "predicted_content",
    "predicted_design",
    "predicted_coping",
    "predicted_quitting",
]

STOPWORDS = {
    "a", "about", "above", "across", "after", "again", "against", "all", "also",
    "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "could", "did", "do",
    "does", "doing", "during", "each", "few", "for", "from", "further", "had", "has",
    "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "like",
    "likely", "may", "me", "more", "most", "my", "myself", "no", "nor", "not", "of",
    "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out",
    "over", "own", "same", "she", "should", "so", "some", "such", "than", "that",
    "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why",
    "will", "with", "would", "you", "your", "yours", "yourself", "yourselves",
}

FIRST_PERSON_RE = re.compile(r"\b(i|me|my|mine)\b", flags=re.IGNORECASE)
HISTORY_TERMS = [
    "past", "previously", "prior", "rated", "ratings", "liked", "disliked",
    "preference", "preferences", "similar", "history", "profile", "consistently",
]
DEMO_TERMS = [
    "motivation", "support", "participant", "person", "demographic", "demographics",
    "motivated", "supportive",
]
MESSAGE_TERMS = [
    "practical", "actionable", "clear", "plain", "simple", "mindfulness",
    "acceptance", "distraction", "urge", "craving", "smoking", "quit", "plan",
]
DT_COLOR = "#0173B2"
BASELINE_COLOR = "#7F7F7F"


def _load_rows(model_id: str, filename: str) -> dict[tuple[str, str], dict]:
    model_dir = Path(MODEL_CONFIGS[model_id]["dir"])
    with open(model_dir / filename) as handle:
        payload = json.load(handle)
    rows = [value for value in payload.values() if isinstance(value, dict)]
    return {(row["response_id"], row["input_message"]): row for row in rows}


def _tokenize(text: str) -> list[str]:
    return [
        token for token in re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())
        if len(token) >= 3 and token not in STOPWORDS
    ]


def _term_set(text: str, ngram: int = 1) -> set[str]:
    tokens = _tokenize(text)
    if ngram == 1:
        return set(tokens)
    if len(tokens) < ngram:
        return set()
    return {" ".join(tokens[idx:idx + ngram]) for idx in range(len(tokens) - ngram + 1)}


def _contains_any(text: str, terms: list[str]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\S+\b", text))


def _sentence_count(text: str) -> int:
    pieces = [piece.strip() for piece in re.split(r"[.!?]+", text) if piece.strip()]
    return len(pieces)


def _token_jaccard(left: str, right: str) -> float:
    left_terms = _term_set(left, ngram=1)
    right_terms = _term_set(right, ngram=1)
    if not left_terms and not right_terms:
        return 1.0
    union = left_terms | right_terms
    if not union:
        return 1.0
    return len(left_terms & right_terms) / len(union)


def _prediction_tuple(row: dict) -> tuple[str, str, str, str]:
    return tuple(str(row.get(col, "")) for col in PREDICTION_COLS)


def build_pairwise_rows() -> pd.DataFrame:
    rows: list[dict] = []
    for model_id, model_cfg in MODEL_CONFIGS.items():
        dt_rows = _load_rows(model_id, DIGITAL_TWIN_FILE)
        baseline_rows = _load_rows(model_id, BASELINE_FILE)
        overlap = sorted(set(dt_rows) & set(baseline_rows))

        for response_id, input_message in overlap:
            dt_row = dt_rows[(response_id, input_message)]
            baseline_row = baseline_rows[(response_id, input_message)]
            dt_expl = str(dt_row.get("explanation", "")).strip()
            baseline_expl = str(baseline_row.get("explanation", "")).strip()
            dt_preds = _prediction_tuple(dt_row)
            baseline_preds = _prediction_tuple(baseline_row)

            rows.append({
                "Model": model_cfg["display"],
                "Response_ID": response_id,
                "Input_Message": input_message,
                "Digital_Twin_Explanation": dt_expl,
                "Baseline_Explanation": baseline_expl,
                "Digital_Twin_Words": _word_count(dt_expl),
                "Baseline_Words": _word_count(baseline_expl),
                "Digital_Twin_Sentences": _sentence_count(dt_expl),
                "Baseline_Sentences": _sentence_count(baseline_expl),
                "Word_Delta_DT_minus_Baseline": _word_count(dt_expl) - _word_count(baseline_expl),
                "Sentence_Delta_DT_minus_Baseline": _sentence_count(dt_expl) - _sentence_count(baseline_expl),
                "Token_Jaccard": _token_jaccard(dt_expl, baseline_expl),
                "Digital_Twin_First_Person": bool(FIRST_PERSON_RE.search(dt_expl)),
                "Baseline_First_Person": bool(FIRST_PERSON_RE.search(baseline_expl)),
                "Digital_Twin_History_Terms": _contains_any(dt_expl, HISTORY_TERMS),
                "Baseline_History_Terms": _contains_any(baseline_expl, HISTORY_TERMS),
                "Digital_Twin_Demo_Terms": _contains_any(dt_expl, DEMO_TERMS),
                "Baseline_Demo_Terms": _contains_any(baseline_expl, DEMO_TERMS),
                "Digital_Twin_Message_Terms": _contains_any(dt_expl, MESSAGE_TERMS),
                "Baseline_Message_Terms": _contains_any(baseline_expl, MESSAGE_TERMS),
                "Digital_Twin_Structured_Labels": all(label in dt_expl for label in ["Content:", "Design:", "Coping:", "Quitting:"]),
                "Baseline_Structured_Labels": all(label in baseline_expl for label in ["Content:", "Design:", "Coping:", "Quitting:"]),
                "Digital_Twin_Predictions": " | ".join(dt_preds),
                "Baseline_Predictions": " | ".join(baseline_preds),
                "Any_Prediction_Difference": bool(dt_preds != baseline_preds),
            })

    return pd.DataFrame(rows)


def build_summary(pair_df: pd.DataFrame) -> pd.DataFrame:
    if pair_df.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for model_name, sub_df in list(pair_df.groupby("Model")) + [("ALL", pair_df)]:
        rows.append({
            "Model": model_name,
            "N_Paired_Items": len(sub_df),
            "Pct_Any_Prediction_Difference": 100.0 * sub_df["Any_Prediction_Difference"].mean(),
            "Mean_DT_Words": sub_df["Digital_Twin_Words"].mean(),
            "Mean_Baseline_Words": sub_df["Baseline_Words"].mean(),
            "Mean_DT_Sentences": sub_df["Digital_Twin_Sentences"].mean(),
            "Mean_Baseline_Sentences": sub_df["Baseline_Sentences"].mean(),
            "Mean_Word_Delta_DT_minus_Baseline": sub_df["Word_Delta_DT_minus_Baseline"].mean(),
            "Mean_Token_Jaccard": sub_df["Token_Jaccard"].mean(),
            "Pct_DT_First_Person": 100.0 * sub_df["Digital_Twin_First_Person"].mean(),
            "Pct_Baseline_First_Person": 100.0 * sub_df["Baseline_First_Person"].mean(),
            "Pct_DT_History_Terms": 100.0 * sub_df["Digital_Twin_History_Terms"].mean(),
            "Pct_Baseline_History_Terms": 100.0 * sub_df["Baseline_History_Terms"].mean(),
            "Pct_DT_Demo_Terms": 100.0 * sub_df["Digital_Twin_Demo_Terms"].mean(),
            "Pct_Baseline_Demo_Terms": 100.0 * sub_df["Baseline_Demo_Terms"].mean(),
            "Pct_DT_Message_Terms": 100.0 * sub_df["Digital_Twin_Message_Terms"].mean(),
            "Pct_Baseline_Message_Terms": 100.0 * sub_df["Baseline_Message_Terms"].mean(),
            "Pct_DT_Structured_Labels": 100.0 * sub_df["Digital_Twin_Structured_Labels"].mean(),
            "Pct_Baseline_Structured_Labels": 100.0 * sub_df["Baseline_Structured_Labels"].mean(),
        })

    return pd.DataFrame(rows)


def _doc_frequency(texts: list[str], ngram: int) -> Counter:
    counts: Counter = Counter()
    for text in texts:
        counts.update(_term_set(text, ngram=ngram))
    return counts


def build_keyword_summary(pair_df: pd.DataFrame) -> pd.DataFrame:
    if pair_df.empty:
        return pd.DataFrame()

    dt_texts = pair_df["Digital_Twin_Explanation"].astype(str).tolist()
    baseline_texts = pair_df["Baseline_Explanation"].astype(str).tolist()
    rows: list[dict] = []

    for ngram, label in [(1, "unigram"), (2, "bigram")]:
        dt_counts = _doc_frequency(dt_texts, ngram=ngram)
        baseline_counts = _doc_frequency(baseline_texts, ngram=ngram)
        terms = set(dt_counts) | set(baseline_counts)
        min_docs = 12 if ngram == 1 else 6
        total_dt = len(dt_texts)
        total_baseline = len(baseline_texts)

        for term in terms:
            dt_docs = dt_counts.get(term, 0)
            baseline_docs = baseline_counts.get(term, 0)
            if max(dt_docs, baseline_docs) < min_docs:
                continue
            dt_rate = dt_docs / total_dt
            baseline_rate = baseline_docs / total_baseline
            rows.append({
                "Ngram": label,
                "Term": term,
                "DT_Doc_Freq": dt_docs,
                "Baseline_Doc_Freq": baseline_docs,
                "DT_Doc_Rate": dt_rate,
                "Baseline_Doc_Rate": baseline_rate,
                "Rate_Delta_DT_minus_Baseline": dt_rate - baseline_rate,
            })

    keyword_df = pd.DataFrame(rows)
    if keyword_df.empty:
        return keyword_df
    keyword_df = keyword_df.sort_values(
        ["Ngram", "Rate_Delta_DT_minus_Baseline", "Term"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    return keyword_df


def build_examples(pair_df: pd.DataFrame) -> pd.DataFrame:
    if pair_df.empty:
        return pd.DataFrame()

    example_rows: list[dict] = []
    for model_name, sub_df in pair_df.groupby("Model"):
        differing = sub_df[sub_df["Any_Prediction_Difference"]].sort_values(
            ["Word_Delta_DT_minus_Baseline", "Token_Jaccard"],
            ascending=[False, True],
        )
        if differing.empty:
            differing = sub_df.sort_values(
                ["Word_Delta_DT_minus_Baseline", "Token_Jaccard"],
                ascending=[False, True],
            )
        for rank, (_, row) in enumerate(differing.head(3).iterrows(), start=1):
            example_rows.append({
                "Model": model_name,
                "Example_Rank": rank,
                "Any_Prediction_Difference": bool(row["Any_Prediction_Difference"]),
                "Input_Message": row["Input_Message"],
                "Digital_Twin_Predictions": row["Digital_Twin_Predictions"],
                "Baseline_Predictions": row["Baseline_Predictions"],
                "Digital_Twin_Explanation": row["Digital_Twin_Explanation"],
                "Baseline_Explanation": row["Baseline_Explanation"],
            })
    return pd.DataFrame(example_rows)


def write_report(summary_df: pd.DataFrame, keyword_df: pd.DataFrame, example_df: pd.DataFrame) -> None:
    report_path = Path(figures_path("explanation_comparison_report")).with_suffix(".md")
    all_row = summary_df[summary_df["Model"] == "ALL"].iloc[0]
    per_model_n = int(summary_df[summary_df["Model"] != "ALL"]["N_Paired_Items"].iloc[0])
    pooled_pairs = int(all_row["N_Paired_Items"])

    def _top_terms(sign: str, ngram: str, limit: int = 8) -> list[str]:
        subset = keyword_df[keyword_df["Ngram"] == ngram].copy()
        subset = subset.sort_values("Rate_Delta_DT_minus_Baseline", ascending=(sign == "baseline"))
        if sign == "baseline":
            subset = subset[subset["Rate_Delta_DT_minus_Baseline"] < 0]
        else:
            subset = subset[subset["Rate_Delta_DT_minus_Baseline"] > 0]
        return subset["Term"].head(limit).tolist()

    dt_bigrams = _top_terms("dt", "bigram")
    baseline_bigrams = _top_terms("baseline", "bigram")
    dt_unigrams = _top_terms("dt", "unigram")
    baseline_unigrams = _top_terms("baseline", "unigram")

    lines = [
        "# Explanation Comparison Report",
        "",
        f"Paired comparison: `{DIGITAL_TWIN_LABEL}` vs `{BASELINE_LABEL}` on the exact overlapping test items.",
        "",
        "## Pooled Summary",
        "",
        f"- Paired items: `{per_model_n}` per model, `{pooled_pairs}` paired items pooled overall (`{2 * pooled_pairs}` explanations total).",
        f"- Any prediction difference: `{all_row['Pct_Any_Prediction_Difference']:.1f}%` of paired items.",
        f"- Mean explanation length: `{DIGITAL_TWIN_LABEL}` `{all_row['Mean_DT_Words']:.1f}` words vs `{BASELINE_LABEL}` `{all_row['Mean_Baseline_Words']:.1f}` words.",
        f"- First-person wording: `{DIGITAL_TWIN_LABEL}` `{all_row['Pct_DT_First_Person']:.1f}%` vs `{BASELINE_LABEL}` `{all_row['Pct_Baseline_First_Person']:.1f}%`.",
        f"- Explicit history/preference cues: `{DIGITAL_TWIN_LABEL}` `{all_row['Pct_DT_History_Terms']:.1f}%` vs `{BASELINE_LABEL}` `{all_row['Pct_Baseline_History_Terms']:.1f}%`.",
        f"- Generic demographic cues: `{DIGITAL_TWIN_LABEL}` `{all_row['Pct_DT_Demo_Terms']:.1f}%` vs `{BASELINE_LABEL}` `{all_row['Pct_Baseline_Demo_Terms']:.1f}%`.",
        f"- Mean token-level Jaccard overlap between paired explanations: `{all_row['Mean_Token_Jaccard']:.3f}`.",
        "",
        "## Distinguishing Phrases",
        "",
        f"- More common in `{DIGITAL_TWIN_LABEL}` bigrams: {', '.join(f'`{term}`' for term in dt_bigrams) if dt_bigrams else 'n/a'}",
        f"- More common in `{BASELINE_LABEL}` bigrams: {', '.join(f'`{term}`' for term in baseline_bigrams) if baseline_bigrams else 'n/a'}",
        f"- More common in `{DIGITAL_TWIN_LABEL}` unigrams: {', '.join(f'`{term}`' for term in dt_unigrams) if dt_unigrams else 'n/a'}",
        f"- More common in `{BASELINE_LABEL}` unigrams: {', '.join(f'`{term}`' for term in baseline_unigrams) if baseline_unigrams else 'n/a'}",
        "",
        "## Representative Paired Examples",
        "",
    ]

    for _, row in example_df.iterrows():
        lines.extend([
            f"### {row['Model']} example {int(row['Example_Rank'])}",
            "",
            f"- Prediction changed: `{row['Any_Prediction_Difference']}`",
            f"- Message: {row['Input_Message']}",
            f"- {DIGITAL_TWIN_LABEL}: `{row['Digital_Twin_Predictions']}`",
            f"- {BASELINE_LABEL}: `{row['Baseline_Predictions']}`",
            f"- {DIGITAL_TWIN_LABEL} explanation: {row['Digital_Twin_Explanation']}",
            f"- {BASELINE_LABEL} explanation: {row['Baseline_Explanation']}",
            "",
        ])

    report_path.write_text("\n".join(lines))


def plot_explanation_overview(summary_df: pd.DataFrame, keyword_df: pd.DataFrame) -> None:
    if summary_df.empty or keyword_df.empty:
        return

    pooled = summary_df[summary_df["Model"] == "ALL"].iloc[0]
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5.5),
        gridspec_kw={"width_ratios": [1.35, 1.0, 1.0]},
    )
    apply_repo_plot_style(fig, axes)

    # Panel 1: cue prevalence
    cue_labels = [
        "First-person",
        "History /\nPreference",
        "Demographic\nCues",
        "Structured\nLabels",
    ]
    dt_vals = [
        float(pooled["Pct_DT_First_Person"]),
        float(pooled["Pct_DT_History_Terms"]),
        float(pooled["Pct_DT_Demo_Terms"]),
        float(pooled["Pct_DT_Structured_Labels"]),
    ]
    baseline_vals = [
        float(pooled["Pct_Baseline_First_Person"]),
        float(pooled["Pct_Baseline_History_Terms"]),
        float(pooled["Pct_Baseline_Demo_Terms"]),
        float(pooled["Pct_Baseline_Structured_Labels"]),
    ]
    x = range(len(cue_labels))
    width = 0.34
    axes[0].bar([i - width / 2 for i in x], dt_vals, width=width, color=DT_COLOR, label=DIGITAL_TWIN_LABEL)
    axes[0].bar([i + width / 2 for i in x], baseline_vals, width=width, color=BASELINE_COLOR, label=BASELINE_LABEL)
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(cue_labels, fontsize=12, fontweight="bold")
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("Paired Explanations Mentioning Cue (%)", fontsize=13, fontweight="bold")
    axes[0].set_title("Explanation Cue Rates", fontsize=14, fontweight="bold")
    axes[0].legend(prop={"weight": "bold", "size": 12}, frameon=False, loc="upper left")
    axes[0].text(
        0.98,
        0.04,
        (
            f"Mean words: {pooled['Mean_DT_Words']:.1f} vs {pooled['Mean_Baseline_Words']:.1f}\n"
            f"Prediction changes: {pooled['Pct_Any_Prediction_Difference']:.1f}%\n"
            f"Token overlap: {pooled['Mean_Token_Jaccard']:.3f}"
        ),
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "#4D4D4D", "alpha": 0.9, "boxstyle": "round,pad=0.3"},
    )

    # Panel 2: DT-specific phrases
    top_dt = keyword_df[
        (keyword_df["Ngram"] == "bigram") &
        (keyword_df["Rate_Delta_DT_minus_Baseline"] > 0)
    ].head(8).copy()
    top_dt = top_dt.sort_values("Rate_Delta_DT_minus_Baseline", ascending=True)
    axes[1].barh(
        top_dt["Term"],
        100 * top_dt["Rate_Delta_DT_minus_Baseline"],
        color=DT_COLOR,
        alpha=0.9,
    )
    axes[1].set_title(f"Phrases More Common in {DIGITAL_TWIN_LABEL}", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Doc Frequency Delta (pp)", fontsize=13, fontweight="bold")

    # Panel 3: baseline-specific phrases
    top_base = keyword_df[
        (keyword_df["Ngram"] == "bigram") &
        (keyword_df["Rate_Delta_DT_minus_Baseline"] < 0)
    ].sort_values("Rate_Delta_DT_minus_Baseline", ascending=True).head(8).copy()
    top_base["Baseline_Advantage_pp"] = -100 * top_base["Rate_Delta_DT_minus_Baseline"]
    top_base = top_base.sort_values("Baseline_Advantage_pp", ascending=True)
    axes[2].barh(
        top_base["Term"],
        top_base["Baseline_Advantage_pp"],
        color=BASELINE_COLOR,
        alpha=0.9,
    )
    axes[2].set_title(f"Phrases More Common in {BASELINE_LABEL}", fontsize=14, fontweight="bold")
    axes[2].set_xlabel("Doc Frequency Delta (pp)", fontsize=13, fontweight="bold")

    for ax in axes[1:]:
        ax.grid(axis="x", alpha=0.12, color="#4D4D4D", linestyle="--", linewidth=0.6)
        ax.grid(False, axis="y")
        ax.tick_params(axis="both", labelsize=11)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")

    fig.suptitle(
        "Digital-Twin vs Zero-Shot Explanation Differences",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    fig.subplots_adjust(top=0.82, bottom=0.14, left=0.06, right=0.99, wspace=0.42)
    save_figure(fig, figures_path("explanation_comparison_overview"))


def main() -> None:
    pair_df = build_pairwise_rows()
    summary_df = build_summary(pair_df)
    keyword_df = build_keyword_summary(pair_df)
    example_df = build_examples(pair_df)

    pair_df.to_csv(Path(figures_path("explanation_pairwise_comparison")).with_suffix(".csv"), index=False)
    summary_df.to_csv(Path(figures_path("explanation_pairwise_summary")).with_suffix(".csv"), index=False)
    keyword_df.to_csv(Path(figures_path("explanation_keyword_summary")).with_suffix(".csv"), index=False)
    example_df.to_csv(Path(figures_path("explanation_pair_examples")).with_suffix(".csv"), index=False)
    if not summary_df.empty:
        write_report(summary_df, keyword_df, example_df)
        plot_explanation_overview(summary_df, keyword_df)

    if summary_df.empty:
        print("No overlapping paired items found.")
        return

    pooled = summary_df[summary_df["Model"] == "ALL"].iloc[0]
    per_model_n = int(summary_df[summary_df["Model"] != "ALL"]["N_Paired_Items"].iloc[0])
    print(f"Paired items per model: {per_model_n}")
    print(f"Paired items pooled overall: {int(pooled['N_Paired_Items'])}")
    print(f"Prediction differences: {pooled['Pct_Any_Prediction_Difference']:.1f}%")
    print(
        f"Mean words: {DIGITAL_TWIN_LABEL} {pooled['Mean_DT_Words']:.1f} vs "
        f"{BASELINE_LABEL} {pooled['Mean_Baseline_Words']:.1f}"
    )


if __name__ == "__main__":
    main()
