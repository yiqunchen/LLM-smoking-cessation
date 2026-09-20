"""
Create explanation word clouds from paired digital-twin vs zero-shot explanations.

Design choice:
- Main figure: 3 x 2 grid (domain x style), pooled across models on the exact
  overlapping paired items.
- Supplementary figure: 5 x 2 grid (model x style), pooled across domains.

Usage:
  UV_CACHE_DIR=/tmp/uv-cache uv run --with wordcloud python analysis-script/explanation_wordclouds.py
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

from revision_utils import MODEL_CONFIGS, figures_path, save_figure


STYLE_FILES = {
    "Digital Twin": "digital_twin_4_cbtact_7030.json",
    "Zero-shot (select)": "generic_llm_2_zero_shot_select.json",
}
STYLE_COLORS = {
    "Digital Twin": "#0173B2",
    "Zero-shot (select)": "#7F7F7F",
}
DOMAIN_ORDER = ["content", "coping", "quitting"]
DOMAIN_LABELS = {
    "content": "Content",
    "coping": "Coping",
    "quitting": "Quitting",
}

STOPWORDS = {
    "accept", "acceptable", "across", "actually", "after", "again", "against", "all",
    "also", "although", "always", "and", "another", "any", "around", "because", "been",
    "before", "being", "between", "both", "briefly", "but", "can", "clear", "content",
    "coping", "could", "daily", "design", "does", "doing", "during", "easy", "enough",
    "even", "extremely", "fairly", "feels", "find", "fine", "for", "from", "generic",
    "get", "getting", "gives", "good", "help", "helpful", "helps", "high", "highly",
    "how", "immediate", "immediately", "indirectly", "instead", "into", "isn't", "it's",
    "its", "just", "kind", "lacks", "less", "likely", "little", "long", "longer", "look",
    "looks", "low", "mainly", "make", "managing", "may", "maybe", "moderate",
    "moderately", "more", "most", "mostly", "much", "needs", "not", "nothing", "offers",
    "often", "only", "overall", "participant", "people", "person", "plain", "practical",
    "provides", "quitting", "rated", "rating", "ratings", "really", "relevant", "say",
    "seems", "simple", "simply", "smoking", "some", "someone", "somewhat", "specific",
    "still", "strategy", "strong", "support", "supportive", "supports", "that", "the",
    "their", "them", "there", "these", "they", "this", "though", "through", "tool",
    "use", "useful", "very", "well", "when", "while", "with", "without", "would", "yet",
    "you", "your",
    # section labels / boilerplate
    "message", "messages", "designs", "coping", "quitting", "content", "participant's",
}
TOKEN_MAP = {
    "motivated": "motivation",
    "motivation": "motivation",
    "supportive": "support",
    "supports": "support",
    "supporting": "support",
    "smokers": "smoker",
    "cravings": "craving",
    "urges": "urge",
    "attempts": "attempt",
    "ratings": "history",
    "rated": "history",
    "past": "history",
    "previously": "history",
    "prior": "history",
    "similarity": "similar",
    "preferences": "preference",
}
SECTION_PATTERN = re.compile(r"(Content|Design|Coping|Quitting)\s*:\s*", re.IGNORECASE)


def _load_rows(model_id: str, filename: str) -> dict[tuple[str, str], dict]:
    model_dir = Path(MODEL_CONFIGS[model_id]["dir"])
    with open(model_dir / filename) as handle:
        payload = json.load(handle)
    rows = [value for value in payload.values() if isinstance(value, dict)]
    return {(row["response_id"], row["input_message"]): row for row in rows}


def parse_sections(text: str) -> dict[str, str]:
    text = str(text or "").strip()
    if not text:
        return {}

    if text.startswith("{") and ("content" in text.lower()) and ("coping" in text.lower()):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, dict):
                sections = {}
                for key, value in parsed.items():
                    key_norm = str(key).strip().lower()
                    if key_norm in {"content", "coping", "quitting"}:
                        sections[key_norm] = str(value).strip()
                if sections:
                    return sections
        except Exception:
            pass

    matches = list(SECTION_PATTERN.finditer(text))
    if not matches:
        return {}

    sections = {}
    for idx, match in enumerate(matches):
        key = match.group(1).strip().lower()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        if key in {"content", "coping", "quitting"}:
            sections[key] = text[start:end].strip(" \n:-")
    return sections


def tokenize(text: str) -> list[str]:
    tokens = []
    for raw in re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower()):
        token = TOKEN_MAP.get(raw, raw)
        if len(token) < 3 or token in STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def build_frequency(texts: list[str], top_n: int = 80) -> Counter:
    counter: Counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    return Counter(dict(counter.most_common(top_n)))


def single_color_func(color: str):
    def _color_func(*args, **kwargs):
        return color
    return _color_func


def generate_wordcloud(counter: Counter, color: str) -> WordCloud:
    wc = WordCloud(
        width=1200,
        height=800,
        background_color="white",
        max_words=80,
        prefer_horizontal=0.85,
        collocations=False,
        margin=3,
        min_font_size=10,
        max_font_size=130,
    )
    return wc.generate_from_frequencies(counter).recolor(color_func=single_color_func(color))


def collect_texts():
    domain_texts: dict[tuple[str, str], list[str]] = defaultdict(list)
    model_texts: dict[tuple[str, str], list[str]] = defaultdict(list)
    coverage_rows: list[dict] = []

    for model_id, model_cfg in MODEL_CONFIGS.items():
        loaded = {style: _load_rows(model_id, filename) for style, filename in STYLE_FILES.items()}
        overlap = sorted(set(loaded["Digital Twin"]) & set(loaded["Zero-shot (select)"]))

        parsed_counts = Counter()
        for key in overlap:
            parsed_by_style = {}
            for style, rows in loaded.items():
                expl = str(rows[key].get("explanation", "")).strip()
                model_texts[(model_cfg["display"], style)].append(expl)
                parsed_by_style[style] = parse_sections(expl)

            # Keep domain clouds on the exact same paired subset:
            # a section is included only when both prompt styles produced
            # a parseable section for that same message and domain.
            for domain in DOMAIN_ORDER:
                dt_text = parsed_by_style["Digital Twin"].get(domain, "")
                zs_text = parsed_by_style["Zero-shot (select)"].get(domain, "")
                if dt_text and zs_text:
                    domain_texts[(domain, "Digital Twin")].append(dt_text)
                    domain_texts[(domain, "Zero-shot (select)")].append(zs_text)
                    parsed_counts[domain] += 1

        for style in STYLE_FILES:
            for domain in DOMAIN_ORDER:
                coverage_rows.append({
                    "Model": model_cfg["display"],
                    "Style": style,
                    "Domain": DOMAIN_LABELS[domain],
                    "Overlap_Items": len(overlap),
                    "Paired_Parsed_Domain_Sections": parsed_counts[domain],
                })

    return domain_texts, model_texts, pd.DataFrame(coverage_rows)


def plot_domain_style_wordclouds(domain_texts: dict[tuple[str, str], list[str]]) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 16))
    fig.patch.set_facecolor("white")

    for row_idx, domain in enumerate(DOMAIN_ORDER):
        for col_idx, style in enumerate(STYLE_FILES.keys()):
            ax = axes[row_idx, col_idx]
            texts = domain_texts.get((domain, style), [])
            freq = build_frequency(texts)
            ax.set_facecolor("white")
            ax.axis("off")
            if freq:
                wc = generate_wordcloud(freq, STYLE_COLORS[style])
                ax.imshow(wc, interpolation="bilinear")
            ax.set_title(
                f"{DOMAIN_LABELS[domain]} | {style}\n(n = {len(texts)} paired sections)",
                fontsize=13,
                fontweight="bold",
                pad=12,
            )

    fig.suptitle(
        "Explanation Word Clouds by Domain and Prompt Style",
        fontsize=17,
        fontweight="bold",
        y=0.995,
    )
    fig.subplots_adjust(top=0.95, bottom=0.02, left=0.03, right=0.97, hspace=0.18, wspace=0.04)
    save_figure(fig, figures_path("explanation_wordclouds_domain_style"))


def plot_model_style_wordclouds(model_texts: dict[tuple[str, str], list[str]]) -> None:
    model_order = [cfg["display"] for cfg in MODEL_CONFIGS.values()]
    fig, axes = plt.subplots(len(model_order), 2, figsize=(12.5, 22))
    fig.patch.set_facecolor("white")

    for row_idx, model_name in enumerate(model_order):
        for col_idx, style in enumerate(STYLE_FILES.keys()):
            ax = axes[row_idx, col_idx]
            texts = model_texts.get((model_name, style), [])
            freq = build_frequency(texts)
            ax.set_facecolor("white")
            ax.axis("off")
            if freq:
                wc = generate_wordcloud(freq, STYLE_COLORS[style])
                ax.imshow(wc, interpolation="bilinear")
            ax.set_title(
                f"{model_name} | {style}",
                fontsize=12,
                fontweight="bold",
                pad=10,
            )

    fig.suptitle(
        "Explanation Word Clouds by Model and Prompt Style",
        fontsize=17,
        fontweight="bold",
        y=0.995,
    )
    fig.subplots_adjust(top=0.975, bottom=0.01, left=0.03, right=0.97, hspace=0.22, wspace=0.04)
    save_figure(fig, figures_path("explanation_wordclouds_model_style"))


def main() -> None:
    domain_texts, model_texts, coverage_df = collect_texts()
    coverage_df.to_csv(Path(figures_path("explanation_wordcloud_coverage")).with_suffix(".csv"), index=False)
    plot_domain_style_wordclouds(domain_texts)
    plot_model_style_wordclouds(model_texts)
    print("Saved explanation word cloud figures and coverage CSV.")


if __name__ == "__main__":
    main()
