#!/usr/bin/env python3
"""
Plot pooled raw human-rating distributions by gender and race subgroup.

Uses all ratings from the digital-twin 70/30 train + test partitions so the
full dataset is represented exactly once.

Outputs:
    revision/figures/all_rating_subgroup_histograms.png/pdf
"""

from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from revision_utils import load_canonical_data, figures_path, save_figure


FONT_CHAIN = ["Arial", "Helvetica", "Helvetica Neue", "Avenir",
              "Avenir Next", "DejaVu Sans"]
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": FONT_CHAIN,
    "font.size": 15,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.8,
    "grid.alpha": 0.12,
    "grid.color": "#4D4D4D",
    "grid.linestyle": "--",
    "grid.linewidth": 0.7,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 1.8,
    "ytick.major.width": 1.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


RATING_MAPS = {
    "content": {"Very poor": 1, "Poor": 2, "Acceptable": 3, "Good": 4, "Very good": 5},
    "coping": {
        "Not at all helpful": 1,
        "Somewhat helpful": 2,
        "Moderately helpful": 3,
        "Very helpful": 4,
        "Extremely helpful": 5,
        "Not Helpful": 1,
    },
    "quitting": {
        "Not at all helpful": 1,
        "Somewhat helpful": 2,
        "Moderately helpful": 3,
        "Very helpful": 4,
        "Extremely helpful": 5,
        "Not Helpful": 1,
    },
}

GENDER_ORDER = ["Male", "Female", "Other"]
RACE_ORDER = ["White", "Black / African American", "Other"]

GENDER_COLORS = {
    "Male": "#0173B2",
    "Female": "#DE8F05",
    "Other": "#7F7F7F",
}

RACE_COLORS = {
    "White": "#7F7F7F",
    "Black / African American": "#029E73",
    "Other": "#CA9161",
}


def classify_gender(gender_val: str | None) -> str:
    if not isinstance(gender_val, str) or gender_val.strip() == "":
        return "Other"
    gender = gender_val.strip()
    if gender == "Male":
        return "Male"
    if gender == "Female":
        return "Female"
    return "Other"


def classify_race(race_val: str | None) -> str:
    if not isinstance(race_val, str) or race_val.strip() == "":
        return "Other"
    race = race_val.strip()
    if race == "White":
        return "White"
    if race == "Black or African American":
        return "Black / African American"
    return "Other"


def load_all_records() -> list[dict]:
    train, test = load_canonical_data("7030", "digital_twin")
    return train + test


def collect_grouped_scores(records: list[dict]) -> dict:
    gender_scores = defaultdict(lambda: defaultdict(list))
    race_scores = defaultdict(lambda: defaultdict(list))
    gender_message_counts = Counter()
    race_message_counts = Counter()
    gender_other_raw = Counter()
    race_other_raw = Counter()

    for row in records:
        metadata = row.get("metadata", {}) if isinstance(row, dict) else {}
        ratings = row.get("ratings", {}) if isinstance(row, dict) else {}

        raw_gender = metadata.get("gender_identity")
        raw_race = metadata.get("race_ethnicity")
        gender_group = classify_gender(raw_gender)
        race_group = classify_race(raw_race)

        gender_message_counts[gender_group] += 1
        race_message_counts[race_group] += 1

        if gender_group == "Other":
            gender_other_raw[str(raw_gender)] += 1
        if race_group == "Other":
            race_other_raw[str(raw_race)] += 1

        for domain, mapping in RATING_MAPS.items():
            score = mapping.get(ratings.get(domain))
            if score is None:
                continue
            gender_scores[gender_group][domain].append(score)
            race_scores[race_group][domain].append(score)

    return {
        "gender_scores": gender_scores,
        "race_scores": race_scores,
        "gender_message_counts": gender_message_counts,
        "race_message_counts": race_message_counts,
        "gender_other_raw": gender_other_raw,
        "race_other_raw": race_other_raw,
    }


def proportions(scores: list[int]) -> np.ndarray:
    arr = np.asarray(scores, dtype=int)
    counts = np.array([(arr == score).sum() for score in range(1, 6)], dtype=float)
    if counts.sum() == 0:
        return counts
    return counts / counts.sum()


def plot_panel(
    ax,
    domain: str,
    title: str,
    order: list[str],
    colors: dict[str, str],
    scores: dict,
    message_counts: Counter,
    show_legend: bool,
):
    x = np.arange(1, 6, dtype=float)
    bar_width = 0.24
    offsets = np.linspace(-bar_width, bar_width, len(order))

    for offset, group in zip(offsets, order):
        vals = proportions(scores[group][domain])
        ax.bar(
            x + offset,
            vals,
            width=bar_width * 0.92,
            color=colors[group],
            edgecolor="black",
            linewidth=1.0,
            alpha=0.95,
            label=f"{group} (n={message_counts[group]})",
        )

    ax.set_title(title, fontsize=18, fontweight="bold", pad=12)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("Ground-Truth Rating", fontsize=16, fontweight="bold")
    ax.set_ylabel("Proportion of Ratings", fontsize=16, fontweight="bold")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, 0.5)
    ax.grid(axis="y", alpha=0.12, color="#4D4D4D", linestyle="--", linewidth=0.7)
    ax.set_facecolor("white")
    ax.tick_params(axis="both", labelsize=15, width=1.8, length=6, direction="out")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
    if show_legend:
        ax.legend(loc="upper right", frameon=False,
                  prop={"weight": "bold", "size": 13})


def main():
    print("=" * 70)
    print("All-Rating Subgroup Histograms")
    print("=" * 70)

    records = load_all_records()
    grouped = collect_grouped_scores(records)

    print(f"Loaded {len(records)} total messages from digital-twin train+test.")
    print(f"Gender message counts: {dict(grouped['gender_message_counts'])}")
    print(f"Race message counts: {dict(grouped['race_message_counts'])}")
    print(f"Gender Other categories: {dict(grouped['gender_other_raw'])}")
    print(f"Race Other categories: {dict(grouped['race_other_raw'])}")

    domains = ["content", "coping", "quitting"]
    domain_titles = {"content": "Content", "coping": "Coping", "quitting": "Quitting"}

    fig, axes = plt.subplots(3, 2, figsize=(18.5, 16.2), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "All Human Ratings: Raw Score Distributions by Domain and Demographic Subgroup",
        fontsize=22,
        fontweight="bold",
        y=0.98,
    )

    for row_idx, domain in enumerate(domains):
        plot_panel(
            axes[row_idx, 0],
            domain=domain,
            title=f"{domain_titles[domain]}: Gender",
            order=GENDER_ORDER,
            colors=GENDER_COLORS,
            scores=grouped["gender_scores"],
            message_counts=grouped["gender_message_counts"],
            show_legend=(row_idx == 0),
        )
        plot_panel(
            axes[row_idx, 1],
            domain=domain,
            title=f"{domain_titles[domain]}: Race",
            order=RACE_ORDER,
            colors=RACE_COLORS,
            scores=grouped["race_scores"],
            message_counts=grouped["race_message_counts"],
            show_legend=(row_idx == 0),
        )
        axes[row_idx, 1].set_ylabel("")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, figures_path("all_rating_subgroup_histograms"))


if __name__ == "__main__":
    main()
