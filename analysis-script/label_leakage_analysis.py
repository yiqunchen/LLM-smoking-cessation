#!/usr/bin/env python3
"""
Label Leakage Analysis -- Revision Script
=========================================
Addresses Reviewer R3 concern about potential data leakage.

Checks:
  1. Few-shot exemplar overlap with test set (participant split)
  2. PP profile/test text overlap and TF-IDF similarity
  3. Cross-split message text overlap (participant split; expected)

Outputs:
  - revision/figures/label_leakage_analysis.csv
  - revision/figures/profile_test_similarity_distribution.png/pdf
"""

import os
import sys

# Ensure analysis-script is on path for sibling imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from revision_utils import (
    load_canonical_data,
    DOMAINS,
    save_figure,
    figures_path,
    PROJECT_ROOT,
)
from prompt_config import prepare_few_shot_examples, FEW_SHOT_EXAMPLES


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _item_key(item: dict) -> tuple:
    """Canonical identity key for a data item."""
    return (item["response_id"], item["input_message"])


def _risk_level(pct: float) -> str:
    if pct == 0:
        return "None"
    elif pct < 1:
        return "Low"
    elif pct < 10:
        return "Medium"
    else:
        return "High"


# ---------------------------------------------------------------------------
# Check 1 -- Few-shot exemplar overlap with test set
# ---------------------------------------------------------------------------

def check_few_shot_overlap(results: list) -> list:
    """Return overlap rows for the leakage table."""
    print("\n=== Check 1: Few-shot exemplar overlap with test set ===")

    # Load participant 70/30 splits
    train_data, test_data = load_canonical_data("7030", "participant")

    # Build training dict expected by prepare_few_shot_examples
    train_dict = {str(i): item for i, item in enumerate(train_data)}
    prepare_few_shot_examples(train_dict)

    # Collect all exemplar keys
    exemplar_keys = set()
    for dim, examples in FEW_SHOT_EXAMPLES.items():
        for level in ("high", "low"):
            ex = examples.get(level)
            if ex is not None:
                exemplar_keys.add(_item_key(ex))

    # Build test key set
    test_keys = {_item_key(item) for item in test_data}

    overlap = exemplar_keys & test_keys
    n_overlap = len(overlap)
    n_checked = len(exemplar_keys)
    pct = (n_overlap / n_checked * 100) if n_checked else 0

    print(f"  Exemplars checked : {n_checked}")
    print(f"  Overlapping w/test: {n_overlap}")
    if overlap:
        for rid, msg in overlap:
            print(f"    - response_id={rid}, message={msg[:60]}...")

    results.append({
        "Check_Type": "Few-shot exemplar overlap",
        "Description": "Exemplar (response_id, input_message) pairs that also appear in participant test set",
        "N_Overlapping": n_overlap,
        "Total_Checked": n_checked,
        "Pct_Overlap": round(pct, 2),
        "Risk_Level": _risk_level(pct),
    })
    return results


# ---------------------------------------------------------------------------
# Check 2 -- PP profile / test overlap & similarity
# ---------------------------------------------------------------------------

def check_digital_twin_overlap(results: list) -> list:
    """Check exact overlap and compute TF-IDF cosine similarities."""
    print("\n=== Check 2: PP profile/test overlap & similarity ===")

    train_data, test_data = load_canonical_data("7030", "digital_twin")

    # --- 2a. Exact text overlap ---
    train_keys = {_item_key(item) for item in train_data}
    test_keys = {_item_key(item) for item in test_data}

    exact_overlap = train_keys & test_keys
    n_exact = len(exact_overlap)
    n_checked = len(test_keys)
    pct_exact = (n_exact / n_checked * 100) if n_checked else 0

    print(f"  Exact (response_id, input_message) overlap: {n_exact} / {n_checked} ({pct_exact:.2f}%)")

    results.append({
        "Check_Type": "PP exact text overlap",
        "Description": "Profile items with identical (response_id, input_message) in test set",
        "N_Overlapping": n_exact,
        "Total_Checked": n_checked,
        "Pct_Overlap": round(pct_exact, 2),
        "Risk_Level": _risk_level(pct_exact),
    })

    # --- 2b. TF-IDF cosine similarity per participant ---
    print("  Computing per-participant TF-IDF cosine similarities...")

    # Build per-participant message lists
    train_by_pid = {}
    for item in train_data:
        pid = item["response_id"]
        train_by_pid.setdefault(pid, []).append(item["input_message"])

    test_by_pid = {}
    for item in test_data:
        pid = item["response_id"]
        test_by_pid.setdefault(pid, []).append(item["input_message"])

    # Only look at participants present in both splits
    common_pids = set(train_by_pid.keys()) & set(test_by_pid.keys())
    print(f"  Participants in both splits: {len(common_pids)}")

    max_sims = []
    for pid in sorted(common_pids):
        profile_msgs = train_by_pid[pid]
        test_msgs = test_by_pid[pid]

        all_msgs = profile_msgs + test_msgs
        vectorizer = TfidfVectorizer(max_features=5000)
        tfidf_matrix = vectorizer.fit_transform(all_msgs)

        profile_vecs = tfidf_matrix[: len(profile_msgs)]
        test_vecs = tfidf_matrix[len(profile_msgs) :]

        sim_matrix = cosine_similarity(profile_vecs, test_vecs)
        max_sim = float(sim_matrix.max())
        max_sims.append(max_sim)

    max_sims = np.array(max_sims)

    # Summary stats
    print(f"  Max cosine similarity distribution (n={len(max_sims)}):")
    print(f"    mean={max_sims.mean():.4f}  median={np.median(max_sims):.4f}  "
          f"min={max_sims.min():.4f}  max={max_sims.max():.4f}")

    n_high = int((max_sims >= 0.9).sum())
    pct_high = (n_high / len(max_sims) * 100) if len(max_sims) else 0

    results.append({
        "Check_Type": "PP TF-IDF cosine similarity",
        "Description": (
            f"Per-participant max cosine sim between profile & test messages "
            f"(mean={max_sims.mean():.3f}, median={np.median(max_sims):.3f})"
        ),
        "N_Overlapping": n_high,
        "Total_Checked": len(max_sims),
        "Pct_Overlap": round(pct_high, 2),
        "Risk_Level": _risk_level(pct_high),
    })

    # --- Plot histogram ---
    fig, ax = plt.subplots(figsize=(11.5, 6.8), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.hist(max_sims, bins=30, edgecolor="#0173B2", linewidth=1.5,
            alpha=0.82, color="#0173B2")
    ax.axvline(max_sims.mean(), color="#E02020", linestyle="--", linewidth=2.2,
               label=f"Mean = {max_sims.mean():.3f}")
    ax.axvline(np.median(max_sims), color="#DE8F05", linestyle="--", linewidth=2.2,
               label=f"Median = {np.median(max_sims):.3f}")
    ax.set_xlabel("Max Cosine Similarity (per participant)",
                  fontsize=16, fontweight="bold")
    ax.set_ylabel("Count", fontsize=16, fontweight="bold")
    ax.set_title("Distribution of Max TF-IDF Cosine Similarity\n"
                 "Between PP Profile and Test Messages",
                 fontsize=18, fontweight="bold", pad=14)
    ax.grid(axis="y", alpha=0.12, color="#4D4D4D",
            linestyle="--", linewidth=0.7)
    ax.tick_params(axis="both", labelsize=14, width=1.8,
                   length=6, direction="out")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.legend(frameon=False, prop={"weight": "bold", "size": 13})

    save_figure(fig, figures_path("profile_test_similarity_distribution"))

    return results


# ---------------------------------------------------------------------------
# Check 3 -- Cross-split message text overlap (participant split)
# ---------------------------------------------------------------------------

def check_cross_split_message_overlap(results: list) -> list:
    """Count unique input_message texts appearing in both train and test."""
    print("\n=== Check 3: Cross-split message text overlap (participant split) ===")

    train_data, test_data = load_canonical_data("7030", "participant")

    train_msgs = {item["input_message"] for item in train_data}
    test_msgs = {item["input_message"] for item in test_data}

    overlap = train_msgs & test_msgs
    n_overlap = len(overlap)
    n_unique_test = len(test_msgs)
    pct = (n_overlap / n_unique_test * 100) if n_unique_test else 0

    print(f"  Unique messages in train: {len(train_msgs)}")
    print(f"  Unique messages in test : {n_unique_test}")
    print(f"  Messages in both splits : {n_overlap} ({pct:.1f}%)")
    print("  (Expected -- participant split shares messages across participants)")

    results.append({
        "Check_Type": "Cross-split message text overlap",
        "Description": (
            "Same input_message text in both train & test (expected with "
            "participant-level split; not a label leak)"
        ),
        "N_Overlapping": n_overlap,
        "Total_Checked": n_unique_test,
        "Pct_Overlap": round(pct, 2),
        "Risk_Level": "Low",
    })
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("Label Leakage Analysis")
    print("=" * 65)

    results = []

    results = check_few_shot_overlap(results)
    results = check_digital_twin_overlap(results)
    results = check_cross_split_message_overlap(results)

    # Save CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(PROJECT_ROOT, "revision", "figures", "label_leakage_analysis.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Print summary
    print("\n--- Summary ---")
    print(df.to_string(index=False))
    print("\nDone.")


if __name__ == "__main__":
    main()
