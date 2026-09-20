#!/usr/bin/env python
"""
Cost and latency analysis for the LLM smoking-cessation revision.

Reconstructs representative prompts for each method, estimates input token
counts via a single tiktoken encoder for cross-model consistency, applies
OpenRouter pricing captured at experiment time, and reports per-call,
per-participant, and total-study cost.

Outputs
-------
- revision/figures/cost_latency_analysis.csv
- revision/figures/cost_per_participant.png / .pdf

Usage
-----
    uv run python analysis-script/cost_latency_analysis.py
"""

import os
import sys
import json

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from revision_utils import (
    MODEL_CONFIGS,
    METHOD_CONFIGS,
    COLORS,
    load_canonical_data,
    save_figure,
    figures_path,
    PROJECT_ROOT,
)
from prompt_config import (
    generate_zero_shot_prompt,
    generate_zero_shot_feature_select_prompt,
    generate_few_shot_prompt,
    generate_few_shot_feature_select_prompt,
    generate_continuous_rating_prompt,
    generate_digital_twin_prompt,
    generate_hybrid_rf_digital_twin_prompt,
    prepare_few_shot_examples,
)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import tiktoken

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# OpenRouter prices captured when the experiments were run
# (USD per 1 M tokens; accurate as of January 2026)
PRICING = {
    "GPT-4o-mini":    {"input": 0.15,  "output": 0.60},
    "GPT-5":          {"input": 1.25,  "output": 10.00},
    "DeepSeek-R1":    {"input": 0.55,  "output": 2.19},
    "Grok-4-Fast":    {"input": 0.20,  "output": 0.50},
    "Gemini-2.5-Pro": {"input": 1.25,  "output": 10.00},
}

# Assumed output tokens per LLM call (JSON response)
OUTPUT_TOKENS = 200

# Number of test items (total LLM calls per model x method)
N_TEST = 274

# Messages per participant (for per-participant cost)
MSGS_PER_PARTICIPANT = 10

# Map method file keys -> prompt generator functions
METHOD_TO_GENERATOR = {
    "generic_llm_1_zero_shot.json":        generate_zero_shot_prompt,
    "generic_llm_2_zero_shot_select.json": generate_zero_shot_feature_select_prompt,
    "generic_llm_3_few_shot.json":         generate_few_shot_prompt,
    "generic_llm_4_few_shot_select.json":  generate_few_shot_feature_select_prompt,
    "generic_llm_5_continuous.json":       generate_continuous_rating_prompt,
    "digital_twin_4_cbtact_7030.json":     generate_digital_twin_prompt,
    "hybrid":                              generate_hybrid_rf_digital_twin_prompt,
}

METHOD_DISPLAY_OVERRIDES = {
    "Digital Twin": "PP",
    "Hybrid RF+DT": "Hybrid RF+PP",
}


# ---------------------------------------------------------------------------
# Token counting helpers
# ---------------------------------------------------------------------------

# Use a single tokenizer approximation across models for consistency.
_token_enc = tiktoken.get_encoding("o200k_base")


def count_tokens(text, model_display):
    """Return estimated token count for *text*.

    This uses the same tiktoken encoder for every model as a consistent
    approximation of prompt length. It is exact only for OpenAI-style
    tokenization and remains approximate for non-OpenAI providers.
    """
    del model_display  # kept for a stable public function signature
    return len(_token_enc.encode(text))


# ---------------------------------------------------------------------------
# Representative prompt construction
# ---------------------------------------------------------------------------

def _build_profile_messages(train_data, response_id, fallback_records):
    """Build up to 7 profile messages for a participant."""
    same_participant = [d for d in train_data if d["response_id"] == response_id]
    if len(same_participant) < 1:
        same_participant = fallback_records[:7]
    return [
        {
            "input_message": d["input_message"],
            "ratings": d.get("ratings", {}),
        }
        for d in same_participant[:7]
    ]


def _find_hybrid_sample(digital_test):
    """Return a digital-twin test item that has saved RF predictions, if possible."""
    rf_pred_path = os.path.join(
        PROJECT_ROOT,
        "results_manuscript_hybrid_rf_grok4",
        "rf_predictions_all_features.json",
    )
    try:
        with open(rf_pred_path) as f:
            rf_preds = json.load(f)
    except Exception:
        rf_preds = {}

    for item in digital_test:
        if item["response_id"] in rf_preds:
            return json.loads(json.dumps(item, default=str))
    return json.loads(json.dumps(digital_test[0], default=str))


def build_representative_prompts(participant_train, participant_test, digital_train, digital_test):
    """Build one representative prompt per method.

    Returns dict  method_file -> prompt_text (str).
    """
    participant_sample = json.loads(json.dumps(participant_test[0], default=str))
    digital_sample = json.loads(json.dumps(digital_test[0], default=str))
    hybrid_sample = _find_hybrid_sample(digital_test)

    # Prepare few-shot examples (required before calling few-shot generators)
    train_dict = {i: item for i, item in enumerate(participant_train)}
    prepare_few_shot_examples(train_dict)

    digital_profile_messages = _build_profile_messages(
        digital_train,
        digital_sample["response_id"],
        digital_train,
    )

    prompts = {}
    for method_file, gen_fn in METHOD_TO_GENERATOR.items():
        if method_file == "digital_twin_4_cbtact_7030.json":
            data_item = json.loads(json.dumps(digital_sample, default=str))
            data_item["profile_messages"] = digital_profile_messages
        elif method_file == "hybrid":
            data_item = json.loads(json.dumps(hybrid_sample, default=str))
            data_item["profile_messages"] = _build_profile_messages(
                digital_train,
                hybrid_sample["response_id"],
                digital_train,
            )
        else:
            data_item = json.loads(json.dumps(participant_sample, default=str))

        try:
            prompt_text = gen_fn(data_item)
        except Exception as exc:
            print(f"  WARNING: could not generate prompt for {method_file}: {exc}")
            prompt_text = ""

        prompts[method_file] = prompt_text

    return prompts


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("COST ANALYSIS")
    print("=" * 70)

    # Load canonical data needed for prompt reconstruction
    participant_train, participant_test = load_canonical_data("7030", "participant")
    digital_train, digital_test = load_canonical_data("7030", "digital_twin")
    print(
        f"Loaded participant split: {len(participant_train)} train, {len(participant_test)} test.\n"
        f"Loaded digital-twin split: {len(digital_train)} train, {len(digital_test)} test."
    )

    # Build representative prompts (one per method)
    prompts = build_representative_prompts(
        participant_train,
        participant_test,
        digital_train,
        digital_test,
    )
    print(f"Generated representative prompts for {len(prompts)} methods.\n")

    # ---- Compute costs for every (model, method) pair ----
    rows = []
    for model_id, model_cfg in MODEL_CONFIGS.items():
        model_name = model_cfg["display"]
        pricing = PRICING.get(model_name)
        if pricing is None:
            print(f"  Skipping {model_name} (no pricing data)")
            continue

        for method_file, method_cfg in METHOD_CONFIGS.items():
            method_name = METHOD_DISPLAY_OVERRIDES.get(
                method_cfg["display"],
                method_cfg["display"],
            )
            prompt_text = prompts.get(method_file, "")
            if not prompt_text:
                continue

            input_tokens = count_tokens(prompt_text, model_name)
            output_tokens = OUTPUT_TOKENS

            cost_input  = input_tokens  * pricing["input"]  / 1_000_000
            cost_output = output_tokens * pricing["output"] / 1_000_000
            cost_per_call = cost_input + cost_output

            cost_per_participant = cost_per_call * MSGS_PER_PARTICIPANT
            cost_per_1000_messages = cost_per_call * 1000
            total_cost = cost_per_call * N_TEST

            rows.append({
                "Model":               model_name,
                "Method":              method_name,
                "Input_Tokens":        input_tokens,
                "Output_Tokens":       output_tokens,
                "Cost_Per_Call":       round(cost_per_call, 6),
                "Cost_Per_Participant": round(cost_per_participant, 6),
                "Cost_Per_1000_Messages": round(cost_per_1000_messages, 4),
                "Total_Cost":          round(total_cost, 4),
            })

            print(
                f"  {model_name:18s} | {method_name:22s} | "
                f"in={input_tokens:6d}  out={output_tokens:4d} | "
                f"$/call={cost_per_call:.6f}  $/part={cost_per_participant:.5f}  "
                f"total=${total_cost:.4f}"
            )

    df = pd.DataFrame(rows)

    # ---- Save CSV ----
    csv_path = os.path.join(PROJECT_ROOT, "revision", "figures", "cost_latency_analysis.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}")

    # ---- Grouped bar chart: cost per participant ----
    _plot_cost_per_participant(df)

    print("\nDone.")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_cost_per_participant(df):
    """Grouped bar chart: methods on x-axis, grouped by model colors."""
    models  = list(dict.fromkeys(df["Model"]))   # preserve insertion order
    methods = [
        method for method in dict.fromkeys(df["Method"])
        if method != "Zero-shot (w/ prob)"
    ]

    n_models  = len(models)
    n_methods = len(methods)
    x = np.arange(n_methods)
    width = 0.8 / n_models  # bar width

    fig, ax = plt.subplots(figsize=(14, 7))

    method_labels = {
        "Zero-shot (all)": "Zero-shot\n(all)",
        "Zero-shot (select)": "Zero-shot\n(select)",
        "Few-shot (all)": "Few-shot\n(all)",
        "Few-shot (select)": "Few-shot\n(select)",
        "Zero-shot (w/ prob)": "Prob.\nratings",
        "PP": "PP",
        "Hybrid RF+PP": "Hybrid\nRF+PP",
    }

    for j, model in enumerate(models):
        subset = df[df["Model"] == model]
        vals = []
        for method in methods:
            row = subset[subset["Method"] == method]
            vals.append(row["Cost_Per_1000_Messages"].values[0] if len(row) else 0)
        offset = (j - n_models / 2 + 0.5) * width
        ax.bar(
            x + offset, vals, width,
            label=model, color=COLORS.get(model, "#4D4D4D"),
            edgecolor="black", linewidth=0.4,
        )

    ax.set_xlabel("Method", fontsize=13, fontweight='bold')
    ax.set_ylabel("Estimated Cost (USD per 1,000 messages)", fontsize=13, fontweight='bold')
    ax.set_title("Estimated API Cost by Model and Method (1,000 messages)", fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([method_labels.get(m, m) for m in methods], fontsize=11, fontweight='bold')
    ax.legend(
        title="Model", fontsize=9, title_fontsize=10,
        loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0,
    )
    ax.set_ylim(0, df["Cost_Per_1000_Messages"].max() * 1.08)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"${x:,.0f}" if x >= 1 else f"${x:.1f}"))
    ax.grid(axis="y", color="#4D4D4D", alpha=0.12)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
    fig.tight_layout()

    out_path = figures_path("cost_per_participant")
    save_figure(fig, out_path)
    print(f"Saved figure: {out_path}.png / .pdf")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
