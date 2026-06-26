#!/usr/bin/env python3
"""
Ablation study for PP prompt components (Reviewer R3, R4).

Disentangles the contribution of participant profile (demographics),
rating history, and message text to PP performance.

Three ablation conditions:
  1. profile-only  -- metadata + message, NO history
  2. history-only  -- history ratings + message, NO metadata
  3. message-only  -- just the message text

Run:
    uv run python analysis-script/ablation_study.py           # dry-run (default)
    uv run python analysis-script/ablation_study.py --run     # actual API execution (future)
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'analysis-script'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from revision_utils import (
    load_canonical_data,
    load_results_file,
    compute_all_metrics,
    RATING_MAPS,
    DOMAINS,
    MODEL_CONFIGS,
    COLORS,
    save_figure,
    figures_path,
    FIGURES_DIR,
    DPI,
)
from prompt_config import generate_digital_twin_prompt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ABLATION_CONDITIONS = ['message-only', 'profile-only', 'history-only', 'full-PP']
CONDITION_LABELS = {
    'message-only': 'Message Only',
    'profile-only': 'Profile Only\n(Demographics)',
    'history-only': 'History Only\n(Past Ratings)',
    'full-PP': 'Full PP',
}

# Top-3 models for the ablation study
ABLATION_MODELS = {
    'gpt-5': {
        'display': 'GPT-5',
        'input_price_per_1m': 2.00,
        'output_price_per_1m': 8.00,
    },
    'deepseek_deepseek-r1-0528': {
        'display': 'DeepSeek-R1',
        'input_price_per_1m': 0.55,
        'output_price_per_1m': 2.19,
    },
    'gemini-2.5-pro': {
        'display': 'Gemini-2.5-Pro',
        'input_price_per_1m': 2.00,
        'output_price_per_1m': 8.00,
    },
}

# Token estimation constants (empirical averages)
EST_TOKENS_MESSAGE_ONLY = 450      # system prompt + message + output format
EST_TOKENS_PROFILE_ONLY = 900      # system + metadata (~26 fields) + message + format
EST_TOKENS_HISTORY_ONLY = 1100     # system + ~2 profile messages with ratings + message + format
EST_TOKENS_FULL_DT = 1500          # system + metadata + history + message + format
EST_OUTPUT_TOKENS = 200            # JSON output + explanation

CONDITION_TOKEN_ESTIMATES = {
    'message-only': EST_TOKENS_MESSAGE_ONLY,
    'profile-only': EST_TOKENS_PROFILE_ONLY,
    'history-only': EST_TOKENS_HISTORY_ONLY,
    'full-PP': EST_TOKENS_FULL_DT,
}

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results_ablation')
REVISION_FIGURES_DIR = os.path.join(PROJECT_ROOT, 'revision', 'figures')


# ---------------------------------------------------------------------------
# Prompt generators for ablation conditions
# ---------------------------------------------------------------------------

def _format_profile_messages(profile_messages: list) -> str:
    """Format profile messages with their ratings (replicates prompt_config helper)."""
    out = []
    for pm in profile_messages:
        msg = pm.get('input_message', '')
        if not msg:
            continue
        ratings = pm.get('ratings', {}) or {}
        if not isinstance(ratings, dict):
            ratings = {}
        ratings_lines = []
        for key in ['content', 'design', 'coping', 'quitting']:
            if key in ratings and ratings[key] is not None:
                ratings_lines.append(f"{key}: {ratings[key]}")
        out.append(f"Past message:\n{msg}\nRatings:\n" + "\n".join(ratings_lines) + "\n\n---\n")
    return "".join(out)


def generate_profile_only_prompt(data: dict) -> str:
    """Ablation: metadata (demographics) + message, NO history.

    Isolates the contribution of demographic matching.
    """
    # Build metadata text from demographics only
    metadata_text = ""
    if 'metadata' in data and data['metadata']:
        for key, value in data['metadata'].items():
            if value is not None and str(value) != 'nan':
                metadata_text += f"- {key}: {value}\n"

    # NOTE: No profile_messages section

    prompt = f"""
You are an AI assistant simulating this participant. Your task is to predict how the participant will rate a new smoking-cessation support message, using their demographic profile.
Base your prediction on the participant's characteristics and how someone with this profile would typically respond.
Be sure to carefully follow all provided Instructions for formatting your answer.
---
Instructions:
### RATING DIMENSIONS
1. **content** -- How would you rate the content (that is, the words and meaning) of this message?
2. **design** -- How would you rate the design (that is, how the message looks) of this message?
3. **coping** -- How helpful would this message be to support you in coping with a smoking urge or craving?
4. **quitting** -- How helpful would this message be to support you in quitting or reducing smoking?

### Allowed rating categories
**content / design** -> Very poor . Poor . Acceptable . Good . Very good
**coping / quitting** -> Not at all helpful . Somewhat helpful . Moderately helpful . Very helpful . Extremely helpful



### INPUTS

Here is the message provided to the participant to be rated:

\\"{data['input_message']}\\"

Participant demographics:
{metadata_text}
---

### OUTPUT FORMAT
Return **exactly** this JSON object:

{{{{
  "response_id": "{data['response_id']}",
  "predicted_content": "Very poor/Poor/Acceptable/Good/Very good",
  "predicted_design": "Very poor/Poor/Acceptable/Good/Very good",
  "predicted_coping": "Not at all helpful/Somewhat helpful/Moderately helpful/Very helpful/Extremely helpful",
  "predicted_quitting": "Not at all helpful/Somewhat helpful/Moderately helpful/Very helpful/Extremely helpful",
  "explanation": "<= 2 sentences per dimension reflecting the participant's likely view."
}}}}
"""
    return prompt


def generate_history_only_prompt(data: dict) -> str:
    """Ablation: history ratings + message, NO metadata/demographics.

    Isolates the contribution of rating history.
    """
    # Build history text from profile_messages only -- no demographics
    history_text = ""
    if data.get('profile_messages'):
        history_text = _format_profile_messages(data['profile_messages'])
    else:
        history_text = "(No prior rating history available for this participant.)\n"

    prompt = f"""
You are an AI assistant simulating this participant. Your task is to predict how the participant will rate a new smoking-cessation support message, using ONLY their past message ratings shown below.
Base your prediction on how similar the new message is to the participant's previously rated messages. Remain consistent with the participant's prior rating patterns.
Be sure to carefully follow all provided Instructions for formatting your answer.
---
Instructions:
### RATING DIMENSIONS
1. **content** -- How would you rate the content (that is, the words and meaning) of this message?
2. **design** -- How would you rate the design (that is, how the message looks) of this message?
3. **coping** -- How helpful would this message be to support you in coping with a smoking urge or craving?
4. **quitting** -- How helpful would this message be to support you in quitting or reducing smoking?

### Allowed rating categories
**content / design** -> Very poor . Poor . Acceptable . Good . Very good
**coping / quitting** -> Not at all helpful . Somewhat helpful . Moderately helpful . Very helpful . Extremely helpful



### INPUTS

Here is the message provided to the participant to be rated:

\\"{data['input_message']}\\"

Participant's past message ratings:
{history_text}
---

### OUTPUT FORMAT
Return **exactly** this JSON object:

{{{{
  "response_id": "{data['response_id']}",
  "predicted_content": "Very poor/Poor/Acceptable/Good/Very good",
  "predicted_design": "Very poor/Poor/Acceptable/Good/Very good",
  "predicted_coping": "Not at all helpful/Somewhat helpful/Moderately helpful/Very helpful/Extremely helpful",
  "predicted_quitting": "Not at all helpful/Somewhat helpful/Moderately helpful/Very helpful/Extremely helpful",
  "explanation": "<= 2 sentences per dimension reflecting the participant's likely view based on rating patterns."
}}}}
"""
    return prompt


def generate_message_only_prompt(data: dict) -> str:
    """Ablation: message text only, NO participant context.

    Measures pure message-level prediction without any personalisation.
    """
    prompt = f"""
You are an AI assistant evaluating a smoking-cessation support message. Your task is to predict how a typical participant would rate this message.
You have NO information about the specific participant. Base your prediction solely on the message content.
Be sure to carefully follow all provided Instructions for formatting your answer.
---
Instructions:
### RATING DIMENSIONS
1. **content** -- How would you rate the content (that is, the words and meaning) of this message?
2. **design** -- How would you rate the design (that is, how the message looks) of this message?
3. **coping** -- How helpful would this message be to support you in coping with a smoking urge or craving?
4. **quitting** -- How helpful would this message be to support you in quitting or reducing smoking?

### Allowed rating categories
**content / design** -> Very poor . Poor . Acceptable . Good . Very good
**coping / quitting** -> Not at all helpful . Somewhat helpful . Moderately helpful . Very helpful . Extremely helpful



### INPUTS

Here is the message to be rated:

\\"{data['input_message']}\\"

---

### OUTPUT FORMAT
Return **exactly** this JSON object:

{{{{
  "response_id": "{data['response_id']}",
  "predicted_content": "Very poor/Poor/Acceptable/Good/Very good",
  "predicted_design": "Very poor/Poor/Acceptable/Good/Very good",
  "predicted_coping": "Not at all helpful/Somewhat helpful/Moderately helpful/Very helpful/Extremely helpful",
  "predicted_quitting": "Not at all helpful/Somewhat helpful/Moderately helpful/Very helpful/Extremely helpful",
  "explanation": "<= 2 sentences per dimension."
}}}}
"""
    return prompt


# Map condition names to prompt generators
PROMPT_GENERATORS = {
    'message-only': generate_message_only_prompt,
    'profile-only': generate_profile_only_prompt,
    'history-only': generate_history_only_prompt,
    'full-PP': generate_digital_twin_prompt,
}


# ---------------------------------------------------------------------------
# Data preparation: attach profile messages from train split
# ---------------------------------------------------------------------------

def prepare_test_data_with_profiles() -> list:
    """Load digital twin 7030 test data and attach profile messages from train."""
    train_data, test_data = load_canonical_data('7030', 'digital_twin')

    # Build map: response_id -> list of prior messages with ratings
    response_id_to_profile = {}
    for it in train_data:
        if not isinstance(it, dict):
            continue
        rid = it.get('response_id')
        if rid is None:
            continue
        profile_entry = {
            'input_message': it.get('input_message'),
            'ratings': it.get('ratings', {}),
        }
        response_id_to_profile.setdefault(rid, []).append(profile_entry)

    # Attach profile_messages to test items
    for it in test_data:
        if not isinstance(it, dict):
            continue
        rid = it.get('response_id')
        if rid and rid in response_id_to_profile:
            it['profile_messages'] = response_id_to_profile[rid]
        else:
            it['profile_messages'] = []

    return test_data


# ---------------------------------------------------------------------------
# Dry-run: cost estimation
# ---------------------------------------------------------------------------

def estimate_costs(test_data: list) -> None:
    """Print estimated API call counts and costs for ablation study."""
    n_test = len(test_data)
    n_models = len(ABLATION_MODELS)
    # Only 3 ablation conditions (message-only, profile-only, history-only);
    # full-PP results already exist.
    n_new_conditions = 3
    total_calls = n_test * n_models * n_new_conditions

    print("=" * 70)
    print("ABLATION STUDY -- DRY RUN COST ESTIMATION")
    print("=" * 70)
    print(f"\nTest set size:       {n_test} items")
    print(f"Models:              {n_models} ({', '.join(m['display'] for m in ABLATION_MODELS.values())})")
    print(f"New conditions:      {n_new_conditions} (message-only, profile-only, history-only)")
    print(f"Total API calls:     {total_calls:,}")
    print()

    # Show a sample prompt for each condition
    sample_item = test_data[0]
    print("-" * 70)
    print("SAMPLE PROMPT LENGTHS (first test item):")
    print("-" * 70)
    for cond_name, gen_func in PROMPT_GENERATORS.items():
        if cond_name == 'full-PP':
            continue  # already have these results
        prompt = gen_func(sample_item)
        char_count = len(prompt)
        est_tokens = char_count // 4  # rough estimate: 1 token ~ 4 chars
        print(f"  {cond_name:20s}: {char_count:,} chars  (~{est_tokens:,} tokens)")
    print()

    # Per-model cost breakdown
    print("-" * 70)
    print(f"{'Model':<20s} {'Condition':<20s} {'Calls':>8s} {'Est Input Tok':>14s} "
          f"{'Est Output Tok':>14s} {'Est Cost ($)':>12s}")
    print("-" * 70)

    grand_total = 0.0
    for model_id, mcfg in ABLATION_MODELS.items():
        model_total = 0.0
        for cond in ['message-only', 'profile-only', 'history-only']:
            calls = n_test
            input_tokens = calls * CONDITION_TOKEN_ESTIMATES[cond]
            output_tokens = calls * EST_OUTPUT_TOKENS
            cost = (
                (input_tokens / 1_000_000) * mcfg['input_price_per_1m']
                + (output_tokens / 1_000_000) * mcfg['output_price_per_1m']
            )
            model_total += cost
            print(f"  {mcfg['display']:<18s} {cond:<20s} {calls:>8,d} "
                  f"{input_tokens:>14,d} {output_tokens:>14,d} ${cost:>10.4f}")
        print(f"  {'':18s} {'SUBTOTAL':<20s} {'':>8s} {'':>14s} {'':>14s} ${model_total:>10.4f}")
        grand_total += model_total
        print()

    print("-" * 70)
    print(f"  {'GRAND TOTAL':<40s} {total_calls:>8,d} {'':>14s} {'':>14s} ${grand_total:>10.4f}")
    print("=" * 70)
    print()
    print("NOTE: Full PP results already exist and do not need re-running.")
    print("      Cost estimates use rough token approximations.")
    print()


# ---------------------------------------------------------------------------
# Run mode (placeholder for future API calls)
# ---------------------------------------------------------------------------

def run_ablation(test_data: list) -> None:
    """Placeholder for actual API execution. NOT IMPLEMENTED YET."""
    print("=" * 70)
    print("ABLATION STUDY -- RUN MODE")
    print("=" * 70)
    print()
    print("[NOT IMPLEMENTED] API calls would go here.")
    print()
    print("When implemented, this will:")
    print("  1. For each model in:", ', '.join(m['display'] for m in ABLATION_MODELS.values()))
    print("  2. For each condition: message-only, profile-only, history-only")
    print(f"  3. Send {len(test_data)} API calls per (model, condition) pair")
    print(f"  4. Save results to: {RESULTS_DIR}/results_ablation_{{model}}_{{condition}}.json")
    print()
    print("To proceed, implement the async API call loop following the pattern in main_eval.py.")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Load existing full-PP results
# ---------------------------------------------------------------------------

def load_full_dt_metrics() -> list:
    """Load existing full PP (cbtact 7030) results for each model."""
    rows = []
    dt_file = 'digital_twin_4_cbtact_7030.json'
    for model_id, mcfg in ABLATION_MODELS.items():
        global_cfg = MODEL_CONFIGS.get(model_id)
        if global_cfg is None:
            continue
        filepath = os.path.join(PROJECT_ROOT, global_cfg['dir'], dt_file)
        df = load_results_file(filepath)
        if df is None:
            continue
        for domain in DOMAINS:
            gt_col = f'gt_{domain}_num'
            pred_col = f'pred_{domain}_num'
            if gt_col not in df.columns or pred_col not in df.columns:
                continue
            valid = df[gt_col].notna() & df[pred_col].notna()
            gt = df.loc[valid, gt_col].values
            pred = df.loc[valid, pred_col].values
            rids = df.loc[valid, 'response_id'].values if 'response_id' in df.columns else None
            metrics = compute_all_metrics(gt, pred, rids)
            rows.append({
                'Model': mcfg['display'],
                'Condition': 'full-PP',
                'Domain': domain.capitalize(),
                'Data_Source': 'observed',
                **metrics,
            })
    return rows


# ---------------------------------------------------------------------------
# Load ablation results
# ---------------------------------------------------------------------------

def load_ablation_results() -> list:
    """Try to load real ablation results; return empty list if none found."""
    rows = []
    for model_id, mcfg in ABLATION_MODELS.items():
        for cond in ['message-only', 'profile-only', 'history-only']:
            model_safe = model_id.replace('/', '-')
            filepath = os.path.join(RESULTS_DIR, f'results_ablation_{model_safe}_{cond}.json')
            df = load_results_file(filepath)
            if df is None:
                continue
            for domain in DOMAINS:
                gt_col = f'gt_{domain}_num'
                pred_col = f'pred_{domain}_num'
                if gt_col not in df.columns or pred_col not in df.columns:
                    continue
                valid = df[gt_col].notna() & df[pred_col].notna()
                gt = df.loc[valid, gt_col].values
                pred = df.loc[valid, pred_col].values
                rids = df.loc[valid, 'response_id'].values if 'response_id' in df.columns else None
                metrics = compute_all_metrics(gt, pred, rids)
                rows.append({
                    'Model': mcfg['display'],
                    'Condition': cond,
                    'Domain': domain.capitalize(),
                    'Data_Source': 'observed',
                    **metrics,
                })
    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_ablation_results(results_df: pd.DataFrame) -> None:
    """Create grouped bar chart: conditions on x-axis, grouped by model, 1x3 domain subplots."""
    plt.style.use('seaborn-v0_8-whitegrid')

    domains = ['Content', 'Coping', 'Quitting']
    conditions_order = ['message-only', 'profile-only', 'history-only', 'full-PP']

    models = results_df['Model'].unique().tolist()
    # Ensure consistent model ordering
    model_order = ['GPT-5', 'DeepSeek-R1', 'Gemini-2.5-Pro']
    models = [m for m in model_order if m in models]
    n_models = len(models)

    # Assign colors from COLORS dict
    model_colors = [COLORS.get(m, '#888888') for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(20, 8), sharey=False)
    fig.suptitle('Ablation Study: PP Component Contributions',
                 fontsize=16, fontweight='bold', y=1.02)

    bar_width = 0.22
    x_base = np.arange(len(conditions_order))

    for ax_idx, domain in enumerate(domains):
        ax = axes[ax_idx]
        domain_df = results_df[results_df['Domain'] == domain]

        for m_idx, model_name in enumerate(models):
            model_df = domain_df[domain_df['Model'] == model_name]
            # Get F1 values for each condition (use F1 as primary metric)
            vals = []
            for cond in conditions_order:
                row = model_df[model_df['Condition'] == cond]
                if len(row) > 0:
                    vals.append(row['F1'].values[0])
                else:
                    vals.append(0.0)

            offset = (m_idx - (n_models - 1) / 2) * bar_width
            bars = ax.bar(x_base + offset, vals, bar_width,
                          label=model_name if ax_idx == 0 else "",
                          color=model_colors[m_idx], edgecolor='white',
                          linewidth=0.5, alpha=0.9)

            # Add value labels on bars
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                            f'{val:.2f}', ha='center', va='bottom', fontsize=7,
                            fontweight='bold')

        ax.set_title(f'{domain}', fontsize=14, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Macro F1' if ax_idx == 0 else '', fontsize=12)
        ax.set_xticks(x_base)
        ax.set_xticklabels([CONDITION_LABELS[c] for c in conditions_order],
                           fontsize=9, ha='center')
        ax.set_ylim(0, max(0.60, domain_df['F1'].max() * 1.25 if len(domain_df) > 0 else 0.60))
        ax.tick_params(axis='y', labelsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=n_models,
               fontsize=12, bbox_to_anchor=(0.5, 0.98), frameon=True,
               fancybox=True, shadow=False)

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    # Save
    out_path = os.path.join(REVISION_FIGURES_DIR, 'ablation_study')
    save_figure(fig, out_path, dpi=DPI)


def save_ablation_csv(results_df: pd.DataFrame) -> str:
    """Save ablation results to CSV."""
    out_path = os.path.join(REVISION_FIGURES_DIR, 'ablation_results.csv')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Select columns for output
    cols = ['Model', 'Condition', 'Domain', 'Data_Source', 'Accuracy', 'F1', 'Kappa', 'QWK']
    available = [c for c in cols if c in results_df.columns]
    results_df[available].to_csv(out_path, index=False, float_format='%.4f')
    print(f"  Saved: {out_path}")
    return out_path


def write_ablation_status(note: str) -> str:
    """Write a no-mock status note for the revision folder."""
    out_path = os.path.join(REVISION_FIGURES_DIR, 'ablation_status.md')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as handle:
        handle.write(note.rstrip() + '\n')
    print(f"  Saved: {out_path}")
    return out_path


def remove_stale_ablation_figure() -> None:
    """Remove any stale ablation figure when no observed comparison data exist."""
    for filename in ['ablation_study.png', 'ablation_study.pdf']:
        path = os.path.join(REVISION_FIGURES_DIR, filename)
        if os.path.exists(path):
            os.remove(path)
            print(f"  Removed stale artifact: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Ablation study for digital twin prompt components (R3/R4)."
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--dry-run', action='store_true', default=True,
                            help='Estimate costs only (default)')
    mode_group.add_argument('--run', action='store_true',
                            help='Execute API calls (NOT YET IMPLEMENTED)')
    parser.add_argument('--plot-only', action='store_true',
                        help='Only generate the ablation figure (skip cost estimation)')
    args = parser.parse_args()

    # If --run is set, dry_run is False
    if args.run:
        args.dry_run = False

    print()
    print("=" * 70)
    print("  ABLATION STUDY: PP Component Contributions")
    print("  Reviewers R3 & R4 -- Disentangling profile, history, message")
    print("=" * 70)
    print()

    # 1. Load and prepare test data
    print("[1/4] Loading digital twin 7030 test data...")
    test_data = prepare_test_data_with_profiles()
    n_with_profiles = sum(1 for it in test_data if it.get('profile_messages'))
    print(f"      Loaded {len(test_data)} test items "
          f"({n_with_profiles} with profile messages attached)")
    print()

    # 2. Dry-run cost estimation
    if not args.plot_only:
        if args.dry_run:
            print("[2/4] Estimating API costs (dry-run mode)...")
            estimate_costs(test_data)
        else:
            print("[2/4] Running ablation API calls...")
            run_ablation(test_data)

    # 3. Collect results for plotting
    print("[3/4] Collecting results for visualisation...")

    # Always load existing full-PP results
    full_dt_rows = load_full_dt_metrics()
    print(f"      Full PP results: {len(full_dt_rows)} rows "
          f"({len(full_dt_rows) // 3 if full_dt_rows else 0} model-domain pairs)")

    # Try loading real ablation results
    ablation_rows = load_ablation_results()

    if ablation_rows:
        print(f"      Real ablation results found: {len(ablation_rows)} rows")
    else:
        print("      No real ablation results found in the repo.")
        print("      Skipping any mock preview generation to keep revision outputs observed-only.")

    # Combine all results
    all_rows = ablation_rows + full_dt_rows
    results_df = pd.DataFrame(all_rows)

    if results_df.empty:
        print("      WARNING: No results to plot. Exiting.")
        return

    print(f"      Total rows for plotting: {len(results_df)}")
    print()

    # 4. Plot and save
    print("[4/4] Writing observed-only ablation outputs...")
    csv_path = save_ablation_csv(results_df)
    if ablation_rows:
        plot_ablation_results(results_df)
    else:
        remove_stale_ablation_figure()
        status_path = write_ablation_status(
            "# Ablation Status\n\n"
            "No observed `message-only`, `profile-only`, or `history-only` ablation outputs "
            "exist in this repo yet.\n\n"
            "Available observed data:\n"
            "- `full-PP` reference rows in `ablation_results.csv`\n"
            "- Saved PP prompt variants already in the repo: "
            "`digital_twin_1_full_7030.json`, `digital_twin_2_select_7030.json`, "
            "`digital_twin_3_feedback_7030.json`, and `digital_twin_4_cbtact_7030.json` "
            "(coverage varies by model; these are not the same as the requested "
            "`message-only` / `profile-only` / `history-only` ablations)\n"
            "- Prompt constructors in `analysis-script/ablation_study.py`\n"
            "- Canonical PP splits in `data_splits/canonical/`\n\n"
            "Under the no-mock-data policy, `ablation_study.png` and `ablation_study.pdf` "
            "are intentionally omitted until real API-run ablation outputs are available."
        )
        print("      No observed message/profile/history ablation results; figure not generated.")

    # Summary
    print()
    print("=" * 70)
    print("OUTPUTS:")
    print(f"  CSV:    {csv_path}")
    if ablation_rows:
        print(f"  Figure: {os.path.join(REVISION_FIGURES_DIR, 'ablation_study.png')}")
        print(f"  Figure: {os.path.join(REVISION_FIGURES_DIR, 'ablation_study.pdf')}")
    else:
        print("  Figure: not generated (no observed ablation condition results in repo)")
        print(f"  Status: {status_path}")
    print("=" * 70)
    print()


if __name__ == '__main__':
    main()
