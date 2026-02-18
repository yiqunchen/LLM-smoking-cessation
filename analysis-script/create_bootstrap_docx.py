"""
Create Word document with bootstrap analysis results tables.
Separate tables for Digital Twin, Few-shot, and Zero-shot methods.
"""

import pandas as pd
import numpy as np
import json
import os
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from sklearn.metrics import cohen_kappa_score, accuracy_score

# Rating mappings
RATING_MAPS = {
    'content': {'Very poor': 1, 'Poor': 2, 'Acceptable': 3, 'Good': 4, 'Very good': 5},
    'coping': {
        'Not at all helpful': 1, 'Somewhat helpful': 2, 'Moderately helpful': 3,
        'Very helpful': 4, 'Extremely helpful': 5, 'Not Helpful': 1
    },
    'quitting': {
        'Not at all helpful': 1, 'Somewhat helpful': 2, 'Moderately helpful': 3,
        'Very helpful': 4, 'Extremely helpful': 5, 'Not Helpful': 1
    }
}

DOMAINS = ['content', 'coping', 'quitting']

MODELS = {
    'GPT-4o-mini': 'results_manuscript_gpt-4o-mini',
    'GPT-5': 'results_manuscript_gpt-5',
    'DeepSeek-R1': 'results_manuscript_deepseek_deepseek-r1-0528',
    'Grok-4-Fast': 'results_manuscript_x-ai_grok-4-fast',
    'Gemini-2.5-Pro': 'results_manuscript_gemini-2.5-pro'
}

# Method configurations
METHOD_CONFIGS = {
    'Zero-shot': 'generic_llm_1_zero_shot.json',
    'Few-shot': 'generic_llm_3_few_shot.json',
    'Digital Twin (CBT/ACT 70/30)': 'digital_twin_4_cbtact_7030.json',
}


def load_results(results_path):
    """Load evaluation results and convert to DataFrame."""
    if not os.path.exists(results_path):
        return None

    with open(results_path, 'r') as f:
        results = json.load(f)

    rows = []
    for item in results.values():
        if item.get("predicted_content") != "ERROR":
            rows.append(item)

    if not rows:
        return None

    df = pd.DataFrame(rows)

    for domain in DOMAINS:
        if f'ground_truth_{domain}' in df.columns and f'predicted_{domain}' in df.columns:
            df[f'gt_{domain}_num'] = df[f'ground_truth_{domain}'].map(RATING_MAPS[domain])
            df[f'pred_{domain}_num'] = df[f'predicted_{domain}'].map(RATING_MAPS[domain])

    return df


def map_to_directional(value):
    """Map 1-5 rating to directional: 0=low (1,2), 1=neutral (3), 2=high (4,5)"""
    if value <= 2:
        return 0
    elif value == 3:
        return 1
    else:
        return 2


def aggregate_to_message_level(df, domain):
    """Aggregate predictions to message level."""
    gt_col = f'gt_{domain}_num'
    pred_col = f'pred_{domain}_num'

    if gt_col not in df.columns or pred_col not in df.columns:
        return None

    valid_mask = df[gt_col].notna() & df[pred_col].notna()
    valid_df = df[valid_mask].copy()

    message_df = valid_df.groupby('input_message').agg({
        gt_col: ['mean', 'count'],
        pred_col: 'mean'
    }).reset_index()

    message_df.columns = ['message', 'mean_gt', 'n_ratings', 'mean_pred']
    message_df['gt_rounded'] = message_df['mean_gt'].round().astype(int)
    message_df['pred_rounded'] = message_df['mean_pred'].round().astype(int)
    message_df['gt_directional'] = message_df['gt_rounded'].apply(map_to_directional)
    message_df['pred_directional'] = message_df['pred_rounded'].apply(map_to_directional)

    return message_df


def bootstrap_metrics(df_individual, n_bootstrap=1000):
    """Calculate bootstrap CI for key metrics.

    Uses INDIVIDUAL-level for both point estimates AND bootstrap CIs.
    This matches the main figures (n=274 individual ratings).
    Bootstrap resamples individual ratings to compute CIs.
    """
    gt = df_individual['gt_num'].values
    pred = df_individual['pred_num'].values
    gt_dir = df_individual['gt_dir'].values
    pred_dir = df_individual['pred_dir'].values
    n = len(gt)

    # Point estimates at INDIVIDUAL level (matches main figures)
    acc = accuracy_score(gt, pred)
    dir_acc = accuracy_score(gt_dir, pred_dir)
    kappa = cohen_kappa_score(gt, pred)

    # Bootstrap at INDIVIDUAL level
    acc_boots, dir_acc_boots, kappa_boots = [], [], []

    np.random.seed(42)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        acc_boots.append(accuracy_score(gt[idx], pred[idx]))
        dir_acc_boots.append(accuracy_score(gt_dir[idx], pred_dir[idx]))
        kappa_boots.append(cohen_kappa_score(gt[idx], pred[idx]))

    def get_ci(boots):
        return (np.percentile(boots, 2.5), np.percentile(boots, 97.5))

    return {
        'accuracy': (acc, get_ci(acc_boots)),
        'dir_accuracy': (dir_acc, get_ci(dir_acc_boots)),
        'kappa': (kappa, get_ci(kappa_boots)),
    }


def compute_all_bootstrap_results(base_dir):
    """Compute bootstrap results for all methods, models, and domains."""
    all_results = {}

    for method_name, method_file in METHOD_CONFIGS.items():
        all_results[method_name] = []

        for model_name, model_dir in MODELS.items():
            result_path = os.path.join(base_dir, model_dir, method_file)
            df = load_results(result_path)

            if df is None:
                continue

            for domain in DOMAINS:
                # Prepare individual-level data
                gt_col = f'gt_{domain}_num'
                pred_col = f'pred_{domain}_num'

                if gt_col not in df.columns or pred_col not in df.columns:
                    continue

                valid_mask = df[gt_col].notna() & df[pred_col].notna()
                df_valid = df[valid_mask].copy()

                if len(df_valid) < 10:
                    continue

                # Rename columns for bootstrap function
                df_valid['gt_num'] = df_valid[gt_col]
                df_valid['pred_num'] = df_valid[pred_col]
                df_valid['gt_dir'] = df_valid['gt_num'].apply(map_to_directional)
                df_valid['pred_dir'] = df_valid['pred_num'].apply(map_to_directional)

                metrics = bootstrap_metrics(df_valid)
                n = len(df_valid)  # Report individual-level n

                all_results[method_name].append({
                    'Model': model_name,
                    'Domain': domain.capitalize(),
                    'N': n,
                    'Accuracy': metrics['accuracy'],
                    'Dir. Accuracy': metrics['dir_accuracy'],
                    'Kappa': metrics['kappa'],
                })

    return all_results


def create_method_table(doc, method_name, results):
    """Create a table for one method type."""
    doc.add_heading(method_name, level=2)

    table = doc.add_table(rows=1, cols=5)
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    headers = ['Model', 'Domain', 'Accuracy [95% CI]', 'Dir. Acc. [95% CI]', 'Kappa [95% CI]']
    header_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        header_cells[i].text = h
        header_cells[i].paragraphs[0].runs[0].bold = True
        header_cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    for r in results:
        row = table.add_row()
        cells = row.cells
        cells[0].text = r['Model']
        cells[1].text = r['Domain']

        # Format metrics with CI
        for i, metric in enumerate(['Accuracy', 'Dir. Accuracy', 'Kappa']):
            val, (ci_l, ci_u) = r[metric]
            if np.isnan(val):
                cells[i+2].text = 'N/A'
            else:
                cells[i+2].text = f"{val:.3f} [{ci_l:.3f}, {ci_u:.3f}]"
            cells[i+2].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

        cells[0].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cells[1].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph()


def create_all_results_docx():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'figures')

    print("Computing bootstrap results for all methods...")
    all_results = compute_all_bootstrap_results(base_dir)

    # Create Word document
    doc = Document()

    title = doc.add_heading('Bootstrap Analysis Results (95% CI)', level=1)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph('Message-level bootstrap with n=1000 resamples. '
                      'Metrics computed on mean ratings per message.')
    doc.add_paragraph()

    # Create separate table for each method
    for method_name in ['Digital Twin (CBT/ACT 70/30)', 'Few-shot', 'Zero-shot']:
        if method_name in all_results and all_results[method_name]:
            create_method_table(doc, method_name, all_results[method_name])

    # Add summary comparison
    doc.add_heading('Summary: Best Kappa by Method', level=2)

    summary_table = doc.add_table(rows=1, cols=4)
    summary_table.style = 'Table Grid'

    header_cells = summary_table.rows[0].cells
    for i, h in enumerate(['Method', 'Best Model', 'Best Domain', 'Kappa [95% CI]']):
        header_cells[i].text = h
        header_cells[i].paragraphs[0].runs[0].bold = True
        header_cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    for method_name in ['Digital Twin (CBT/ACT 70/30)', 'Few-shot', 'Zero-shot']:
        if method_name not in all_results or not all_results[method_name]:
            continue

        # Find best kappa
        best = max(all_results[method_name], key=lambda x: x['Kappa'][0])
        row = summary_table.add_row()
        cells = row.cells
        cells[0].text = method_name
        cells[1].text = best['Model']
        cells[2].text = best['Domain']
        val, (ci_l, ci_u) = best['Kappa']
        cells[3].text = f"{val:.3f} [{ci_l:.3f}, {ci_u:.3f}]"

        for cell in cells:
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Save document
    output_path = os.path.join(output_dir, 'bootstrap_results_by_method.docx')
    doc.save(output_path)
    print(f"\nSaved: {output_path}")

    return output_path


if __name__ == '__main__':
    create_all_results_docx()
