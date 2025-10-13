import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr

# Add mappings to handle small inconsistencies in LLM output
RATING_MAPS = {
    'content': {'Very poor': 1, 'Poor': 2, 'Acceptable': 3, 'Good': 4, 'Very good': 5},
    'design': {'Very poor': 1, 'Poor': 2, 'Acceptable': 3, 'Good': 4, 'Very good': 5},
    'coping': {'Not at all helpful': 1, 'Somewhat helpful': 2, 'Moderately helpful': 3, 'Very helpful': 4, 'Extremely helpful': 5, 'Not Helpful': 1},
    'quitting': {'Not at all helpful': 1, 'Somewhat helpful': 2, 'Moderately helpful': 3, 'Very helpful': 4, 'Extremely helpful': 5, 'Not Helpful': 1}
}

def load_and_preprocess_data(results_path: str) -> pd.DataFrame:
    """Loads and preprocesses the LLM evaluation results."""
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    rows = []
    for item in results.values():
        if item.get("predicted_content") == "ERROR":
            continue
        rows.append(item)
    
    df = pd.DataFrame(rows)
    
    # Convert predictions and ground truth to numeric
    for domain, rating_map in RATING_MAPS.items():
        df[f'gt_{domain}_num'] = df[f'ground_truth_{domain}'].map(rating_map)
        df[f'pred_{domain}_num'] = df[f'predicted_{domain}'].map(rating_map)

    return df

def calculate_global_metrics(df: pd.DataFrame, domains: list):
    """Calculates global Cohen's Kappa for each domain."""
    print("--- Global Agreement Metrics ---")
    for domain in domains:
        gt_col = f'gt_{domain}_num'
        pred_col = f'pred_{domain}_num'
        
        # Filter out NaNs for fair comparison
        valid_mask = df[gt_col].notna() & df[pred_col].notna()
        gt = df.loc[valid_mask, gt_col]
        pred = df.loc[valid_mask, pred_col]
        
        if len(gt) > 1:
            kappa = cohen_kappa_score(gt, pred)
            print(f"  {domain.title():<10}: Cohen's Kappa = {kappa:.3f}")

def calculate_per_participant_ranking(df: pd.DataFrame, domains: list):
    """Calculates the average per-participant Spearman's Rho."""
    print("\n--- Per-Participant Rank Correlation (Spearman's Rho) ---")
    
    participant_groups = df.groupby('response_id')
    
    results = {domain: [] for domain in domains}
    
    for participant_id, group in participant_groups:
        # Spearman's Rho requires at least 2 data points with variance
        if len(group) < 2:
            continue
            
        for domain in domains:
            gt_col = f'gt_{domain}_num'
            pred_col = f'pred_{domain}_num'
            
            # Ensure there's variance in both truth and prediction to calculate correlation
            if group[gt_col].nunique() > 1 and group[pred_col].nunique() > 1:
                rho, p_val = spearmanr(group[gt_col], group[pred_col])
                if not np.isnan(rho):
                    results[domain].append(rho)
                    
    print("On average, how well does the model rank messages for an individual?")
    for domain, rhos in results.items():
        if rhos:
            avg_rho = np.mean(rhos)
            num_participants = len(rhos)
            print(f"  {domain.title():<10}: Average Spearman's Rho = {avg_rho:.3f} (across {num_participants} participants)")
        else:
            print(f"  {domain.title():<10}: Not enough data with variance to calculate rank correlation.")


def main():
    parser = argparse.ArgumentParser(description="Analyze LLM ranking performance with Kappa and Spearman's Rho.")
    parser.add_argument('results_path', help="Path to the JSON file with evaluation results.")
    args = parser.parse_args()
    
    df = load_and_preprocess_data(args.results_path)
    domains = ['content', 'design', 'coping', 'quitting']
    
    calculate_global_metrics(df, domains)
    calculate_per_participant_ranking(df, domains)

if __name__ == '__main__':
    main()


