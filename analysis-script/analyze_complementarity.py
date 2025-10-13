import pandas as pd
import numpy as np
import json
import os

def analyze_complementarity(
    aligned_preds_path: str = 'llm_vs_individual_comparison/aligned_predictions.csv',
    full_data_path: str = 'data/processed_llm_data.json',
    output_dir: str = 'llm_vs_individual_comparison'
):
    """Analyzes cases where one model is correct and the other is not."""
    
    preds_df = pd.read_csv(aligned_preds_path)
    
    # Load full data to get all metadata
    with open(full_data_path, 'r') as f:
        full_data = json.load(f)
    
    full_df = pd.DataFrame(full_data)
    meta_df = pd.json_normalize(full_df['metadata'])
    full_df = pd.concat([full_df[['response_id', 'input_message']], meta_df], axis=1)
    
    # Merge predictions with full metadata
    merged = pd.merge(preds_df, full_df, on=['response_id', 'input_message'], how='left')

    domains = ['content', 'design', 'coping', 'quitting']
    
    analysis_results = {}

    for domain in domains:
        gt_col = f'{domain}_num'
        ind_col = f'ind_{domain}_oof'
        llm_col = f'llm_{domain}_num'

        if gt_col not in merged.columns or ind_col not in merged.columns or llm_col not in merged.columns:
            continue

        # Create correctness flags
        merged['ind_correct'] = (merged[ind_col] == merged[gt_col])
        merged['llm_correct'] = (merged[llm_col] == merged[gt_col])
        
        # Define complementary subsets
        ind_only_correct_mask = (merged['ind_correct'] == True) & (merged['llm_correct'] == False)
        llm_only_correct_mask = (merged['llm_correct'] == True) & (merged['ind_correct'] == False)
        
        ind_only_df = merged[ind_only_correct_mask]
        llm_only_df = merged[llm_only_correct_mask]
        
        analysis_results[domain] = {}

        # Analyze some key features
        features_to_compare = [
            'age_years', 'days_smoked_past_30d', 'quit_attempts_count',
            'quit_motivation_level', 'social_support_to_quit', 'gender_identity'
        ]
        
        for feature in features_to_compare:
            if feature not in merged.columns:
                continue
            
            # For numeric features, compare means
            if pd.api.types.is_numeric_dtype(merged[feature]):
                analysis_results[domain][feature] = {
                    'ind_only_correct_mean': ind_only_df[feature].mean(),
                    'llm_only_correct_mean': llm_only_df[feature].mean(),
                    'overall_mean': merged[feature].mean()
                }
            # For categorical features, compare value counts
            else:
                 analysis_results[domain][feature] = {
                    'ind_only_correct_dist': (ind_only_df[feature].value_counts(normalize=True) * 100).to_dict(),
                    'llm_only_correct_dist': (llm_only_df[feature].value_counts(normalize=True) * 100).to_dict(),
                    'overall_dist': (merged[feature].value_counts(normalize=True) * 100).to_dict()
                }

    # Save detailed analysis to a file
    output_path = os.path.join(output_dir, 'complementarity_analysis.json')
    with open(output_path, 'w') as f:
        json.dump(analysis_results, f, indent=2)
        
    print(f"Saved complementarity analysis to {output_path}")
    
    # Print a summary of the most interesting findings
    print("\n--- Complementarity Analysis Summary ---")
    for domain, features in analysis_results.items():
        print(f"\n## Domain: {domain.title()}")
        if 'quit_motivation_level' in features:
            print("\n* Quit Motivation Level (% distribution):")
            print(f"  - LLM Correct Only: {features['quit_motivation_level']['llm_only_correct_dist']}")
            print(f"  - Ind Correct Only: {features['quit_motivation_level']['ind_only_correct_dist']}")
        if 'age_years' in features:
             print("\n* Age (mean):")
             print(f"  - LLM Correct Only: {features['age_years']['llm_only_correct_mean']:.1f}")
             print(f"  - Ind Correct Only: {features['age_years']['ind_only_correct_mean']:.1f}")


if __name__ == '__main__':
    analyze_complementarity()


