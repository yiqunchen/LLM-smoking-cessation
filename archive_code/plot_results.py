import json
import os
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import re

def extract_model_mode_from_filename(filename):
    """
    Extract model, mode, and prompt_config from filename like 'evaluation_results_gpt-4o-mini_vision_zero-shot.json'
    Returns (model, mode, prompt_config) tuple
    """
    basename = os.path.basename(filename)
    # Pattern to match: evaluation_results_{model}_{mode}_{prompt_config}.json or checkpoint_results_{model}_{mode}_{prompt_config}.json
    pattern = r'(?:evaluation_results|checkpoint_results)_(.+)_([^_]+)_([^_]+)\.json'
    match = re.search(pattern, basename)
    if match:
        return match.group(1), match.group(2), match.group(3)
    else:
        # Fallback pattern for old format without prompt_config: evaluation_results_{model}_{mode}.json
        pattern_old = r'(?:evaluation_results|checkpoint_results)_(.+)_([^_]+)\.json'
        match_old = re.search(pattern_old, basename)
        if match_old:
            return match_old.group(1), match_old.group(2), "unknown_prompt"
        else:
            # Ultimate fallback if pattern doesn't match
            return "unknown_model", "unknown_mode", "unknown_prompt"

def plot_results(input_file, output_dir=None):
    """
    Loads model evaluation results, calculates metrics, and generates plots.

    Args:
        input_file (str): Path to the JSON file containing evaluation results.
        output_dir (str): Directory to save plots and metrics. If None, auto-generated from model/mode/prompt_config.
    """
    # Extract model, mode, and prompt_config from filename
    model, mode, prompt_config = extract_model_mode_from_filename(input_file)
    
    # Create model/mode/prompt_config-specific output directory if not provided
    if output_dir is None:
        output_dir = f'plots_{model}_{mode}_{prompt_config}'
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Processing results for model: {model}, mode: {mode}, prompt_config: {prompt_config}")
    print(f"Output directory: {output_dir}")

    # Load the results data
    with open(input_file, 'r') as f:
        results_data = json.load(f)

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(results_data, orient='index')
    
    # Define the rating dimensions and their possible labels
    dimensions = {
        'content': ["Very poor", "Poor", "Acceptable", "Good", "Very good"],
        'design': ["Very poor", "Poor", "Acceptable", "Good", "Very good"],
        'coping': ["Not at all helpful", "Somewhat helpful", "Moderately helpful", "Very helpful", "Extremely helpful"],
        'quitting': ["Not at all helpful", "Somewhat helpful", "Moderately helpful", "Very helpful", "Extremely helpful"]
    }
    
    summary_metrics = {}

    for dim, labels in dimensions.items():
        true_col = f'ground_truth_{dim}'
        pred_col = f'predicted_{dim}'

        # Filter out rows with errors, missing values, or invalid data
        filtered_df = df[
            (df[pred_col] != "ERROR") & 
            (df[true_col].notna()) & 
            (df[pred_col].notna()) &
            (df[true_col] != "") &
            (df[pred_col] != "")
        ]

        # Calculate accuracy
        accuracy = accuracy_score(filtered_df[true_col], filtered_df[pred_col])
        
        # Generate classification report
        report = classification_report(filtered_df[true_col], filtered_df[pred_col], labels=labels, output_dict=True, zero_division=0)

        summary_metrics[dim] = {
            'accuracy': accuracy,
            'classification_report': report
        }

        # Generate and save confusion matrix plot
        cm = confusion_matrix(filtered_df[true_col], filtered_df[pred_col], labels=labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix for {dim.capitalize()} Rating - {model} ({mode}, {prompt_config})')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plot_path = os.path.join(output_dir, f'confusion_matrix_{dim}_{model}_{mode}_{prompt_config}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved confusion matrix for {dim} to {plot_path}")

    # Save summary metrics to a JSON file
    metrics_path = os.path.join(output_dir, f'summary_metrics_{model}_{mode}_{prompt_config}.json')
    with open(metrics_path, 'w') as f:
        json.dump(summary_metrics, f, indent=4)
    print(f"Saved summary metrics to {metrics_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot evaluation results.")
    parser.add_argument('input_file', type=str, help="Path to the evaluation results JSON file.")
    parser.add_argument('--output_dir', type=str, default=None, help="Directory to save plots and metrics. If not provided, auto-generated from model/mode.")
    args = parser.parse_args()
    
    plot_results(args.input_file, args.output_dir) 