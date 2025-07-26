import json
import os
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def plot_results(input_file, output_dir='plots'):
    """
    Loads model evaluation results, calculates metrics, and generates plots.

    Args:
        input_file (str): Path to the JSON file containing evaluation results.
        output_dir (str): Directory to save plots and metrics.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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

        # Filter out rows with errors
        filtered_df = df[df[pred_col] != "ERROR"]

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
        plt.title(f'Confusion Matrix for {dim.capitalize()} Rating')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plot_path = os.path.join(output_dir, f'confusion_matrix_{dim}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved confusion matrix for {dim} to {plot_path}")

    # Save summary metrics to a JSON file
    metrics_path = os.path.join(output_dir, 'summary_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(summary_metrics, f, indent=4)
    print(f"Saved summary metrics to {metrics_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot evaluation results.")
    parser.add_argument('input_file', type=str, help="Path to the evaluation results JSON file.")
    parser.add_argument('--output_dir', type=str, default='plots', help="Directory to save plots and metrics.")
    args = parser.parse_args()
    
    plot_results(args.input_file, args.output_dir) 