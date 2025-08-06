import json
import os
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import re

def extract_model_mode_from_filename(filename):
    """Extract model and mode from calibrated results filename"""
    basename = os.path.basename(filename)
    # Pattern to match: calibrated_results_{model}_{mode}.json
    pattern = r'calibrated_results_(.+)_([^_]+)\.json'
    match = re.search(pattern, basename)
    if match:
        return match.group(1), match.group(2)
    else:
        return "unknown_model", "unknown_mode"

def plot_calibration_results(calibrated_file, output_dir=None):
    """
    Analyze and plot calibration results.
    """
    model, mode = extract_model_mode_from_filename(calibrated_file)
    
    if output_dir is None:
        output_dir = f'calibration_plots_{model}_{mode}'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Processing calibration results for model: {model}, mode: {mode}")
    print(f"Output directory: {output_dir}")

    # Load calibrated results
    with open(calibrated_file, 'r') as f:
        results_data = json.load(f)

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(results_data, orient='index')
    
    dimensions = ['content', 'design', 'coping', 'quitting']
    metrics_summary = {}

    for dim in dimensions:
        print(f"\nAnalyzing {dim} dimension...")
        
        # Filter valid data
        valid_mask = (
            df[f'ground_truth_{dim}'].notna() & 
            df[f'predicted_{dim}'].notna() & 
            df[f'calibrated_{dim}'].notna()
        )
        valid_df = df[valid_mask].copy()
        
        if len(valid_df) == 0:
            print(f"No valid data for {dim}")
            continue
            
        ground_truth = valid_df[f'ground_truth_{dim}'].values
        predictions = valid_df[f'predicted_{dim}'].values
        calibrated = valid_df[f'calibrated_{dim}'].values
        confidence = valid_df[f'confidence_{dim}'].fillna(0).values
        
        # Calculate metrics
        mse_before = mean_squared_error(ground_truth, predictions)
        mse_after = mean_squared_error(ground_truth, calibrated)
        mae_before = mean_absolute_error(ground_truth, predictions)
        mae_after = mean_absolute_error(ground_truth, calibrated)
        
        corr_before, _ = pearsonr(ground_truth, predictions)
        corr_after, _ = pearsonr(ground_truth, calibrated)
        
        metrics_summary[dim] = {
            'mse_before': mse_before,
            'mse_after': mse_after,
            'mae_before': mae_before, 
            'mae_after': mae_after,
            'correlation_before': corr_before,
            'correlation_after': corr_after,
            'n_samples': len(valid_df)
        }
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{dim.capitalize()} Ratings - Before vs After Calibration ({model}, {mode})', fontsize=16)
        
        # Scatter plot: Before calibration
        axes[0, 0].scatter(ground_truth, predictions, alpha=0.6, c=confidence, cmap='viridis')
        axes[0, 0].plot([1, 5], [1, 5], 'r--', alpha=0.8)
        axes[0, 0].set_xlabel('Ground Truth')
        axes[0, 0].set_ylabel('Predicted (Before)')
        axes[0, 0].set_title(f'Before Calibration\nMSE: {mse_before:.3f}, r: {corr_before:.3f}')
        axes[0, 0].set_xlim(0.5, 5.5)
        axes[0, 0].set_ylim(0.5, 5.5)
        
        # Scatter plot: After calibration
        axes[0, 1].scatter(ground_truth, calibrated, alpha=0.6, c=confidence, cmap='viridis')
        axes[0, 1].plot([1, 5], [1, 5], 'r--', alpha=0.8)
        axes[0, 1].set_xlabel('Ground Truth')
        axes[0, 1].set_ylabel('Predicted (After)')
        axes[0, 1].set_title(f'After Calibration\nMSE: {mse_after:.3f}, r: {corr_after:.3f}')
        axes[0, 1].set_xlim(0.5, 5.5)
        axes[0, 1].set_ylim(0.5, 5.5)
        
        # Residual comparison
        residuals_before = predictions - ground_truth
        residuals_after = calibrated - ground_truth
        
        axes[0, 2].scatter(ground_truth, residuals_before, alpha=0.6, label='Before', color='red')
        axes[0, 2].scatter(ground_truth, residuals_after, alpha=0.6, label='After', color='blue')
        axes[0, 2].axhline(y=0, color='black', linestyle='--', alpha=0.8)
        axes[0, 2].set_xlabel('Ground Truth')
        axes[0, 2].set_ylabel('Residuals')
        axes[0, 2].set_title('Residuals Comparison')
        axes[0, 2].legend()
        
        # Distribution of predictions
        axes[1, 0].hist(predictions, bins=20, alpha=0.7, label='Before', color='red', density=True)
        axes[1, 0].hist(ground_truth, bins=20, alpha=0.7, label='Ground Truth', color='green', density=True)
        axes[1, 0].set_xlabel('Rating')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Distribution: Before Calibration')
        axes[1, 0].legend()
        
        axes[1, 1].hist(calibrated, bins=20, alpha=0.7, label='After', color='blue', density=True)
        axes[1, 1].hist(ground_truth, bins=20, alpha=0.7, label='Ground Truth', color='green', density=True)
        axes[1, 1].set_xlabel('Rating')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Distribution: After Calibration')
        axes[1, 1].legend()
        
        # Confidence vs accuracy
        abs_errors_before = np.abs(residuals_before)
        abs_errors_after = np.abs(residuals_after)
        
        # Bin by confidence and calculate mean absolute error
        conf_bins = np.linspace(1, 5, 11)
        bin_centers = (conf_bins[:-1] + conf_bins[1:]) / 2
        mae_by_conf_before = []
        mae_by_conf_after = []
        
        for i in range(len(conf_bins)-1):
            mask = (confidence >= conf_bins[i]) & (confidence < conf_bins[i+1])
            if np.sum(mask) > 0:
                mae_by_conf_before.append(np.mean(abs_errors_before[mask]))
                mae_by_conf_after.append(np.mean(abs_errors_after[mask]))
            else:
                mae_by_conf_before.append(np.nan)
                mae_by_conf_after.append(np.nan)
        
        axes[1, 2].plot(bin_centers, mae_by_conf_before, 'ro-', label='Before', alpha=0.7)
        axes[1, 2].plot(bin_centers, mae_by_conf_after, 'bo-', label='After', alpha=0.7)
        axes[1, 2].set_xlabel('Confidence')
        axes[1, 2].set_ylabel('Mean Absolute Error')
        axes[1, 2].set_title('Confidence vs Accuracy')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'calibration_analysis_{dim}_{model}_{mode}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved analysis plot for {dim} to {plot_path}")

    # Create summary metrics plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Calibration Summary - {model} ({mode})', fontsize=16)
    
    dims = list(metrics_summary.keys())
    if len(dims) > 0:
        mse_before = [metrics_summary[d]['mse_before'] for d in dims]
        mse_after = [metrics_summary[d]['mse_after'] for d in dims]
        mae_before = [metrics_summary[d]['mae_before'] for d in dims]
        mae_after = [metrics_summary[d]['mae_after'] for d in dims]
        
        x = np.arange(len(dims))
        width = 0.35
        
        # MSE comparison
        axes[0, 0].bar(x - width/2, mse_before, width, label='Before', alpha=0.8, color='red')
        axes[0, 0].bar(x + width/2, mse_after, width, label='After', alpha=0.8, color='blue')
        axes[0, 0].set_xlabel('Dimension')
        axes[0, 0].set_ylabel('Mean Squared Error')
        axes[0, 0].set_title('MSE: Before vs After Calibration')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(dims)
        axes[0, 0].legend()
        
        # MAE comparison
        axes[0, 1].bar(x - width/2, mae_before, width, label='Before', alpha=0.8, color='red')
        axes[0, 1].bar(x + width/2, mae_after, width, label='After', alpha=0.8, color='blue')
        axes[0, 1].set_xlabel('Dimension')
        axes[0, 1].set_ylabel('Mean Absolute Error')
        axes[0, 1].set_title('MAE: Before vs After Calibration')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(dims)
        axes[0, 1].legend()
        
        # Correlation comparison
        corr_before = [metrics_summary[d]['correlation_before'] for d in dims]
        corr_after = [metrics_summary[d]['correlation_after'] for d in dims]
        
        axes[1, 0].bar(x - width/2, corr_before, width, label='Before', alpha=0.8, color='red')
        axes[1, 0].bar(x + width/2, corr_after, width, label='After', alpha=0.8, color='blue')
        axes[1, 0].set_xlabel('Dimension')
        axes[1, 0].set_ylabel('Pearson Correlation')
        axes[1, 0].set_title('Correlation: Before vs After Calibration')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(dims)
        axes[1, 0].legend()
        
        # Sample sizes
        sample_sizes = [metrics_summary[d]['n_samples'] for d in dims]
        axes[1, 1].bar(dims, sample_sizes, alpha=0.8, color='green')
        axes[1, 1].set_xlabel('Dimension')
        axes[1, 1].set_ylabel('Number of Samples')
        axes[1, 1].set_title('Sample Sizes')
        
        plt.tight_layout()
        summary_path = os.path.join(output_dir, f'calibration_summary_{model}_{mode}.png')
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved summary plot to {summary_path}")

    # Save metrics to JSON
    metrics_path = os.path.join(output_dir, f'calibration_metrics_{model}_{mode}.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"Saved metrics to {metrics_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot calibration results.")
    parser.add_argument('calibrated_file', type=str, help="Path to the calibrated results JSON file.")
    parser.add_argument('--output_dir', type=str, default=None, help="Directory to save plots.")
    args = parser.parse_args()
    
    plot_calibration_results(args.calibrated_file, args.output_dir) 