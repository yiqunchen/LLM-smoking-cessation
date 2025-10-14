import json
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve
from scipy.optimize import minimize 
from scipy.special import softmax

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
        # Skip entries with errors
        if item.get("predicted_content") == "ERROR":
            continue
        rows.append(item)
    
    df = pd.DataFrame(rows)
    
    # Convert predictions and ground truth to numeric
    for domain, rating_map in RATING_MAPS.items():
        df[f'gt_{domain}_num'] = df[f'ground_truth_{domain}'].map(rating_map)
        df[f'pred_{domain}_num'] = df[f'predicted_{domain}'].map(rating_map)

    return df

def get_probabilities_from_dict(row, domain):
    """Extracts and orders probabilities from the nested dictionary."""
    prob_dict = row[f'predicted_{domain}_probabilities']
    rating_map = RATING_MAPS[domain]
    
    # Initialize probabilities for all 5 classes
    probas = np.zeros(5)
    if not isinstance(prob_dict, dict):
        return probas # Return zeros if data is malformed
        
    for rating_text, rating_num in rating_map.items():
        # LLM might not be consistent with keys, so check carefully
        prob = prob_dict.get(rating_text, 0.0)
        # The rating_num is 1-5, so it corresponds to index 0-4
        probas[rating_num - 1] += float(prob if prob is not None else 0.0)

    # Normalize to ensure they sum to 1, handling potential LLM errors
    if probas.sum() > 0:
        return probas / probas.sum()
    return probas


def calculate_metrics(df: pd.DataFrame, domains: list) -> dict:
    """Calculates accuracy and AUC for each domain."""
    metrics = {}
    for domain in domains:
        gt = df[f'gt_{domain}_num'].dropna()
        pred = df.loc[gt.index, f'pred_{domain}_num']
        
        # Extract probabilities for AUC calculation
        probas = np.vstack(df.loc[gt.index].apply(lambda row: get_probabilities_from_dict(row, domain), axis=1))
        
        # Binarize labels for AUC
        classes = np.arange(1, 6)
        gt_binarized = label_binarize(gt, classes=classes)

        metrics[domain] = {
            'accuracy': accuracy_score(gt, pred),
            'auc_macro_ovr': roc_auc_score(gt_binarized, probas, multi_class='ovr', average='macro')
        }
    return metrics

def plot_reliability_diagram(y_true, y_prob, n_bins, title, ax):
    """Plots a reliability diagram."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
    ax.plot(prob_pred, prob_true, marker='.', label='Model')
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

def calculate_ece(y_true, y_prob, n_bins=10):
    """Calculates the Expected Calibration Error."""
    bin_limits = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_mask = (y_prob > bin_limits[i]) & (y_prob <= bin_limits[i+1])
        if np.sum(bin_mask) > 0:
            bin_accuracy = np.mean(y_true[bin_mask])
            bin_confidence = np.mean(y_prob[bin_mask])
            ece += np.abs(bin_accuracy - bin_confidence) * (np.sum(bin_mask) / len(y_true))
    return ece

class TemperatureScaler:
    """
    Implements Temperature Scaling for calibrating model probabilities.
    The temperature 'T' is found by minimizing the Negative Log-Likelihood (NLL)
    on a calibration set, which is a convex optimization problem.
    """
    def __init__(self):
        self.temperature = 1.0

    def _nll(self, temp, logits, labels):
        """Negative Log-Likelihood loss."""
        scaled_logits = logits / temp
        log_probs = -np.log(softmax(scaled_logits, axis=1))
        return np.mean(np.diag(log_probs[:, labels.astype(int) - 1]))

    def fit(self, logits, labels):
        """Finds the optimal temperature."""
        result = minimize(self._nll, x0=1.0, args=(logits, labels), method='L-BFGS-B', bounds=[(0.1, 10.0)])
        self.temperature = result.x[0]
        return self

    def transform(self, logits):
        """Applies the temperature to logits to get calibrated probabilities."""
        return softmax(logits / self.temperature, axis=1)

def main():
    parser = argparse.ArgumentParser(description="Analyze LLM evaluation results with a focus on calibration.")
    parser.add_argument('results_path', help="Path to the JSON file with evaluation results.")
    parser.add_argument('--output-dir', default='llm_calibration_analysis', help="Directory to save plots.")
    parser.add_argument('--seed', type=int, default=202509, help="Random seed for reproducibility.")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load and process data
    df = load_and_preprocess_data(args.results_path)
    domains = ['content', 'design', 'coping', 'quitting']
    
    # 2. Calculate base performance metrics
    print("--- Base Performance Metrics ---")
    base_metrics = calculate_metrics(df, domains)
    for domain, metrics in base_metrics.items():
        print(f"  {domain.title()}: Accuracy={metrics['accuracy']:.3f}, AUC (Macro OVR)={metrics['auc_macro_ovr']:.3f}")

    # 3. Analyze confidence scores
    plt.figure(figsize=(12, 5))
    for i, domain in enumerate(domains):
        ax = plt.subplot(1, len(domains), i + 1)
        confidence_col = f'predicted_{domain}_confidence'
        if confidence_col in df.columns:
            df['correct'] = df[f'gt_{domain}_num'] == df[f'pred_{domain}_num']
            confidence_analysis = df.groupby(pd.cut(df[confidence_col], np.arange(0, 1.1, 0.1)))['correct'].mean()
            confidence_analysis.plot(kind='bar', ax=ax)
            ax.set_title(f'{domain.title()} Confidence vs. Accuracy')
            ax.set_xlabel("Self-Reported Confidence Bins")
            ax.set_ylabel("Mean Accuracy")
            ax.set_ylim(0, 1)
    plt.suptitle("LLM Self-Reported Confidence vs. Actual Accuracy")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(args.output_dir, 'confidence_vs_accuracy.png'))
    plt.close()

    # 4. Calibration Analysis (Before and After)
    print("\n--- Calibration Analysis (ECE) ---")
    fig, axes = plt.subplots(len(domains), 2, figsize=(10, 4 * len(domains)))
    
    for i, domain in enumerate(domains):
        # Extract probabilities and convert to logits
        probas = np.vstack(df.apply(lambda row: get_probabilities_from_dict(row, domain), axis=1))
        # Add a small epsilon to avoid log(0)
        logits = np.log(probas + 1e-9)
        labels = df[f'gt_{domain}_num']

        # --- FIX: Filter out rows with NaN labels before splitting ---
        valid_indices = labels.dropna().index
        
        # Create calibration split from only the valid indices
        np.random.seed(args.seed)
        calib_indices = np.random.choice(valid_indices, size=len(valid_indices) // 4, replace=False)
        eval_indices = valid_indices.difference(calib_indices)
        
        # Before calibration
        y_true_before = labels[eval_indices].values
        # Get the probability of the predicted class
        y_prob_before = np.max(probas[eval_indices], axis=1)
        ece_before = calculate_ece(y_true_before == df.loc[eval_indices, f'pred_{domain}_num'], y_prob_before)
        plot_reliability_diagram(y_true_before == df.loc[eval_indices, f'pred_{domain}_num'], y_prob_before, n_bins=10, title=f'{domain.title()} (Before Calibration)', ax=axes[i, 0])

        # Calibrate
        scaler = TemperatureScaler().fit(logits[calib_indices], labels[calib_indices])
        calibrated_probas = scaler.transform(logits[eval_indices])
        
        # After calibration
        calibrated_preds = np.argmax(calibrated_probas, axis=1) + 1
        y_prob_after = np.max(calibrated_probas, axis=1)
        ece_after = calculate_ece(y_true_before == calibrated_preds, y_prob_after)
        plot_reliability_diagram(y_true_before == calibrated_preds, y_prob_after, n_bins=10, title=f'{domain.title()} (After Calibration, T={scaler.temperature:.2f})', ax=axes[i, 1])
        
        print(f"  {domain.title()}: ECE Before={ece_before:.4f} -> ECE After={ece_after:.4f}")

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'calibration_plots.png'))
    plt.close()

if __name__ == '__main__':
    main()
