# Comprehensive Analysis Guide

This guide explains the publication-ready analysis tools for comparing LLM models in the smoking cessation message evaluation task.

## Quick Start

```bash
# Run comprehensive analysis (generates all figures and tables)
conda activate research
python analysis-script/comprehensive_model_comparison.py
```

## Output Files

All outputs are saved to `results_manuscript/figures/`:

### Main Results (for manuscript body)

1. **Table 1**: `table1_main_comparison.csv/tex/md`
   - Primary comparison: GPT-5 vs DeepSeek-R1
   - Metrics with 95% confidence intervals
   - Includes: Accuracy, Cohen's κ, F1 Macro, Acc±1, Kendall's τ, Spearman's ρ

2. **Figure 2**: `fig2_kappa_vs_accuracy_scatter.png`
   - Scatter plot: Cohen's Kappa vs Accuracy
   - Left panel: Primary models (GPT-5, DeepSeek-R1)
   - Right panel: All models
   - Color-coded by domain (Content, Coping, Quitting)

3. **Figure 3**: `fig3_radar_charts_by_domain.png`
   - Radar charts comparing models across multiple metrics
   - One chart per domain
   - Metrics: Accuracy, κ, F1, Acc±1, Spearman's ρ

4. **Figure 4**: `fig4_digital_twin_learning_curves.png`
   - Learning curves for digital twin models
   - Shows performance vs training set size (10%, 30%, 70%, 90%)
   - Separate plots for Accuracy and Cohen's Kappa
   - All three domains shown

5. **Figure 5**: `fig5_heatmap_*.png`
   - Heatmaps for accuracy, kappa, and F1 macro
   - All models × all domains
   - Color gradient for easy comparison

### Statistical Tests

**Table 2**: `table2_statistical_tests.csv/md`
- Paired t-tests between GPT-5 and DeepSeek-R1
- Wilcoxon signed-rank tests (non-parametric)
- Mean differences and significance indicators
- Per-domain comparisons

### Supplementary Materials

**Supplementary Table**: `supplementary_table_all_models.csv/tex`
- Complete results for all 5 models
- All metrics for all methods
- Includes: GPT-4o-mini, Grok-4-Fast, Gemini-2.5-Pro

## Key Features

### 1. Bootstrap Confidence Intervals
- 1000 bootstrap iterations
- 95% confidence intervals for Accuracy, Kappa, and F1
- Enables robust uncertainty quantification

### 2. Multiple Metrics
- **Exact Accuracy**: Strict classification accuracy
- **Accuracy±1**: Allows ±1 error (ordinal classification)
- **Cohen's Kappa**: Agreement beyond chance
- **Kendall's Tau**: Rank correlation (ordinal)
- **Spearman's ρ**: Per-participant ranking correlation
- **F1/Precision/Recall**: Standard classification metrics (macro-averaged)

### 3. Statistical Significance
- Paired t-tests for comparing models on same samples
- Wilcoxon signed-rank tests (non-parametric alternative)
- Reports mean differences and p-values

### 4. Multi-level Comparisons
- **Primary models**: GPT-5 and DeepSeek-R1 (main text)
- **Supplementary models**: GPT-4o-mini, Grok-4-Fast, Gemini-2.5-Pro
- **Per-domain analysis**: Content, Coping, Quitting
- **Learning curves**: Digital twin models with varying training sizes

## Interpretation Guide

### Cohen's Kappa (κ)
- **< 0.00**: Poor agreement (worse than chance)
- **0.00-0.20**: Slight agreement
- **0.21-0.40**: Fair agreement
- **0.41-0.60**: Moderate agreement
- **0.61-0.80**: Substantial agreement
- **0.81-1.00**: Almost perfect agreement

### Spearman's ρ (per-participant)
- Measures whether models correctly rank messages *within* each participant
- More relevant for personalized recommendations
- Averaged across all participants

### Accuracy±1
- Important for ordinal scales (1-5 ratings)
- Being "close" may be acceptable for message recommendation
- Higher than exact accuracy, shows directional correctness

## Analysis Philosophy

Following best practices from ML classification papers:

1. **Primary vs Supplementary**: Focus main text on GPT-5 and DeepSeek-R1 as they represent state-of-the-art reasoning models. Other models in supplementary materials.

2. **Multiple Metrics**: Classification tasks require diverse metrics beyond accuracy:
   - Kappa: Corrects for class imbalance and chance agreement
   - Ranking metrics: Important for recommendation systems
   - Ordinal metrics: Respect the 1-5 rating scale structure

3. **Uncertainty Quantification**: Bootstrap CIs show robustness of findings

4. **Statistical Testing**: Formal tests establish whether differences are significant

5. **Visual Clarity**: Multiple visualization types for different insights:
   - Scatter plots: Relationship between metrics
   - Radar charts: Multi-dimensional comparison
   - Heatmaps: Dense matrix visualization
   - Learning curves: Sample efficiency

## Customization

### Changing Primary Models
Edit `PRIMARY_MODELS` in `comprehensive_model_comparison.py`:
```python
PRIMARY_MODELS = ['your-model-1', 'your-model-2']
```

### Adding New Metrics
Add to `calculate_comprehensive_metrics()` function:
```python
new_metric = your_metric_function(gt, pred)
metrics[domain]['new_metric'] = new_metric
```

### Adjusting Confidence Level
Change `np.percentile([2.5, 97.5])` for different CI levels:
- 90% CI: `[5, 95]`
- 99% CI: `[0.5, 99.5]`

## Citation

If using these analysis tools, cite:
- **Cohen's Kappa**: Cohen, J. (1960). A coefficient of agreement for nominal scales.
- **Bootstrap CI**: Efron, B., & Tibshirani, R. J. (1994). An Introduction to the Bootstrap.
- **Kendall's Tau**: Kendall, M. G. (1938). A new measure of rank correlation.

## Troubleshooting

### "Model not found in results"
- Ensure you've run `e2e_pipeline.sh` or `run_full_manuscript_pipeline.sh` for that model
- Check that result files exist in `results_manuscript_<model>/`

### Empty plots
- Verify that at least 2 models have valid results
- Check console output for loading errors

### Statistical tests fail
- Requires exactly 2 primary models with matching samples
- Ensure both models have completed the same method (e.g., `5_continuous`)

## Additional Resources

- **Full pipeline**: See `e2e_pipeline.sh` for running all experiments
- **Individual analysis**: See `analyze_manuscript_results.py` for method-level analysis
- **Calibration analysis**: See `analyze_llm_calibration.py` for probability calibration
- **Traditional ML baseline**: See `compare_llm_vs_individual.py` for ML benchmarks

