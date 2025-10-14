# Comprehensive Cross-Model Analysis Figures

This directory contains publication-ready figures comparing **ALL 5 models** across **ALL 6 methods** for the smoking cessation message evaluation task.

## Quick Start

```bash
# Generate all figures
conda activate research
python analysis-script/create_comprehensive_figures.py
```

## Models Analyzed

1. **GPT-4o-mini** (OpenAI)
2. **GPT-5** (OpenAI)
3. **DeepSeek-R1** (DeepSeek)
4. **Grok-4-Fast** (xAI via OpenRouter)
5. **Gemini-2.5-Pro** (Google)

## Methods Analyzed

### Generic LLM Methods (70/30 participant split)
1. **Zero-shot (all)** - Zero-shot with all participant features
2. **Zero-shot (select)** - Zero-shot with selected features
3. **Few-shot (all)** - Few-shot with all participant features
4. **Few-shot (select)** - Few-shot with selected features
5. **Continuous (NL)** - Continuous rating + natural language profile

### Digital Twin Methods (70/30 message split)
6. **Digital Twin (70/30)** - CBT/ACT-informed digital twin with 70% training

## Domains Evaluated

- **Content** - Message content quality (grammar, clarity, relevance)
- **Coping** - Helpfulness for coping with cravings
- **Quitting** - Helpfulness for quitting smoking

## Generated Figures

### 📊 Radar Charts (3 files)
**Files:** `radar_all_methods_content.png`, `radar_all_methods_coping.png`, `radar_all_methods_quitting.png`

- **Purpose:** Multi-metric comparison across all models and methods
- **Layout:** 2×3 grid (one subplot per model)
- **Metrics shown:** Accuracy, Cohen's κ, Acc±1, Spearman's ρ
- **Lines:** Each line represents one method for that model
- **Use case:** See which methods work best for each model in each domain

**Key Insights:**
- Digital Twin methods generally show higher performance (larger radar area)
- Few-shot methods show variable performance across models
- Spearman's ρ (per-participant ranking) is generally low, indicating personalization challenges

### 🔥 Heatmaps (3 files)
**Files:** `heatmap_all_methods_accuracy.png`, `heatmap_all_methods_kappa.png`, `heatmap_all_methods_acc_within_1.png`

- **Purpose:** Dense matrix visualization of all models × all methods
- **Layout:** 3 panels (one per domain)
- **Rows:** Models
- **Columns:** Methods
- **Color:** Green = high, Red = low
- **Use case:** Quickly identify best model-method combinations

**Key Insights:**
- Digital Twin consistently shows highest values (greenest)
- Content domain generally easier than Coping/Quitting
- Acc±1 (directional accuracy) much higher than exact accuracy

### 📊 Grouped Bar Charts (3 files)
**Files:** `bars_all_methods_accuracy.png`, `bars_all_methods_kappa.png`, `bars_all_methods_acc_within_1.png`

- **Purpose:** Direct comparison of models within each method
- **Layout:** 3 panels (one per domain)
- **X-axis:** Methods
- **Bars:** One bar per model (colored)
- **Value labels:** Shown on top of each bar
- **Use case:** Compare models head-to-head for each method

**Key Insights:**
- Models cluster closely for most methods (no clear winner)
- Digital Twin shows largest spread between models
- Gemini-2.5-Pro and Grok-4-Fast competitive with GPT-5

### 🎯 Scatter Plot (1 file)
**File:** `scatter_kappa_vs_accuracy_all_methods.png`

- **Purpose:** Relationship between agreement (κ) and performance (accuracy)
- **Layout:** 2×3 grid (one subplot per method)
- **Points:** Each point = one model in one domain
- **Colors:** Blue = Content, Green = Coping, Red = Quitting
- **Diagonal line:** Perfect agreement (κ = accuracy)
- **Use case:** Understand if high accuracy implies high agreement

**Key Insights:**
- Most points below diagonal (accuracy > κ), indicating models perform better than chance but don't achieve strong agreement
- Digital Twin shows highest κ values
- Quitting domain (red) consistently lowest on both axes

## Data Tables

### 📄 comprehensive_results_all_methods.csv
**Columns:**
- Model, Model_ID, Method, Category, Domain
- Accuracy, Acc±1, Kappa, Kendall_Tau, Spearman_Rho, N

**Total rows:** 81 (5 models × 6 methods × 3 domains, minus missing DeepSeek few-shot/digital twin)

**Use case:** 
- Import into statistical software for analysis
- Create custom visualizations
- Report exact numbers in manuscript

### 📄 summary_statistics.csv
**Content:** Mean and std for each model-method-domain combination

**Use case:**
- Quick lookup of performance ranges
- Identify high-variance methods

## Interpretation Guide

### Metrics Explained

**Accuracy** (Exact Match)
- Range: 0-1 (higher is better)
- Interpretation: Percentage of exact rating matches
- Typical values: 0.20-0.40 (20-40%)
- **Finding:** Models achieve ~30-40% exact accuracy

**Acc±1** (Directional Accuracy)
- Range: 0-1 (higher is better)
- Interpretation: Percentage within ±1 rating point
- Typical values: 0.60-0.90 (60-90%)
- **Finding:** Models capture directional trends well (70-90%)

**Cohen's Kappa (κ)**
- Range: -1 to 1 (higher is better)
- Interpretation: Agreement beyond chance
- < 0.00: Poor (worse than chance)
- 0.00-0.20: Slight
- 0.21-0.40: Fair
- 0.41-0.60: Moderate
- 0.61-0.80: Substantial
- 0.81-1.00: Almost perfect
- **Finding:** Most methods show slight agreement (κ = 0.00-0.10)

**Kendall's Tau (τ)**
- Range: -1 to 1 (higher is better)
- Interpretation: Rank correlation (ordinal)
- **Finding:** Positive but low (τ = 0.00-0.15)

**Spearman's Rho (ρ)**
- Range: -1 to 1 (higher is better)
- Interpretation: Per-participant ranking correlation
- **Finding:** Highly variable, often negative (ρ = -0.20 to +0.20)
- **Implication:** Zero/few-shot models struggle with personalization

### Domain Difficulty Ranking

1. **Content** (easiest): 30-42% accuracy
   - Most objective criteria
   - Grammar, clarity, relevance
   
2. **Coping** (medium): 24-33% accuracy
   - Requires understanding behavioral strategies
   - Moderate subjectivity

3. **Quitting** (hardest): 16-33% accuracy
   - Highly subjective and personalized
   - Depends on individual quit stage

### Method Performance Ranking

**By Exact Accuracy:**
1. Digital Twin (70/30): 32-42%
2. Few-shot (select): 24-33%
3. Zero-shot (select): 20-39%
4. Continuous (NL): 25-34%
5. Few-shot (all): 20-35%
6. Zero-shot (all): 20-38%

**By Directional Accuracy (Acc±1):**
1. Digital Twin (70/30): 79-91%
2. Zero-shot (all): 73-92%
3. Few-shot (select): 70-88%
4. Continuous (NL): 69-84%
5. Zero-shot (select): 74-88%
6. Few-shot (all): 68-89%

### Model Performance Ranking

**Average Accuracy (across all methods/domains):**
1. Gemini-2.5-Pro: ~31%
2. Grok-4-Fast: ~30%
3. GPT-5: ~29%
4. GPT-4o-mini: ~29%
5. DeepSeek-R1: ~28% (fewer methods available)

**Note:** Differences are small and not statistically significant (p > 0.05)

## Manuscript Recommendations

### Main Text Figures

**Figure 1: Radar Charts**
- Use `radar_all_methods_content.png` or `radar_all_methods_coping.png`
- Shows comprehensive multi-metric comparison
- Highlights that Digital Twin outperforms Generic LLM methods

**Figure 2: Heatmap**
- Use `heatmap_all_methods_accuracy.png`
- Dense visualization of all results
- Easy to see patterns across models/methods/domains

**Figure 3: Bar Chart**
- Use `bars_all_methods_accuracy.png`
- Direct model-to-model comparison
- Shows no clear winner among models

### Supplementary Figures

- All other heatmaps (Kappa, Acc±1)
- All other radar charts (other domains)
- Scatter plot (Kappa vs Accuracy)
- Bar charts for other metrics

### Key Messages for Manuscript

1. **Digital Twin > Generic LLM**: Digital twin methods (with participant history) outperform zero/few-shot by 5-10%

2. **Directional vs Exact**: Models achieve 70-90% directional accuracy but only 20-40% exact accuracy
   - Implication: Suitable for "better/worse" comparisons, not precise ratings

3. **No Model Dominance**: GPT-5, DeepSeek-R1, Gemini-2.5-Pro, Grok-4-Fast all perform similarly
   - Implication: Cost-effective models (Gemini, Grok) are competitive

4. **Domain Difficulty**: Content > Coping > Quitting
   - Implication: Objective criteria easier than subjective/personalized

5. **Personalization Challenge**: Low Spearman's ρ indicates zero/few-shot models don't capture individual preferences
   - Implication: Need participant-specific training (digital twins) or hybrid approaches

6. **Agreement vs Performance**: Models have higher accuracy than κ
   - Implication: Performing better than chance but not achieving strong inter-rater agreement

## Technical Details

- **Figure format:** PNG, 300 DPI (publication-ready)
- **Color scheme:** Husl palette (colorblind-friendly)
- **Style:** Seaborn paper style
- **Font sizes:** Optimized for readability in manuscripts
- **Missing data:** DeepSeek-R1 missing few-shot and digital twin results (not yet run)

## Regenerating Figures

If you need to regenerate with updated results:

```bash
# 1. Ensure all models have been evaluated
bash e2e_pipeline.sh

# 2. Regenerate figures
conda activate research
python analysis-script/create_comprehensive_figures.py

# 3. Check output
ls -lh figures/*.png
```

## Citation

If using these figures, please cite:
- **Cohen's Kappa:** Cohen, J. (1960). A coefficient of agreement for nominal scales.
- **Kendall's Tau:** Kendall, M. G. (1938). A new measure of rank correlation.
- **Spearman's Rho:** Spearman, C. (1904). The proof and measurement of association between two things.

## Questions?

For questions about the analysis or figures, see:
- `analysis-script/create_comprehensive_figures.py` - Main analysis script
- `README.md` - Project overview
- `IMPLEMENTATION_SUMMARY.md` - Methods documentation

