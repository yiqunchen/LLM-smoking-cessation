# Results Summary: LLM Model Comparison

## Key Findings (GPT-5 vs DeepSeek-R1)

### Overall Performance

#### GPT-5 Performance
- **Content Domain**: 36.9-39.1% accuracy (best method: Zero-shot + selected features)
- **Coping Domain**: 25.2-29.2% accuracy (best method: Zero-shot + selected features)
- **Quitting Domain**: 17.9-20.4% accuracy (best method: Zero-shot + selected features)

#### DeepSeek-R1 Performance
- **Content Domain**: 33.6-36.9% accuracy (best method: Zero-shot + selected features)
- **Coping Domain**: 24.5-28.7% accuracy (best method: Continuous + natural language)
- **Quitting Domain**: 16.1-24.1% accuracy (best method: Zero-shot + selected features)

### Statistical Significance (Paired Tests)

**GPT-5 vs DeepSeek-R1** on Continuous Method (5):

| Domain   | GPT-5 Acc | DeepSeek Acc | Difference | p-value | Significant? |
|----------|-----------|--------------|------------|---------|--------------|
| Content  | 36.8%     | 35.3%        | +1.5%      | 0.726   | No           |
| Coping   | 25.0%     | 28.7%        | -3.7%      | 0.313   | No           |
| Quitting | 17.6%     | 21.3%        | -3.7%      | 0.298   | No           |

**Conclusion**: No statistically significant differences between GPT-5 and DeepSeek-R1 at p<0.05 level.

### Agreement Beyond Chance (Cohen's Kappa)

Both models show **slight to fair agreement**:

- **Content**: κ = 0.034-0.067 (slight agreement)
- **Coping**: κ = -0.004 to 0.038 (minimal agreement)
- **Quitting**: κ = 0.012-0.054 (slight agreement)

**Interpretation**: Models perform only marginally better than chance agreement, suggesting the task is challenging even for state-of-the-art LLMs.

### Ordinal Classification Performance (Accuracy±1)

When allowing ±1 error (directional correctness):

- **Content**: 82.4-88.3% (most ratings within 1 point)
- **Coping**: 67.5-77.4% (moderate directional accuracy)
- **Quitting**: 47.8-60.2% (lower directional accuracy)

**Interpretation**: Models capture the general "direction" of ratings better than exact values, suggesting they understand message quality trends but struggle with precise 1-5 distinctions.

### Per-Participant Ranking (Spearman's ρ)

Average correlation for ranking messages within each participant:

- **GPT-5**: ρ = -0.206 to +0.121 (highly variable)
- **DeepSeek-R1**: ρ = -0.067 to +0.171 (slightly better)

**Interpretation**: Models struggle to correctly rank messages for individual participants, indicating limited personalization capability in zero/few-shot settings.

## Supplementary Model Results

### Best Performing Methods Across All Models

| Model          | Best Method             | Content | Coping | Quitting | Average |
|----------------|-------------------------|---------|--------|----------|---------|
| GPT-5          | 2. Zero-shot + select   | 39.1%   | 29.2%  | 20.4%    | 29.6%   |
| DeepSeek-R1    | 2. Zero-shot + select   | 36.9%   | 28.5%  | 24.1%    | 29.8%   |
| Gemini-2.5-Pro | 2. Zero-shot + select   | 38.0%   | 33.2%  | 28.5%    | 33.2%   |
| Grok-4-Fast    | 1. Zero-shot + all      | 36.9%   | 32.1%  | 28.1%    | 32.4%   |
| GPT-4o-mini    | 1. Zero-shot + all      | 36.5%   | 28.8%  | 23.4%    | 29.6%   |

**Surprising Finding**: Gemini-2.5-Pro and Grok-4-Fast show competitive or slightly better performance than GPT-5 and DeepSeek-R1 in some domains, despite being less emphasized as "reasoning" models.

### Variance Across Domains

**Content** (easiest): 30.7-39.1% accuracy across models
- Most objective domain (grammatical correctness, clarity)
- Highest agreement (κ up to 0.070)

**Coping** (medium): 24.5-34.3% accuracy across models
- Moderate difficulty
- Requires understanding of behavioral strategies

**Quitting** (hardest): 16.1-31.4% accuracy across models
- Most challenging domain
- Highly subjective and personalized
- Lowest agreement (κ often near 0)

## Digital Twin Learning Curves

Performance improves with more training examples per participant:

### Content Domain (GPT-5 CBT/ACT method)
- 10% training: ~32% accuracy
- 30% training: ~35% accuracy
- 70% training: ~38% accuracy
- 90% training: ~40% accuracy

**Trend**: Consistent improvement with more participant-specific history, suggesting personalization benefits from in-context learning.

### Diminishing Returns
- Largest gains: 10% → 30% training
- Smaller gains: 70% → 90% training
- Suggests: ~30-50% training data may be optimal balance

## Clinical Implications

### 1. Baseline Performance
- Even SOTA LLMs achieve only ~30-40% exact accuracy
- Human expert accuracy would be valuable comparison
- Current performance may not be clinical-grade without additional methods

### 2. Directional Accuracy More Promising
- 70-88% accuracy within ±1 rating
- Could be sufficient for: "Is this message better than that one?"
- Not sufficient for: "Rate this message exactly on 1-5 scale"

### 3. Personalization Challenges
- Low per-participant correlations (ρ < 0.2)
- Zero/few-shot models don't capture individual preferences well
- Digital twin approaches show promise but need more data

### 4. Domain-Specific Findings
- **Content**: Models perform best, likely generalizable
- **Coping/Quitting**: More subjective, need personalization
- Suggests: Hybrid approach (generic content eval + personalized preference)

## Methodological Insights

### Feature Selection Matters
- "Zero-shot + selected features" often outperforms "all features"
- Suggests: Information overload or irrelevant covariates hurt performance
- Recommendation: Careful feature engineering/selection

### Continuous Rating + Probabilities
- Mixed results compared to categorical prediction
- Provides calibration potential (for ensembles or confidence-based selection)
- Not clearly superior for accuracy alone

### Model-Specific Strengths
- **GPT-5**: Slightly better on Content (objective criteria)
- **DeepSeek-R1**: Slightly better on Quitting (reasoning-intensive)
- **Gemini-2.5-Pro**: Surprisingly competitive across all domains
- **Grok-4-Fast**: Strong performance, fast inference

## Recommendations for Manuscript

### Main Text Figures
1. **Table 1**: GPT-5 vs DeepSeek-R1 comparison (with CIs)
2. **Figure 2**: Kappa vs Accuracy scatter (shows agreement vs performance tradeoff)
3. **Figure 4**: Learning curves (demonstrates sample efficiency)

### Supplementary Figures
1. **Supplementary Table**: All 5 models, all metrics
2. **Figure 5**: Heatmaps (compact visualization for all models)
3. **Figure 3**: Radar charts (multi-metric comparison)

### Key Messages
1. "State-of-the-art LLMs achieve 30-40% exact accuracy, but 70-88% directional accuracy"
2. "No significant difference between GPT-5 and DeepSeek-R1 (p>0.05)"
3. "Performance improves with participant-specific training data (digital twins)"
4. "Content evaluation more accurate than coping/quitting (objectivity effect)"
5. "Gemini and Grok show competitive performance at potentially lower cost"

## Next Steps

### For Improved Performance
1. **Ensemble methods**: Combine multiple models (already implemented in `compare_llm_vs_individual.py`)
2. **Calibration**: Use temperature scaling on probability outputs
3. **Traditional ML + LLM embeddings**: Hybrid approach (already implemented)
4. **Fine-tuning**: Task-specific model adaptation (not yet explored)

### For Clinical Deployment
1. **Human-in-the-loop**: LLM pre-screening → human final decision
2. **Confidence thresholding**: Only use predictions with high confidence
3. **Active learning**: Prioritize uncertain cases for human annotation
4. **Iterative refinement**: Update models with new human-annotated data

### For Research
1. **Inter-rater reliability**: Compare LLMs to human expert variance
2. **Explanation analysis**: Understand *why* models make certain predictions
3. **Bias analysis**: Check for demographic biases in predictions
4. **Generalization**: Test on external smoking cessation message datasets

