# Draft_final (1).docx Main-Text Before/After Edits
Date: 2026-05-27
Source Word draft: `/Users/yiqun/Downloads/Draft_final (1).docx`
Output purpose: paste-ready revision map for updating the main manuscript text, captions, and appendix language. This file does **not** edit the Word document automatically.
## Source Policy
- Main model-comparison claims should use the cleaned canonical PP 70/30 source unless explicitly described as contextual or supplemental.
- Missing split-specific methods are omitted rather than backfilled from another split.
- Historical participant ratings / response history should be described as the strongest observed personalization signal in this benchmark, not as a causal or universal feature-importance proof.
- Preserve the original Figure 1-4 manuscript architecture: update the original slots and captions rather than replacing the manuscript with unrelated new figures.

## Priority Map
| Priority | Where | What changes |
|---|---|---|
| 1 | Title, keywords, abstract | Remove LLM-superiority framing and make response history the central personalization signal. |
| 2 | Methods 2.2-2.4 | State strict shared-row design, k_train = 1/3/7, RF/LLM/hybrid definitions, and fixed Demo+History selection benchmark. |
| 3 | Results 3.1-3.5 | Replace old Results with strict four-method comparison, history-length sensitivity, QWK/Spearman, histograms, and message-selection gain. |
| 4 | Discussion and Conclusion | Reframe LLMs as complementary to strong supervised RF, not uniformly superior. |
| 5 | Figure captions and Appendix A2/A3 | Update figure/caption language and remove old bootstrap/digital-twin hierarchy claims. |

## Exact Before/After Edits

### Edit 01: Title
**Where:** Paragraph 0000 / manuscript title

**Action:** Replace the current title.

**BEFORE - current text in `Draft_final (1).docx`:**

```text
Personalized Prediction of Perceived Message Effectiveness Using Machine Learning and Large Language Models
```

**AFTER - paste this replacement:**

```text
Response-History-Informed Smoking-Cessation Message Evaluation
```

### Edit 02: Keywords
**Where:** Paragraph 0024 / keywords

**Action:** Replace the keyword line.

**BEFORE - current text in `Draft_final (1).docx`:**

```text
Key words: Large language models (LLMs);perceived message effectiveness (PME); smoking cessation; personalized interventions; Machine learning
```

**AFTER - paste this replacement:**

```text
Key words: Large language models (LLMs); supervised learning; response history; rating history; hybrid models; perceived message effectiveness (PME); smoking cessation; personalized interventions
```

### Edit 03: Full Abstract
**Where:** Paragraphs 0018-0023 / Abstract

**Action:** Replace the full abstract body. Keep the heading if the journal requires it.

**BEFORE - current text in `Draft_final (1).docx`:**

```text
Abstract (250/250)

Objective: Perceived message effectiveness (PME)  is important for selecting and optimizing personalized smoking cessation intervention messages for mobile health (mHealth) platform delivery. This study evaluates whether machine learning (ML) and large language models (LLMs) can accurately predict PME for smoking cessation messages.

Materials and Methods: We evaluated multiple models for predicting PME of smoking cessation messages across content quality, coping support, and quitting support. The dataset comprised 3,010 message ratings (5-point Likert-scale) from 301 young adult smokers. We compared (1) supervised learning models trained on PME histories, (2) zero-/few-shot LLMs, and (3) personalized-prompt (PP) LLMs, which incorporate individual characteristics and prior PME histories to generate personalized predictions. Model performance was assessed on held-out messages per participant using accuracy, weighted  κ, and F1.

Results: LLM-based digital twins outperformed zero-/few-shot (+12 percentage points on average) LLMs and supervised baselines (+13 percentage points), achieving accuracies of 0.49 (content), 0.45 (coping), and 0.49 (quitting), with corresponding directional accuracies of 0.75, 0.66, and 0.70 for simplified 3-point scale. PP-LLM predictions also showed greater dispersion across rating categories, indicating improved sensitivity to individual differences.

Discussion: We found that conditioning LLM predictions on both individual profiles and prior rating histories is associated with improved PME prediction and outperforms supervised learning and zero-/few-shot LLM approaches. This improved PME prediction could enable more tailored intervention content in mHealth.

Conclusion: LLM-based digital twin models show potential for predicting PME and may support personalization of mobile smoking cessation and other substance use and health behavior change interventions.
```

**AFTER - paste this replacement:**

```text
Abstract

Objective: Perceived message effectiveness (PME) is important for selecting and optimizing personalized smoking cessation messages for mobile health delivery. This study evaluated whether prior participant ratings support personalized PME prediction and whether large language model (LLM) and hybrid approaches add value beyond response-history-informed supervised learning.

Materials and Methods: We analyzed 3,010 five-point message ratings from 301 young adult smokers across content quality, coping support, and quitting support. We compared supervised learning, zero-/few-shot LLM prompting, and persona-conditioned LLMs (LLM-PP) using individual characteristics and prior PME histories. Evaluation was within-participant: prior ratings served as the personalization history, and held-out ratings were used only for evaluation. Performance was assessed on held-out messages using accuracy, macro-F1, quadratic weighted kappa (QWK), within-participant Spearman correlation, confusion matrices, and message-selection gain over random.

Results: All methods were evaluated on the same within-participant held-out rows (dt10 split; N = 898 with seven prior ratings). A response-history-informed supervised random forest (RF) was the strongest aggregate classifier: mean accuracy across domains was 0.474 for RF and 0.462 for persona-conditioned LLM prediction (LLM-PP); macro-F1 was 0.403 versus 0.370 and quadratic weighted kappa (QWK) 0.527 versus 0.488. Because evaluation was within-participant, the supervised model functioned as a strong participant-calibrated baseline rather than evidence that demographics predict PME; on an unseen-participant split its accuracy fell to roughly 0.35. LLM-PP did not surpass RF on aggregate metrics but selected better messages in the coping and quitting domains (gain over random at K = 5 of 0.57 and 0.54 versus 0.25 and 0.12 for RF) and was most competitive when little prior history was available. Performance improved as more prior participant ratings became available, identifying response history as the strongest observed personalization signal.

Discussion: Prior participant ratings were the strongest observed personalization signal in this benchmark. A response-history-informed supervised baseline captured substantial aggregate PME signal and was difficult to outperform, whereas persona-conditioned LLM scoring provided complementary value for ranking and selecting messages, particularly for coping and quitting and under limited history. These results support a calibrated, task-specific role for LLM-based personalization rather than uniform superiority over supervised learning.

Conclusion: These findings support response history as a central input for personalized smoking-cessation message evaluation and a calibrated role for LLM-based methods as complements to supervised benchmarks.
```

**Notes:** This removes the old +12/+13 percentage-point LLM-superiority claim and replaces it with the strict shared-row interpretation.

### Edit 04: Introduction Digital-Twin Motivation Paragraph
**Where:** Paragraph 0031 / Introduction

**Action:** Replace the paragraph beginning "On the other hand, another line of research...".

**BEFORE - current text in `Draft_final (1).docx`:**

```text
On the other hand, another line of research has focused on personalization and explored persona-based prompting and detailed contextual backstories to condition large language model outputs on individual attributes. This work is sometimes referred to as digital twin (Katsoulakis et al., 2024) approaches, which use individualized conditioning, such as prior decisions, clinical histories, behavioral trajectories, or psychometric profiles, to elicit personalized LLM outputs (Chen et al., 2025; R. Li et al., 2025; Makarov et al., 2025; Sprint et al., 2024; Toubia et al., 2025). By conditioning on real-world data, the “digital twins” aim to approximate not only typical human behavior but the idiosyncratic decision-making patterns of a particular individual, and could be useful to assist in development and refinement of health behavior change intervention approaches. For instance, Sprint et al. (2024) used this approach to predict patient cognitive health diagnosis. Taken together, developments in LLMs and LLM-based digital twins have the potential to dramatically reduce the time and financial resources required for early-stage hypothesis testing, intervention refinement, and product development, which is particularly valuable for mHealth personalized interventions such as those designed for smoking cessation. However, prior work has also noted important limitations of LLMs in reliably simulating individual-level human judgment (Argyle et al., 2023), which our empirical results both reflect and extend.
```

**AFTER - paste this replacement:**

```text
A second line of research has examined whether LLM outputs can be conditioned on person-level information, such as demographic characteristics, prior decisions, clinical histories, behavioral trajectories, or psychometric profiles. This approach is sometimes described as digital-twin prompting in adjacent LLM work, but in the present study we use the term more narrowly: persona-conditioned, history-augmented prediction of PME ratings. This distinction is important because predicting a participant's rating of a message is not equivalent to modeling longitudinal smoking behavior, physiological dynamics, or causal response to intervention. The practical question is therefore not whether LLMs form complete behavioral digital twins, but whether prior participant ratings can support personalized PME prediction and whether persona-conditioned LLM scores add predictive or message-selection value beyond response-history-informed supervised baselines.
```

### Edit 05: Final Introduction Paragraph
**Where:** Paragraph 0032 / Introduction

**Action:** Replace the final Introduction paragraph.

**BEFORE - current text in `Draft_final (1).docx`:**

```text
In this study, we analyzed data from a panel of young adult smokers who evaluated smoking cessation messages. Using these data, we systematically benchmarked a broad suite of LLMs and prompting techniques to predict PME at the individual level. Our contributions are twofold: (1) we explore predictive modeling of PME by evaluating seven different prompting strategies across five LLMs, and (2) we introduce a pilot analytic framework that generalizes to intervention-evaluation settings within and beyond smoking cessation, particularly under realistic personalization conditions characterized by limited per-person data.
```

**AFTER - paste this replacement:**

```text
In this study, we analyzed data from a panel of young adult smokers who rated smoking-cessation messages. We revised the analysis around a strict benchmark question: when evaluated on the same within-participant held-out rows, how do supervised RF and persona-conditioned LLM prediction compare for PME prediction and message selection as more participant rating history becomes available? Our contributions are threefold. First, we quantify how prior participant ratings support personalized PME prediction across content, coping, and quitting domains. Second, we provide a shared-row comparison of history-aware supervised and LLM methods with ordinal and rank-sensitive diagnostics, including QWK, confusion matrices, and within-participant Spearman correlation. Third, we evaluate practical message selection using gain over random while including supervised RF as a direct response-history comparator.
```

### Edit 06: Methods 2.2 Opening And Split Policy
**Where:** Paragraphs 0039-0041 / Section 2.2 Models

**Action:** Replace the current model overview and split description.

**BEFORE - current text in `Draft_final (1).docx`:**

```text
To explore how LLMs can support the design of more effective smoking cessation interventions, we implemented a series of models to predict multiple dimensions of smoking cessation message ratings. In this paper, we focus on three of the four rating dimensions, i.e., content, coping, and quitting, since design reflects the characteristics of the accompanying image. We excluded the design dimension for both methodological and practical considerations. Preliminary analysis revealed high correlations between design and content ratings, indicating that participants evaluated messages holistically rather than distinctly separating visual design elements from textual content. Additionally, given that current LLMs (October 2025) perform markedly better on text than image inputs, focusing our analysis on the three text-based ratings could improve modeling efficiency while maintaining the core predictive capabilities needed to assess PME, aligned with our primary interest in optimizing the text content of intervention messages.

We evaluated methods across three categories: (1) traditional supervised learning baselines, (2) LLMs with zero-/few-shot settings, and (3) “digital twin” approaches, implemented as LLMs conditioned on (i) structured participant profiles and (ii) participant-specific histories of prior messages and ratings. For the LLMs and digital twin categories, we evaluated five popular LLMs with default API parameters: GPT-4o-mini and GPT-5 (OpenAI), DeepSeek-R1 (DeepSeek), Grok-4-Fast (xAI), and Gemini-2.5-Pro (Google).

For all the models, we used within-participant splits (7 messages for training/history; 3 held out for testing). Supervised baselines were fit on the 7 labeled messages and evaluated on the 3 held-out messages. Zero-/few-shot LLMs received only the held-out message text (plus instructions/demonstrations). Digital-twin LLMs additionally received the participant profile and the 7 history message–rating pairs, but never the held-out messages or ratings. Prompts were programmatically assembled from message IDs to ensure held-out items could not be included in the history block; results are reported on held-out messages unless noted.
```

**AFTER - paste this replacement:**

```text
To evaluate response-history-informed personalization, we implemented supervised and LLM models to predict three participant-rated PME domains: content quality, coping support, and quitting support. We excluded the design domain from the primary analysis because it reflects visual presentation of the accompanying image and was highly correlated with content ratings in preliminary analyses. The revised main comparison focuses on two methods evaluated on the same within-participant held-out rows: supervised RF and persona-conditioned LLM prediction (LLM-PP). Generic zero-shot and few-shot LLM prompting results are retained as contextual analyses but are not mixed into the primary AI-vs-ML table because the revised primary analysis required strict shared-row comparisons with supervised RF.

For each participant, up to seven rated messages were used as training or personalization history, and the remaining messages were held out for evaluation; the same participants therefore appeared in both the history and held-out portions, so the benchmark is within-participant by design. This prior-rating history is the central personalization signal tested in the revised analysis. We report sensitivity to the number of available history ratings using k_train = 1, 3, and 7. All primary comparisons use the same held-out rows within a single supervised feature block (demographics) so that supervised and LLM methods are compared directly. LLM-PP used participant profile information plus available prior message-rating history, but never received held-out ratings. Because the supervised model and LLM-PP were scored on identical held-out rows, observed differences reflect the method rather than the evaluation setup.
```

**Notes:** This is the key Methods correction: the manuscript should state shared-row evaluation and history/test splitting before reporting any model ranking.

### Edit 07: Methods 2.2.1 Supervised Benchmark
**Where:** Paragraphs 0042-0043 / Section 2.2.1

**Action:** Replace the heading and paragraph.

**BEFORE - current text in `Draft_final (1).docx`:**

```text
2.2.1 Supervised learning models based on patient characteristics only

Two conventional machine learning models served as non-LLM baselines: a regularized logistic regression (L2 penalty with C=1.0) and a random forest classifiers (default parameters in scikit-learn; (Kramer, 2016). Both models utilized participant-level features only, including age, gender, race/ethnicity, nicotine dependence (time to first cigarette), cigarettes per day, and psychological flexibility. Model training and evaluation followed the same data splits applied to the LLM-based methods described above.
```

**AFTER - paste this replacement:**

```text
2.2.1 Supervised random forest benchmark

The primary supervised benchmark was a random forest classifier trained on labeled ratings from the training/history portion of the split. We used RF as the main supervised comparator because it was the strongest and most stable supervised baseline in the revised analyses. Feature blocks included demographics and, where specified, participant history summaries derived from prior ratings. The supervised model was evaluated only on held-out message ratings and was compared with LLM and hybrid methods on shared rows. Logistic regression was retained as a secondary supervised sensitivity analysis rather than the main comparator. For the revised Figure 2 source audit, we also added matched LR/RF baselines using message embeddings plus prior-rating history on the same cleaned canonical PP 70/30 source; these rows are presented as supervised reference lines rather than mixed into the LLM bar groups.
```

### Edit 08: Methods 2.2.2 Zero-Shot And Few-Shot Role
**Where:** Paragraphs 0044-0045 / Section 2.2.2

**Action:** Keep the section, but add a caveat at the start so it is not interpreted as the primary AI-vs-ML benchmark.

**BEFORE - current text in `Draft_final (1).docx`:**

```text
2.2.2 Zero-shot and few-shot LLMs

For LLMs, we compared five prompt strategies that varied in the amount of participant context and calibration structure provided to the model. We include the detailed prompts in Appendix A1.
```

**AFTER - paste this replacement:**

```text
2.2.2 Zero-shot and few-shot LLMs

Zero-shot and few-shot LLM prompting analyses were retained to contextualize the effect of persona conditioning, but they were not used as the primary AI-vs-ML benchmark because the revised primary analysis required shared-row comparisons with supervised RF and hybrid methods. For LLMs, we compared five prompt strategies that varied in the amount of participant context and calibration structure provided to the model. We include the detailed prompts in Appendix A1.
```

**Notes:** The five prompt descriptions that follow can stay, unless you want to move generic prompt-family details to the supplement.

### Edit 09: Methods 2.2.3 Persona-Conditioned LLM And Hybrids
**Where:** Paragraphs 0051-0053 / Section 2.2.3

**Action:** Replace the heading and main paragraph; keep the de-identification sentence.

**BEFORE - current text in `Draft_final (1).docx`:**

```text
2.2.3 LLM-based digital twins

This framework can be viewed as an extension of the few-shot LLM paradigm, with the few-shot examples now contextualized at the individual level, incorporating each participant’s persona and 7 historical responses (Toubia et al., 2025) to build participant profiles. A sensitivity analysis examining how digital twin performance changes as a function of the number of historical ratings available is provided in Appendix A3. Using these profiles, we evaluated two configurations: (1) a basic profile with all individual characteristics, and (2) an enhanced profile that also incorporated random-forest predictions as prior information. An example prompt for the digital twins is provided in Appendix A1. To assess whether performance gains reflected individual-level preference learning rather than thematic overlap between history and test messages, we additionally evaluated a configuration that incorporated CBT/ACT category labels into the digital twin prompt (Appendix A2).

All data transmitted to commercial LLM APIs were fully de-identified prior to use, with no personally identifiable information included in any prompt.
```

**AFTER - paste this replacement:**

```text
2.2.3 Persona-conditioned LLM prediction

LLM-PP used participant profile information and available prior message-rating history to generate persona-conditioned predictions for held-out messages. Thus, LLM-PP used the same available prior-rating history as personalization context but did not receive held-out ratings. We report LLM-PP as the primary persona-conditioned method; persona conditioning was implemented as an extension of the few-shot paradigm in which the in-context examples are the participant's own prior message-rating pairs.

All data transmitted to commercial LLM APIs were fully de-identified prior to use, with no personally identifiable information included in any prompt.
```

### Edit 10: Methods 2.3 Evaluation Metrics
**Where:** Paragraphs 0054-0056 / Section 2.3

**Action:** Replace the metric description.

**BEFORE - current text in `Draft_final (1).docx`:**

```text
2.3 Evaluation metrics

We evaluated model performance using five metrics. First, we used exact accuracy, defined as the proportion of predictions that matched the true message rating on the five-point Likert scale. This metric is straightforward to interpret but might be non-discriminative in the presence of class imbalance, e.g., our dataset skewed toward the two most positive categories (“Very good/Extremely helpful” and “Good/Very helpful”). Moreover, accuracy does not differentiate between types of prediction errors (e.g., over- versus under-prediction of the scores). Second, we used Cohen's κ, which quantifies the model-vs.-ground truth agreement after adjusting for chance and penalizes agreement that may arise from shared rating tendencies (i.e., marginal distributions). Third, we included macro-averaged F1 across the five Likert scales. F1 score balances precision (how many predicted positives were correct) and recall (how many actual positives were detected), and the macro-average allows each class to contribute equally to the overall performance, but might over-penalize errors on rare classes.

While accurately predicting the granular five classes was  our primary goal, practical message optimization often only needs the ratings to be directional correct. We therefore collapsed ratings into three categories, mapping Very helpful and Extremely helpful to positive, Somewhat helpful and Not at all helpful to negative, and retaining Neutral. Based on these categories, we computed (1) directional accuracy, which captures whether predictions align with the correct direction of human ratings, and (2) directional macro-F1, computed on directional agreement rather than exact rating match. We also report bootstrap confidence intervals for the metrics in Appendix A2, complementing the point estimates in the main text.
```

**AFTER - paste this replacement:**

```text
2.3 Evaluation metrics

We evaluated model performance using complementary aggregate, ordinal, and rank-sensitive metrics. Exact accuracy measured the proportion of held-out ratings for which the predicted five-point Likert category matched the true category. Macro-F1 weighted each rating category equally and was included because the rating distribution was imbalanced toward favorable responses. QWK was added as the primary ordinal agreement metric because it gives partial credit for near misses on the five-point scale and penalizes larger ordinal errors more strongly. We also report row-normalized confusion matrices to show where models over- or under-predicted rating categories.

To assess within-person ranking, we computed Spearman correlation between predicted and observed ratings within each participant and averaged across participants. This metric answers a different question from QWK: QWK measures all-row ordinal agreement, whereas within-participant Spearman evaluates whether a model ranks held-out messages correctly for the same participant. Spearman can be undefined when a model produces tied predictions within a participant, so we treat it as a secondary rank diagnostic rather than the sole evidence of personalization. Directional accuracy on a collapsed three-level scale was retained only as a secondary interpretability analysis. The main revised ordinal evidence is based on QWK and confusion matrices.
```

### Edit 11: Methods 2.4 Top-K Message Selection
**Where:** Paragraphs 0057-0058 / Section 2.4

**Action:** Replace the top-K evaluation description.

**BEFORE - current text in `Draft_final (1).docx`:**

```text
2.4 Top-K Message Selection Evaluation

From a translational perspective, investigators seek to leverage these models to prioritize intervention content for real-world deployment. To evaluate the practical utility of LLM-based message scoring, we assessed whether Digital Twin prompting enables efficient identification of highly rated messages. We treated ordinal rating scales as numeric (1–5) to enable quantitative comparison. For each domain, we selected the top-K messages (K = 5, 10, 15, 20, 25) using three strategies: (1) LLM predicted ratings with Digital Twin prompting, (2) a human "oracle" using actual human ratings, and (3) random selection.
```

**AFTER - paste this replacement:**

```text
2.4 Top-K Message Selection Evaluation

From a translational perspective, investigators may use predictive models to prioritize messages for review or deployment. We therefore evaluated message-selection gain over random. For each domain and method, we ranked messages by predicted score and calculated the mean human rating among the top K selected messages (K = 5, 10, and 25). We compared this value with the mean rating expected under random message selection and with a human-rating oracle that ranked messages by observed ratings. The revised selection analysis includes supervised RF and LLM-PP so that LLM-based selection is compared with a supervised baseline on the same held-out rows. To avoid domain-wise feature-set cherry-picking, all domains use the fixed Demographics + History + Message Embedding feature block. Confidence bands for gain over random use 95% normal-approximation intervals based on message-level standard errors (SD divided by sqrt(n)).
```

### Edit 12: Results Section Replacement
**Where:** Paragraphs 0061-0068 / Results Sections 3.1-3.3

**Action:** Replace the current Results opening and three current result paragraphs with the revised five-section structure below.

**BEFORE - current text in `Draft_final (1).docx`:**

```text
Figure 1 presents  the overview of the study design. Detailed descriptive analyses of participant characteristics can be found in Hamoud et al (Hamoud et al., 2025). We display the best-performing configuration for each model across rating domains in Figure 2. Additional results of the evaluation metrics can be found in Appendix A2. Overall, personalized LLM-based digital twins had the best performance. The best-performing personalized LLM pipelines achieved exact accuracies of 0.49 (content), 0.45 (coping), and 0.49 (quitting), and directional accuracies of 0.75, 0.66, and 0.70, respectively. These results outperformed both zero-/few-shot prompting (≤0.39) and supervised baselines (≤0.38). GPT-5's hybrid model achieved 0.47 accuracy (κ=0.25) for content, 0.43 (κ=0.24) for coping, and 0.45 (κ=0.27) for quitting. Grok-4-Fast's hybrid configuration attained the overall highest quitting accuracy (0.49, κ=0.30), while its digital twin achieved 0.49 (κ=0.24) for content and 0.45 (κ=0.24) for coping. Gemini-2.5-Pro's digital twin performed best for quitting (0.45, κ=0.26) and coping (0.43, κ=0.23). Digital twin personalization also elevated DeepSeek-R1 (0.44, κ=0.20 for content) despite modest zero-shot performance. Logistic regression and random forest baselines ranged from 0.30-0.38 accuracy with κ ≤0.12 across domains.

3.1 Comparison of modeling strategies

Zero-shot prompting across all LLMs produced 24.2-39.1% accuracy (mean 28-32% per model) with κ values near zero, only marginally exceeding supervised baselines. Incorporating six to ten examples in few-shot prompts yielded modest gains (≤4 percentage points). In contrast, digital twin personalization increased accuracy into the low-to-mid 40% range and raised κ to 0.18-0.24. Hybrid pipelines further improved performance for GPT-5 and Grok-4-Fast, adding 1-6 percentage points over the corresponding digital twin configurations. Performance heterogeneity across LLM vendors reached 10 percentage points within the same method family, reinforcing that model choice materially affects downstream accuracy.

Directional metrics highlight practical utility even when exact agreement is limited. Hybrid GPT-5 achieved 72.2% directional accuracy for content (Directional F1=0.56), 62.2% for coping (Directional F1=0.54), and 64.8% for quitting (Directional F1=0.55). Grok-4-Fast's hybrid quitting model yielded 70.0% directional accuracy (Directional F1=0.54). Macro-F1 scores remained modest (0.34-0.43), reflecting class imbalance and human variability, yet still exceeded supervised baselines by 8-12 points.

3.2 Distribution of predicted scores across models

Figure 3 shows the distribution of predicted scores across models. The true score distribution was concentrated in three categories, i.e., the one neutral and two more favorable categories, while the two negative categories were selected but less frequently. Traditional supervised learning models and zero-shot/few-shot LLMs tended to concentrate their predictions within one or two categories. In contrast, the digital twin models produced predictions that were more spread across all five categories.

3.3 Contrasting messages selected by LLM versus human raters

Figure 4 shows that LLM-selected messages, especially those chosen by Grok-4 and GPT-5, consistently achieved higher mean human ratings than random selection across all three domains. All LLMs fell behind the human oracle, but the gap decreased as more messages were selected, reflecting a ceiling effect in the finite message library whereby the oracle must include progressively lower-rated messages as K increases, while LLM selection curves remain flatter due to model uncertainty in ranking messages..
```

**AFTER - paste this replacement:**

```text
Figure 1 summarizes the revised study design and benchmark. Participants rated smoking-cessation messages across content, coping, and quitting domains. Each model received up to seven of a participant's ratings as personalization history and predicted that participant's held-out ratings; all methods were evaluated on the same held-out rows (within-participant dt10 split; N = 898 at k_train = 7). Because ratings were imbalanced toward favorable categories, we report accuracy together with macro-F1, QWK, confusion matrices, and within-participant Spearman correlation.

3.1 Strict shared-row comparison of supervised and LLM methods

Evaluated on identical held-out rows, a response-history-informed supervised random forest (RF) was the strongest aggregate classifier, with persona-conditioned LLM prediction (LLM-PP) close behind. At k_train = 7 (Demographics feature block, N = 898), mean accuracy across Content, Coping, and Quitting was 0.474 for RF and 0.462 for LLM-PP; macro-F1 was 0.403 and 0.370; and QWK was 0.527 and 0.488, respectively. To avoid mixing feature definitions, all three aggregate metrics are reported from the single Demographics feature block.

An important caveat governs how this aggregate result should be read. Because the benchmark is within-participant, each participant's demographic attributes act as a near-unique identifier (seven attributes expand to about 46 indicator features and roughly 55,000 possible profiles for 301 participants). The demographics-only RF therefore behaves as a participant-calibrated baseline that recovers each person's typical rating rather than a demonstration that demographics predict PME: a trivial rule that predicts each participant's most frequent prior rating reproduces RF accuracy (0.53, 0.45, and 0.47 for Content, Coping, and Quitting), and on a strict unseen-participant split the same model falls to approximately 0.35. We therefore interpret supervised RF as a strong same-participant reference, not as evidence of demographic generalization.

3.2 Where each method performs best, by domain

The aggregate ranking masked a clear domain pattern (Figure 2). For Content (message quality), RF was best on every metric (accuracy 0.523 vs 0.499; QWK 0.449 vs 0.402). For Coping and Quitting (helpfulness), the two methods were close on aggregate metrics—LLM-PP slightly exceeded RF on Quitting accuracy (0.457 vs 0.450)—but they diverged on the practical task of message selection, where RF was best for Content while LLM-PP was clearly better for Coping and Quitting (Section 3.5). Consistent with this, RF's within-participant Spearman correlation was undefined in all three domains, because the demographics-only model assigns one value per participant and cannot rank that participant's messages, whereas LLM-PP produced small but defined within-participant correlations (approximately 0.06).

3.3 Prediction improved with participant history, and the LLM was most competitive at low history

Performance improved as more participant history became available. Across k_train = 1, 3, and 7 (Demographics block), supervised RF accuracy increased from 0.420 to 0.448 and 0.474, and LLM-PP from 0.417 to 0.438 and 0.462; QWK increased from 0.418 to 0.447 and 0.527 for RF and from 0.417 to 0.457 and 0.488 for LLM-PP. The supervised advantage grew with history, consistent with RF leveraging more prior ratings per participant. At low history, however, the LLM was most competitive: the two methods were tied on QWK at k_train = 1, and LLM-PP slightly exceeded RF on QWK at k_train = 3 (0.457 vs 0.447) before RF pulled ahead at k_train = 7. Because realistic deployments seldom have many prior ratings for a new user, the low-history regime is the operationally relevant one, and it is where persona-conditioned LLM scoring is most competitive relative to the supervised baseline.

3.4 Rating distributions and confusion matrices

The observed rating distributions were concentrated in the neutral and favorable categories, with fewer low ratings. On the dt10 held-out set, ratings of 4-5 accounted for 68.2% of Content ratings, 61.6% of Coping ratings, and 61.7% of Quitting ratings. This imbalance helps explain why accuracy alone can overstate performance and why macro-F1, QWK, and confusion matrices are necessary. The rating-distribution figures are presented as data context rather than as evidence that any method captures individual differences by producing a wider score distribution.

3.5 Message-selection gain over random

Both methods selected messages with higher mean human ratings than expected under random selection, using the fixed Demographics + History + Message Embedding feature block. At K = 5, supervised RF produced the largest Content gain (0.389 vs 0.184 for LLM-PP), whereas LLM-PP produced the largest gains for Coping (0.570 vs 0.245) and Quitting (0.540 vs 0.123) and remained highest for Quitting at K = 10 (0.521). Thus model-assisted message selection is practical; the supervised baseline is the stronger selector for message quality (Content), while LLM-PP is the stronger selector for the two helpfulness domains.
```

**Notes:** This is the main text block that fixes the old interpretation. It also adds the missing history-length section and message-selection supervised comparator.

### Edit 13: Discussion Replacement
**Where:** Paragraphs 0070-0075 / Discussion

**Action:** Replace the current Discussion body paragraphs with the following revised Discussion body.

**BEFORE - current text in `Draft_final (1).docx`:**

```text
In this study, we evaluated the capacity of LLMs to predict the perceived effectiveness of smoking cessation intervention messages, an essential step towards personalized digital health interventions. Across three message domains (content, coping, and quitting), zero-shot and few-shot LLMs performend similarly (best accuracies: 0.39, 0.34, and 0.33, respectively) to traditional machine learning baselines (accuracies: 0.30-0.38). Notably, personalized digital twin models substantially outperformed all other approaches (best accuracies: 0.49 for content, 0.45 for coping, and 0.49 for quitting). When evaluated using directional accuracy metrics, which assess whether models correctly predicted the relative ordering of message effectiveness, the best-performing digital twin models achieved directional accuracies of 0.75 (content), 0.66 (coping), and 0.70 (quitting). In addition to higher accuracy, digital twin predictions exhibited greater dispersion across rating categories compared with traditional ML and zero /few-shot LLMs, indicating improved sensitivity to individual-level heterogeneity in message perception.

Digital twin-based modeling presents substantial opportunities to advance the development of personalized mHealth interventions for smoking cessation. For example, it enables the personalized selection of previously tested messages, as well as the evaluation of new messages for each individual, based on the preferences learned from their digital twin. In addition to LLM-powered digital twins that are conditioned on static history like our study, they can also be conditioned on dynamically changing information (Makarov et al., 2025; Silva & Vale, 2025). As wearable sensors, smartphones, and passive monitoring technologies continue to generate rich streams of real-world behavioral and physiological data, records such as an individual’s momentary behavior patterns, contextual states, and prior responses to treatment (Businelle et al., 2022; Hébert et al., 2025; Li et al., 2024; Lin et al., 2023; Luken et al., 2023; Thrul et al., 2025) could be valuable. Within smoking cessation, these data streams are particularly valuable given the pronounced heterogeneity in triggers, motivational trajectories, withdrawal symptoms, and relapse dynamics across individuals. By modeling these personalized patterns, digital twins may support the design of adaptive and context-aware intervention protocols, such as selecting coping messages most likely to be effective for a specific user at a specific moment or tailoring content to match dynamic risk states.

Our findings also highlight potential limitations of using models focusing on population level trends (e.g., traditional supervised learning and off-the-shelf zero-shot or few-shot LLMs) for identifying specific individual perceptions, such as PME. Although prior work has mostly applied regression and supervised learning models to identify linguistic or contextual features associated with PME (Hamoud et al., 2025; Solnick et al., 2021; Tripp et al., 2021), in our study, traditional models’ accuracy ranged from 30.3% to 37.6%. Similarly, despite growing evidence that zero-shot and few-shot LLMs (Brown et al., 2020; Li, 2023) can approximate human judgments, their performance in this task remained limited, achieving accuracies of only 0.33-0.39. In contrast, personalized digital twin models produced improvements, reaching 0.45-0.49 across the three message domains. While these gains highlight the promise of individual-level conditioning, the best-performing digital twins still leave room for improvement in fully predicting participants’ exact ratings. One plausible explanation is the inherently noisy nature of individual evaluative judgments. Even within-person consistency is imperfect: the test-retest reliability for individual choices in prior work has been estimated at approximately 81% (Park et al., 2024; Toubia et al., 2025). This suggests that a substantial proportion of variance in PME reflects intra-individual fluctuations, such as momentary affect, context, attention, or cognitive load, that are neither captured in static metadata nor easily recoverable from simple prompts.

Evaluated on directional consistency rather than exact category agreement, digital twin models achieved accuracies of 0.66–0.75, indicating that even when models missed the precise rating, they often correctly inferred positive, neutral, or negative PME scores. This distinction is meaningful because PME in our study relied on a 5-point scale, and prior work suggests that individuals may vary in how they apply such scales due to factors such as personality and culture (Kemmelmeier, 2016; Naemi et al., 2009; Pokropek et al., 2023). As a result, some individuals favor the endpoints of the scale, whereas others gravitate toward midpoints regardless of underlying judgments. Given this known variability in rating, the practical value of exact-category prediction for personalized intervention design remains an open question. For many applications in digital health, particularly those aimed at selecting or tailoring message content, a model’s capacity to correctly identify whether a message elicits a positive or negative reaction from an individual may be more consequential than matching the exact intensity of that reaction.

The distribution of predicted scores further helps contextualize why digital twin models outperformed both supervised learning and off-the-shelf zero-/few-shot LLMs. Digital twins’ broader score dispersion suggests that they were more sensitive to nuanced differences in individuals’ PME across messages, rather than defaulting to a narrow band of responses. This pattern is consistent with the fact that digital twins incorporate person-specific historical characteristics and rating data, and may reflect learning of individual preference patterns rather than population-level averages, though ablation experiments would be needed to confirm the relative contributions of each component. For mHealth interventions focused on personalization, such as tailoring text-based smoking cessation interventions, this sensitivity is particularly valuable.

Despite their promising performance, several limitations of this study warrant consideration and highlight important directions for future work. First, this study draws on data from 301 young adult smokers who were members of an online market research panel, which may limit generalizability. Future work should prioritize collecting larger, more diverse, and higher-quality datasets with repeated assessments per message to reduce intra-individual rating noise and improve the robustness of individual-level predictions.  Training digital twins on larger, more diverse samples would improve external validity and help determine whether the observed performance gains extend across demographic and behavioral subgroups. Second, the high noise of human ratings constrained the performance of the tested models. Future work may benefit from collecting repeated within-person assessments of the same messages across different time points and contexts, to better capture intra-individual variability in PME scores that static one-time ratings cannot reflect.. Third, except for the models reported in this paper, we also explored incorporating pretrained language model embeddings to provide richer semantic representations of message content, these features did not meaningfully improve predictive accuracy. This suggests that static embedding-based semantic information alone may be insufficient for modeling fine-grained, individual-level differences in PME. Fourth, incorporating domain knowledge about message categories (e.g., CBT vs. ACT) produced only modest performance gains. This suggests that while domain information has potential value, its impact may be limited in static modeling settings. Future research could explore integrating time-series data within both LLM-based digital twins and other modeling frameworks, including transformer-based sequence architectures. Finally, While PME is a useful proximal indicator of intervention message quality, it is a distal indicator of behavioral impact(O’Keefe, 2018). Future studies should examine the application of LLM-based digital twin approaches in real-world intervention settings to determine their ability to optimize behavioral outcomes. Future implementations may also not require the extensive self-report battery used in this study. Our analyses in Appendix A3 suggest that digital twin performance remained stable with as few as 3–5 prior message ratings, and zero-shot analyses with five selected features indicate that a lightweight intake profile may be sufficient for meaningful personalization. Future work should also conduct ablation experiments to disentangle the relative contributions of individual profile information, historical message texts, and historical ratings to digital twin performance. Future work could also benchmark digital twin approaches against established recommender system methods such as collaborative filtering, which represent a well-developed family of approaches to user-item rating prediction problems structurally similar to the task addressed here. Finally, real-world deployment of LLM-based digital twins in clinical settings would require careful attention to prediction errors and their potential impact on vulnerable individuals, such as those with high nicotine dependence during active craving episodes, necessitating human oversight and clinical validation before deployment.
```

**AFTER - paste this replacement:**

```text
In this study, we evaluated supervised and LLM approaches for predicting and selecting smoking-cessation messages based on perceived message effectiveness. The stricter shared-row benchmark revised the central interpretation of the original analysis. LLM-PP did not dominate supervised learning on aggregate PME classification. Instead, prior participant ratings were the strongest observed personalization signal: response-history-informed supervised RF was the strongest aggregate classifier, and LLM-based persona scoring remained useful as a complementary signal for ranking and selecting messages, especially in the coping and quitting domains and under limited history.

These findings support a more cautious and operationally useful view of LLM-based personalization. Rather than replacing supervised learning, persona-conditioned LLM scores were most valuable for specific tasks and conditions. The domain pattern was consistent: supervised RF was the stronger aggregate predictor and the better selector for message quality (Content), whereas LLM-PP was the better selector for coping and quitting support and was most competitive when little participant history was available. For mHealth message development, this suggests that LLM-based personalization should be evaluated as one component of a prediction-and-selection pipeline, matched to the domain and to the amount of available history, rather than as a standalone substitute for supervised modeling.

The revised results also show why strong supervised baselines are essential, but they must be interpreted carefully. Because the benchmark is within-participant, the demographics-only supervised model functioned as a participant-identifying baseline that recovered each person's typical rating: a trivial rule predicting each participant's most frequent prior rating reproduced its accuracy, and on an unseen-participant split its accuracy fell to roughly 0.35. Supervised RF should therefore be read as a strong same-participant reference rather than as evidence that demographic attributes generalize to predict PME. This finding places prior participant ratings, not demographics per se, at the center of the revised personalization interpretation, and it clarifies rather than negates the role of persona-conditioned LLMs: LLM-based scores were not uniformly superior for exact rating prediction, but they remained competitive in message-selection analyses—particularly for coping and quitting and at low history—and could be useful where flexible natural-language reasoning, explanation, or rapid scoring of new messages is needed.

Because PME ratings are ordinal and imbalanced, the revised analysis places greater emphasis on QWK and confusion matrices than on exact accuracy alone. QWK showed that supervised RF retained an ordinal-agreement advantage when ample history was available, whereas LLM-PP was competitive at low history. Within-participant Spearman correlation was modest and sensitive to tied predictions, which is expected when each participant contributes only a small number of held-out messages; notably, the demographics-only RF produced tied predictions within each participant and therefore had an undefined within-participant Spearman, whereas LLM-PP, which reads message content, produced small but defined within-participant correlations. We therefore interpret Spearman as a secondary diagnostic of within-participant ranking rather than as definitive evidence of individual-level simulation.

The revised distributional figures are best interpreted as context for class imbalance and model calibration, not as proof that wider prediction dispersion reflects better personalization. Because observed ratings were concentrated in neutral and favorable categories, models could achieve reasonable accuracy while still failing to represent rare low-rating classes. This motivated the addition of macro-F1, QWK, and confusion matrices in the revised analysis.

Several limitations should guide interpretation. First, the data came from 301 young adult smokers recruited through an online research panel, which may limit generalizability. Second, PME is a proximal perceptual endpoint and should not be interpreted as observed smoking behavior or cessation outcome. Third, within-participant rank estimates were based on a small number of held-out messages per participant and were sensitive to tied predictions. Fourth, generic zero-/few-shot LLM results were retained as contextual analyses but should not be interpreted as the primary apples-to-apples comparison against supervised RF. Fifth, the history-length analysis identifies response history as the strongest observed signal in this benchmark but does not prove a causal or universal ordering of feature importance because a clean history-only/profile-only/message-only ablation was not completed. Sixth, the message-selection analysis evaluates gain over random but is not a full recommender-system benchmark. Future work should evaluate these methods prospectively, include repeated PME assessments and behavioral endpoints, and test whether hybrid prediction-and-selection pipelines improve real-world intervention outcomes.
```

### Edit 14: Conclusion
**Where:** Paragraphs 0076-0077 / Conclusion

**Action:** Replace the current Conclusion paragraph. Keep the heading.

**BEFORE - current text in `Draft_final (1).docx`:**

```text
Conclusion

In conclusion, this study evaluated the efficacy of LLM-based approaches for predicting PME in the context of smoking cessation intervention messages. Digital twin models, which condition predictions on an individual’s historical data, demonstrated the highest accuracy. These findings highlight the potential of LLM-based digital twins to support the development of more personalized and effective mobile health interventions for smoking cessation, , though extension to other substance use and health behavior change interventions would require validation in those specific contexts and populations before broader claims can be supported.. Further validation is needed in larger, more diverse populations and in prospective studies designed to evaluate real-world optimization and behavioral outcomes.
```

**AFTER - paste this replacement:**

```text
Conclusion

In conclusion, this study benchmarked supervised and LLM approaches for personalized prediction and selection of smoking-cessation messages. Under strict within-participant shared-row evaluation, prior participant ratings were the strongest observed personalization signal: a response-history-informed supervised RF remained the strongest aggregate benchmark, while LLM-based persona-conditioned scoring contributed complementary value for ranking and selecting messages, especially for coping and quitting support and under limited history. These findings support a calibrated role for persona-conditioned LLMs in mHealth message evaluation, but they do not support broad claims that LLM personalization uniformly outperforms supervised learning. Future prospective studies should test whether these prediction-and-selection methods improve message engagement and smoking-cessation outcomes.
```

### Edit 15: Figure 1 Caption
**Where:** Paragraph 0133 / Figure 1 caption

**Action:** Replace caption and insert updated Figure 1 file in the original Figure 1 slot.

**BEFORE - current text in `Draft_final (1).docx`:**

```text
Figure 1. Overview of study design, model types, and key findings in evaluating large language models for predicting self-rated smoking-cessation message effectiveness. Panel 1 summarizes the study dataset of 301 young adults (ages 18–30) who rated 124 cognitive-behavioral and acceptance-commitment–based messages (3,010 total ratings). Panel 2 outlines the modeling approaches, including traditional supervised machine-learning baselines, zero-shot and few-shot large language models (LLMs), and personalized LLM-based “digital-twin” models. Panel 3 highlights our findings.
```

**AFTER - paste this replacement:**

```text
Figure 1. Revised study design and benchmark framework. Participants rated smoking-cessation messages across content, coping, and quitting domains. Within-participant history ratings were the central personalization input used to evaluate supervised RF and persona-conditioned LLM prediction (LLM-PP) on shared held-out rows (dt10 split). The revised analysis emphasizes strict apples-to-apples comparison, ordinal metrics, and message-selection gain over random.
```

**Notes:** Use figures/llm-message-paper-figure1.svg or figures/llm-message-paper-figure1.pdf. The PNG version also exists.

### Edit 16: Figure 2 Caption
**Where:** Paragraph 0134 / Figure 2 caption

**Action:** Replace caption and use the updated original Figure 2 bar-family files.

**BEFORE - current text in `Draft_final (1).docx`:**

```text
[Old Figure 2 caption in Draft_final (1).docx; replace it in full.]
```

**AFTER - paste this replacement:**

```text
Figure 2. Strict shared-row PME prediction on the dt10 split. Grouped bars compare supervised RF and persona-conditioned LLM prediction (LLM-PP) across Content, Coping, and Quitting on accuracy, macro-F1, and quadratic weighted kappa (QWK), using seven prior ratings as history (k_train = 7) and the single Demographics feature block (N = 898 held-out ratings, identical rows for both methods). RF was the strongest aggregate classifier overall and best for Content, whereas LLM-PP was competitive for Coping and Quitting. A separate context panel reports generic zero-shot and few-shot LLM prompting on the same split; because these cover a smaller subset (N ≈ 87 per cell), they are shown only as context and are not merged into the main comparison.
```

**Notes:** Figure 2 was regenerated on the unified dt10 split (script analysis-script/make_figure2_dt10.py). Main panel: figures/figure2/figure2_main_dt10_rf_vs_llmpp.png (and .pdf). Context panel (generic zero/few-shot, N≈87): figures/figure2/figure2_context_generic_llm.png (and .pdf). Source/provenance: figures/figure2/figure2_dt10_source_table.csv. The earlier PP 70/30 bar family (figures/figure2/bars_all_methods_*.png, source figures/figure2/bars_all_methods_source_table.csv) is superseded for the main text; retain only as a clearly labeled "older 70/30 prompt-family" supplementary panel if desired.

### Edit 17: Figure 3 Caption
**Where:** Paragraph 0135 / Figure 3 caption

**Action:** Replace caption and use the revised histogram/distribution files.

**BEFORE - current text in `Draft_final (1).docx`:**

```text
Figure 3. Distribution of predicted scores across models. Score frequency distributions in the content, coping and quitting domains across models and configurations. Histograms show the distribution of true participant scores (top row) and predicted scores from zero-shot/few-shot prompting, digital twin/hybrid approaches, and baseline machine-learning models (random forest and logistic regression). Columns correspond to language models.
```

**AFTER - paste this replacement:**

```text
Figure 3. Observed and predicted rating distributions across Content, Coping, and Quitting. Rows compare zero-shot (select), a combined supervised-learning row, few-shot (select), and PP. In the supervised row, RF uses demographics and LR uses demographics plus history and message embeddings. Gray bars show observed human ratings; colored grouped bars show model or supervised predictions on the five-point PME scale. The revised Results interpret these distributions as class-imbalance and calibration context, not as proof that any model simulates individual behavior better than the alternatives.
```

**Notes:** Updated files: figures/figure3/figure3_score_distributions_content.png, figures/figure3/figure3_score_distributions_coping.png, figures/figure3/figure3_score_distributions_quitting.png, and figures/figure3/figure3_score_distribution_source.csv.

### Edit 18: Figure 4 Caption
**Where:** Paragraph 0136 / Figure 4 caption

**Action:** Replace caption while preserving the original top-K Figure 4 slot.

**BEFORE - current text in `Draft_final (1).docx`:**

```text
Figure 4. Mean human rating of top-K messages selected by LLMs (Digital Twin prompting), human oracle, and random selection across three evaluation domains. Shaded gray region indicates 95% CI for random selection. LLM-selected messages consistently outperform random selection and approach human oracle performance, particularly for larger K. The decreasing trend in human oracle ratings with increasing K reflects a ceiling effect in the finite 124-message library, while the flatter LLM trajectories reflect model uncertainty in message ranking.
```

**AFTER - paste this replacement:**

```text
Figure 4. Top-K message-selection quality. For each domain and LLM, messages were ranked by predicted PME score, and the mean human rating of the top K selected messages was compared with random selection and a human-rating oracle. The revised manuscript preserves this original figure slot and reports the supervised RF versus LLM-PP gain-over-random comparison as a supporting method-level selection benchmark. In that supporting benchmark, supervised RF provides the response-history-informed comparator for model-assisted selection, using the fixed Demographics + History + Message Embedding feature block across all domains; RF selected the best Content messages while LLM-PP selected better Coping and Quitting messages. Shaded bands represent 95% normal-approximation intervals based on message-level standard errors.
```

**Notes:** Original Figure 4 files: figures/llm_selection_quality.png and figures/top_k_agreement_line.png. Supporting strict benchmark (regenerated on dt10, RF vs LLM-PP only, anchors dropped): figures/figure4/supporting_message_selection_gain_dt10.png (and .pdf), script analysis-script/make_figure4_supporting_dt10.py, source revision/figures/message_selection_methods_k7.csv. The older 4-method supporting_message_selection_gain.png is superseded.

### Edit 19: Appendix A2 Uncertainty Framing
**Where:** Paragraphs 0170-0175 / Appendix A2

**Action:** Replace the old LLM-only hierarchy language with a source-disciplined uncertainty framing.

**BEFORE - current text in `Draft_final (1).docx`:**

```text
Appendix A2: Uncertainty quantification for LLM performance metrics.

In this section, we present results from bootstrap analysis with n=1,000 resamples to quantify uncertainty in our performance estimates. For each bootstrap iteration, we resampled pairs of ground truth and predicted ratings with replacement and recomputed three metrics: (1) accuracy on the 5-point ordinal scale, (2) directional accuracy on a collapsed 3-point scale (low: 1–2, neutral: 3, high: 4–5), and (3) Cohen's kappa to assess agreement beyond chance. Confidence intervals are computed using the 2.5th and 97.5th percentiles of the bootstrap distributions (percentile method).

We listed the performance for digital twin (with CBT/ACT instructions); few shot (all features); and zero-shot (all features) in Tables S2-4. The bootstrap confidence intervals largely confirm the trends we summarized in the main text:

Across prompting strategies, the CIs reveal a clear hierarchy. Digital Twin prompting achieved statistically significant agreement (Cis for cohen’s kappa excluding zero) in 15/15 model-domain combinations, compared to only 2/15 for Few-shot and 1/15 for Zero-shot.

Across models within a prompting strategy, Grok-4-Fast achieved the highest 5-class accuracy (43.7%–48.1% across domains) and most consistent directional accuracy (65.9%–71.1%), with CIs that consistently rank among the highest. GPT-4o-mini showed the lowest accuracy with CIs that occasionally overlap with but generally fall below other models.

Finally, across domains, the CIs reveal that Coping is consistently the most difficult domain across models—showing the lowest accuracy (32.8%–43.7%) and directional accuracy with CI generally ranked below other domains.
```

**AFTER - paste this replacement:**

```text
Appendix A2: Uncertainty quantification for model performance metrics.

The bootstrap confidence intervals summarize uncertainty for the LLM-only prompt-family analyses. These analyses contextualize the effect of history-augmented prompting, but the main revised AI-vs-ML claims are based on strict shared-row comparisons between supervised RF and LLM-PP.

For the revised main benchmark, uncertainty and pairwise comparisons are reported using the cleaned canonical PP 70/30 source with known duplicate test items removed from every method. Accuracy confidence intervals, QWK summaries, pairwise significance tests, and confusion matrices are provided as supplementary revision artifacts.
```

**Notes:** Cite revision/figures/accuracy_confidence_intervals.csv, revision/figures/pairwise_significance_tests.csv, revision/figures/qwk_results.csv, and revision/figures/progress_summary/confusion_matrices_k7.csv as needed.

### Edit 20: Appendix A3 History-Length Sensitivity
**Where:** Paragraphs 0179-0182 / Appendix A3

**Action:** Replace the lightweight-onboarding/reliable-digital-twin language with a cautious history-length interpretation.

**BEFORE - current text in `Draft_final (1).docx`:**

```text
Appendix A3: Accuracy of digital-twin configurations by number of training ratings.

Supplementary Figure 1 illustrates the performance of the digital-twin prompt variants as a function of the number of historical ratings available for each participant (i.e., ratings of other messages used to condition the digital twin). Across models, accuracies remain remarkably stable even when using as few as one to seven prior message ratings. Accuracy exhibits substantially less variability than F1 or rank-based metrics such as Cohen’s κ, which are more sensitive to class imbalance.

These preliminary findings suggest that a lightweight onboarding procedure, such as collecting only a small seed set of message-rating examples, may be sufficient for initializing reliable personalized digital-twin predictions before delivering tailored intervention content.

Figure 5. Model prediction accuracy for digital twin prompt configurations across training set sizes. Different color indicates different models.
```

**AFTER - paste this replacement:**

```text
Appendix A3: Performance by number of prior participant ratings.

Supplementary Figure 1 illustrates model performance as a function of the number of prior participant ratings available for personalization. These sensitivity analyses suggest that a small number of prior ratings can improve model performance for both supervised RF and LLM-PP; the supervised advantage grew with more history, whereas LLM-PP was most competitive at low history (it was tied or slightly ahead on QWK at k_train = 1 and 3). These results support response history as the strongest observed personalization signal in the revised benchmark. Because the rank-sensitive metrics remain modest, these results should be interpreted as evidence that history is useful, not as proof that a lightweight onboarding procedure is sufficient for reliable individual-level simulation.

Supplementary Figure 1. Model performance across training-history sizes. Different colors indicate different model families or methods.
```

## Claims To Delete Or Rewrite Everywhere
Delete or replace these claims if they appear outside the specific paragraphs above:

- "LLM-based digital twins outperformed zero-/few-shot (+12 percentage points) LLMs and supervised baselines (+13 percentage points)."
- "Personalized LLM-based digital twins had the best performance."
- "These results outperformed both zero-/few-shot prompting and supervised baselines."
- "Macro-F1 scores exceeded supervised baselines by 8-12 points."
- "Personalized digital twin models substantially outperformed all other approaches."
- "The distribution of predicted scores explains why digital twin models outperformed supervised learning."
- "Digital twin models demonstrated the highest accuracy."

Replacement through-line:

> Prior participant ratings / response history were the strongest observed personalization signal in the revised benchmark. Response-history-informed supervised RF was the strongest aggregate classifier (a strong same-participant reference, given within-participant evaluation), while persona-conditioned LLM scores provided complementary value for message selection—especially for coping and quitting and at low history.

## Figure Files To Put Back Into Original Slots
| Original slot | Updated file(s) |
|---|---|
| Figure 1 | `figures/llm-message-paper-figure1.svg; figures/llm-message-paper-figure1.pdf; figures/llm-message-paper-figure1.png` |
| Figure 2 | `figures/figure2/figure2_main_dt10_rf_vs_llmpp.png` (main; .pdf available); `figures/figure2/figure2_context_generic_llm.png` (context, N≈87); source `figures/figure2/figure2_dt10_source_table.csv` |
| Figure 3 | `figures/figure3_score_distributions_content.png; figures/figure3_score_distributions_coping.png; figures/figure3_score_distributions_quitting.png` |
| Figure 4 | `figures/llm_selection_quality.png; figures/top_k_agreement_line.png; supporting strict benchmark: revision/figures/message_selection_gain.png` |

## Final Assembly QA
- [ ] Abstract no longer claims LLM/digital-twin aggregate superiority over supervised baselines.
- [ ] Methods explicitly define within-participant dt10 split, shared held-out rows, k_train = 1/3/7, supervised RF, and LLM-PP.
- [ ] Results include strict RF / LLM-PP values from the single Demographics block and history-length sensitivity.
- [ ] Results mention dt10 held-out class imbalance: Content 68.2%, Coping 61.6%, Quitting 61.7% ratings in categories 4-5.
- [ ] Results state the within-participant caveat: demographics-only RF is a participant-calibrated baseline (≈0.35 on an unseen-participant split), not demographic generalization.
- [ ] Figure 2 caption references the dt10 split and the supervised RF vs LLM-PP comparison, with generic zero/few-shot shown as N≈87 context.
- [ ] Figure 3 caption treats distributions as class-imbalance/calibration context, not personalization proof.
- [ ] Figure 4 caption preserves the original top-K LLM selection figure and points to the strict supervised/LLM/hybrid supporting benchmark.
- [ ] Limitations state that the history result is observed benchmark evidence, not causal or universal input-importance proof.
