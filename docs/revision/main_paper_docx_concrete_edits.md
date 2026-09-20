# Concrete Main-Paper Docx Edits

Date: 2026-06-02

Start with `START_HERE_revision_edit_checklist.md`. That file is the current
implementation checklist; this file contains the longer copy-paste prose blocks.

Use this as a docx edit guide. The goal is to revise the main manuscript so the
argument matches the newest within-participant (dt10) shared-row results. The
revised through-line is that, on the within-participant benchmark, supervised RF
is a strong participant-calibrated aggregate classifier while the
persona-conditioned LLM (LLM-PP) is competitive and complementary, especially at
low history and for coping/quitting message selection. The old main claim that
LLM-based digital twins outperform supervised baselines by more than 10
percentage points should be removed from the abstract, Results, Discussion, and
Conclusion. The comparison is now exactly two methods (supervised RF vs LLM-PP);
the earlier RF/LLM hybrid scoring methods are dropped from the manuscript
entirely and must not appear in any revised text.

## Global Terminology Edits

### Replace Title

**Current title**

> Personalized Prediction of Perceived Message Effectiveness Using Large
> Language Model-Based Digital Twins

**Replace with**

> Response-History-Informed Smoking-Cessation Message Evaluation

### Replace Keywords

**Current keywords**

> Large language models (LLMs); digital twins; perceived message effectiveness
> (PME); smoking cessation; personalized interventions

**Replace with**

> Large language models (LLMs); supervised learning; response history; rating
> history; random forest; perceived message effectiveness (PME); smoking
> cessation; personalized interventions

### Search And Replace Across The Docx

Use these replacements unless the sentence is explicitly describing prior
literature:

| Current wording | Preferred wording |
|---|---|
| LLM-based digital twins | persona-conditioned LLM models |
| digital twin models | persona-conditioned LLM models |
| digital twin personalization | response-history-informed prediction |
| outperformed supervised baselines | was benchmarked against supervised RF |
| substantially outperformed all other approaches | complemented strong supervised baselines |
| captured person-specific differences | leveraged prior participant ratings as personalization context |

If "digital twin" is retained anywhere, add the following definition in Methods:

> We use "digital-twin prompting" only as shorthand for persona-conditioned,
> history-augmented LLM prediction. We do not claim to model longitudinal
> smoking behavior, physiological dynamics, or causal treatment response.

## Historical-Ratings Through-Line

Implement this exact message throughout the docx:

> On the within-participant (dt10) benchmark, supervised RF was the strongest
> aggregate classifier (mean QWK 0.527) and the persona-conditioned LLM (LLM-PP)
> was competitive, especially at low history and for coping/quitting message
> selection. More history improved supervised RF (its edge grows with history),
> while LLM-PP was most competitive at low history / cold start and provided
> complementary ordinal and message-selection signal rather than uniform
> aggregate superiority. Because all 301 participants appear in both train and
> test, the supervised RF result is a participant-calibrated reference, not
> evidence that demographics predict PME.

Do not write this as a causal or universal "most important element" claim
unless a clean history-only/profile-only/message-only ablation is added.

Places to update:

1. Title and keywords: include "response history" or "rating history."
2. Abstract Objective: make the objective about response-history-informed
   personalized PME prediction, not digital-twin superiority.
3. Abstract Methods: identify prior participant ratings as the personalization
   context.
4. Abstract Results: include the history-length result across `k_train=1,3,7`.
5. Introduction: position the gap as whether a small number of prior ratings
   can personalize PME prediction and whether LLM scores add beyond that.
6. Methods 2.2: describe the within-participant history/test split as the core
   personalization design.
7. Methods 2.2.1: make history-aware RF the primary supervised comparator.
8. Methods 2.2.3: say LLM-PP uses the same available prior ratings as context.
9. Results 3.1: present the strict two-method RF vs LLM-PP comparison without
   LLM dominance.
10. Results 3.2: make history-length sensitivity the central personalization
    evidence.
11. Discussion first paragraph: lead with response history as the strongest
    observed signal.
12. Limitations: acknowledge that the analysis does not prove causal input
    importance.
13. Figure 1 caption: state that within-participant history ratings are the
    personalization substrate.
14. Figure 2 caption: say the benchmark compares history-aware supervised RF and
    LLM-PP on the same within-participant held-out rows.
15. Figure 4 caption/text or adjacent supporting panel: preserve the original
    top-K selection figure and define supervised RF selection as a
    response-history-informed comparator in the supporting method-level
    benchmark.
16. Appendix A3 / Supplementary Figure 1: describe the learning curves as
    history-length sensitivity.

## Abstract

### Replace The Entire Abstract With This

**Objective:** Perceived message effectiveness (PME) is important for selecting
and optimizing personalized smoking cessation messages for mobile health
delivery. This study evaluated whether prior participant ratings support
personalized PME prediction and whether a persona-conditioned large language
model (LLM) adds value beyond response-history-informed supervised learning.

**Materials and Methods:** We analyzed 3,010 five-point message ratings from
301 young adult smokers across content quality, coping support, and quitting
support. We compared a supervised random forest (RF) using demographic features
and a persona-conditioned LLM (LLM-PP) using individual characteristics and
prior PME histories, with zero-/few-shot LLM prompting reported as contextual
comparison. Both primary methods were evaluated on the same within-participant
held-out rows (dt10 split: 301 participants; N=898 at k_train=7). Within-
participant prior ratings served as the personalization history, and held-out
ratings were used only for evaluation. Performance was assessed on held-out
messages using accuracy, macro-F1, quadratic weighted kappa (QWK), within-
participant Spearman correlation, confusion matrices, and message-selection gain
over random.

**Results:** In strict shared-row comparisons with seven prior ratings (N=898),
supervised RF was the strongest aggregate classifier and LLM-PP was competitive.
Mean accuracy across domains was 0.474 for supervised RF and 0.462 for LLM-PP;
macro-F1 followed the same pattern (0.403 and 0.370). Supervised RF had the
highest QWK (0.527), with LLM-PP close behind (0.488). Because the benchmark is
within-participant (all 301 participants appear in both train and test), the
demographics-only RF functions as a participant-calibrated reference rather than
demographic generalization. Within-participant rank signal was modest and
tie-sensitive. LLM-PP was most competitive at low history: at k_train=1 it tied
RF on QWK and at k_train=3 slightly exceeded it (0.457 vs 0.447), with RF pulling
ahead at k_train=7. Both methods selected messages above random; LLM-PP achieved
the largest selection gains for coping and quitting.

**Discussion:** On the within-participant benchmark, supervised RF was the
strongest aggregate classifier and LLM-PP was competitive. Persona-conditioned
LLM predictions provided complementary PME signal but did not uniformly
outperform supervised RF on aggregate metrics. The participant-calibrated
supervised baseline captured substantial aggregate PME signal, while LLM-PP was
strongest at low history and for coping/quitting message selection.

**Conclusion:** These findings support response history as a central input for
personalized smoking-cessation message evaluation and a calibrated role for
LLM-based methods as complements to supervised benchmarks.

## Introduction

### Replace The Digital-Twin Motivation Paragraph

**Find text beginning**

> On the other hand, another line of research has focused on personalization
> and explored persona-based prompting...

**Replace with**

> A second line of research has examined whether LLM outputs can be conditioned
> on person-level information, such as demographic characteristics, prior
> decisions, clinical histories, behavioral trajectories, or psychometric
> profiles. This approach is sometimes described as digital-twin prompting in
> adjacent LLM work, but in the present study we use the term more narrowly:
> persona-conditioned, history-augmented prediction of PME ratings. This
> distinction is important because predicting a participant's rating of a
> message is not equivalent to modeling longitudinal smoking behavior,
> physiological dynamics, or causal response to intervention. The practical
> question is therefore not whether LLMs form complete behavioral digital
> twins, but whether prior participant ratings can support personalized PME
> prediction and whether persona-conditioned LLM scores add predictive or
> message-selection value beyond response-history-informed supervised
> baselines.

### Replace The Final Introduction Paragraph

**Find text beginning**

> In this study, we analyzed data from a panel of young adult smokers...

**Replace with**

> In this study, we analyzed data from a panel of young adult smokers who rated
> smoking-cessation messages. We revised the analysis around a strict benchmark
> question: when evaluated on the same within-participant held-out rows, how do
> supervised RF and persona-conditioned LLM prediction (LLM-PP) compare for PME
> prediction and message selection as more participant rating history becomes
> available? Our contributions are threefold. First, we quantify how prior
> participant ratings support personalized PME prediction across content,
> coping, and quitting domains. Second, we provide a shared-row comparison of
> supervised RF and LLM-PP with ordinal and rank-sensitive diagnostics,
> including QWK, confusion matrices, and within-participant Spearman
> correlation. Third, we evaluate practical message selection using gain over
> random while including supervised RF as a direct response-history comparator.

## Methods

### Replace Section 2.2 Opening

**Find text beginning**

> To explore how LLMs can support the design of more effective smoking
> cessation interventions...

**Replace with**

> We implemented a supervised random forest and a persona-conditioned LLM to
> predict three participant-rated PME domains: content quality, coping support,
> and quitting support. We excluded the design domain from the primary analysis
> because it
> reflects visual presentation of the accompanying image and was highly
> correlated with content ratings in preliminary analyses. The revised main
> comparison focuses on two methods evaluated on the same within-participant
> held-out rows: supervised RF (demographic features) and persona-conditioned
> LLM prediction (LLM-PP). Generic zero-shot and few-shot LLM prompting results
> are retained as contextual analyses but are not mixed into the primary
> AI-vs-ML table because they were evaluated on a smaller/different subset
> (N ≈ 87 at k_train=7).

> We used the dt10 within-participant split: each of 301 participants rated 10
> messages; for a chosen history length k, k messages served as training or
> personalization history and the remaining (10 − k) were held out for
> evaluation. At the primary k_train=7, this yields N=898 held-out ratings. This
> prior-rating history is the central personalization signal tested in the
> revised analysis. We report sensitivity to the number of available history
> ratings using `k_train=1,3,7`. All primary comparisons use the same held-out
> rows so that supervised RF and LLM-PP are compared directly. Because all 301
> participants appear in both train and test by design, this is a
> within-participant (personalization) benchmark, not unseen-participant
> generalization.

### Replace Section 2.2.1 Heading And Text

**Current heading**

> 2.2.1 Supervised learning models based on patient characteristics only

**Replace heading with**

> 2.2.1 Supervised random forest benchmark

**Replace paragraph with**

> The primary supervised benchmark was a random forest classifier trained on
> labeled ratings from the training/history portion of the dt10 split. We used RF
> as the main supervised comparator because it was the strongest and most stable
> supervised baseline in the revised analyses. The headline supervised model
> used a single feature block of seven demographic attributes (age, gender,
> race/ethnicity, Hispanic/Latino status, sexual orientation, education, and
> household income; ~46 one-hot columns), and excluded smoking, quit, and
> psychosocial variables. The supervised model was evaluated only on held-out
> message ratings and was compared with LLM-PP on the same held-out rows.
> Logistic regression and feature blocks that add participant history summaries
> and message embeddings were retained as secondary supervised sensitivity
> analyses rather than the headline comparator; the message-selection analysis
> uses the fixed Demographics + History + Message Embedding block across all
> domains.

### Revise Section 2.2.2

**Keep this section but change its role. Add this sentence at the start:**

> Zero-shot and few-shot LLM prompting analyses were retained to contextualize
> the effect of persona conditioning, but they were not used as the primary
> AI-vs-ML benchmark because the revised primary analysis required shared-row
> comparisons between supervised RF and LLM-PP.

### Replace Section 2.2.3 Heading And Text

**Current heading**

> 2.2.3 LLM-based digital twins

**Replace heading with**

> 2.2.3 Persona-conditioned LLM prediction

**Replace paragraph with**

> LLM-PP used participant profile information and available prior
> message-rating history to generate persona-conditioned predictions for held-
> out messages. Thus, LLM-PP used the same available prior-rating history as
> personalization context, but did not receive held-out ratings. LLM-PP was
> evaluated on the same within-participant held-out rows as supervised RF, so
> that the persona-conditioned LLM and the supervised baseline were compared
> directly on identical data.

### Replace Section 2.3 Evaluation Metrics

**Find text beginning**

> We evaluated model performance using five metrics.

**Replace with**

> We evaluated model performance using complementary aggregate, ordinal, and
> rank-sensitive metrics. Exact accuracy measured the proportion of held-out
> ratings for which the predicted five-point Likert category matched the true
> category. Macro-F1 weighted each rating category equally and was included
> because the rating distribution was imbalanced toward favorable responses.
> QWK was added as the primary ordinal agreement metric because it gives partial
> credit for near misses on the five-point scale and penalizes larger ordinal
> errors more strongly. We also report row-normalized confusion matrices to
> show where models over- or under-predicted rating categories.

> To assess within-person ranking, we computed Spearman correlation between
> predicted and observed ratings within each participant and averaged across
> participants. This metric answers a different question from QWK: QWK measures
> all-row ordinal agreement, whereas within-participant Spearman evaluates
> whether a model ranks held-out messages correctly for the same participant.
> Spearman can be undefined when a model produces tied predictions within a
> participant, so we treat it as a secondary rank diagnostic rather than the
> sole evidence of personalization.

**Move or de-emphasize the directional accuracy paragraph.** If retained, use:

> Directional accuracy on a collapsed three-level scale was retained only as a
> secondary interpretability analysis. The main revised ordinal evidence is
> based on QWK and confusion matrices.

### Replace Section 2.4 Top-K Message Selection Evaluation

**Find text beginning**

> From a translational perspective, investigators seek to leverage these models
> to prioritize intervention content...

**Replace with**

> From a translational perspective, investigators may use predictive models to
> prioritize messages for review or deployment. We therefore evaluated
> message-selection gain over random. For each domain and method, we ranked
> messages by predicted score and calculated the mean human rating among the
> top `K` selected messages (`K=5,10,25`). We compared this value with the mean
> rating expected under random message selection and with a human-rating oracle
> that ranked messages by observed ratings. The revised selection analysis
> includes supervised RF and LLM-PP so that LLM-based selection is compared with
> a supervised baseline on the same held-out rows. To avoid domain-wise
> feature-set cherry-picking, all domains use the fixed
> `Demographics + History + Message Embedding` supervised feature block.

## Results

### Replace The Opening Results Paragraph

**Find text beginning**

> Figure 1 presents the overview of the study design...

**Replace with**

> Figure 1 summarizes the revised study design and benchmark. Participants rated
> smoking-cessation messages across content, coping, and quitting domains, and
> models were evaluated on held-out ratings after within-participant
> history/test splitting. Prior ratings served as the personalization history
> for supervised RF and persona-conditioned LLM (LLM-PP) prediction. Because
> ratings were imbalanced toward favorable categories, we report accuracy
> together with macro-F1, QWK, confusion matrices, and within-participant
> Spearman correlation.

### Replace Section 3.1

**Current section**

> 3.1 Comparison of modeling strategies

**Replace section title with**

> 3.1 Strict shared-row comparison of supervised RF and LLM-PP

**Replace section text with**

> The strict shared-row comparison changed the interpretation of the original
> analysis. Rather than showing uniform LLM superiority, the updated results
> showed that supervised RF was the strongest aggregate classifier and LLM-PP
> was competitive. At `k_train=7` (N=898), mean accuracy across Content, Coping,
> and Quitting was 0.474 for supervised RF and 0.462 for LLM-PP. Macro-F1
> followed the same pattern, with values of 0.403 for supervised RF and 0.370
> for LLM-PP.

> Supervised RF also had the highest ordinal agreement: mean QWK was 0.527 for
> supervised RF and 0.488 for LLM-PP. Because the benchmark is
> within-participant (all 301 participants appear in both train and test), the
> demographics-only RF acts as a strong participant-calibrated reference rather
> than evidence that demographics predict PME; a trivial "predict each
> participant's most-frequent prior rating" rule reproduces RF accuracy
> (Content 0.53, Coping 0.45, Quitting 0.47), and on a strict unseen-participant
> split RF falls to ~0.35. LLM-PP remained close behind on all aggregate metrics
> and, as shown below, was most competitive at low history and for
> coping/quitting message selection.

### Add A New Section 3.2

**New section title**

> 3.2 Prediction improved with additional participant history

**New text**

> Performance generally improved as more participant history was available.
> Across `k_train=1,3,7` (Demographics block), supervised RF accuracy increased
> from 0.420 to 0.448 and 0.474, while LLM-PP increased from 0.417 to 0.438 and
> 0.462. QWK showed a notable interaction with history length: supervised RF QWK
> increased from 0.418 to 0.447 and 0.527, while LLM-PP QWK increased from 0.417
> to 0.457 and 0.488. LLM-PP tied RF on QWK at k=1 and slightly exceeded it at
> k=3 (0.457 vs 0.447); RF pulled ahead at k=7. These learning curves show that
> RF's aggregate advantage grows with history (consistent with its advantage
> being participant-history memorization), while LLM-PP is most competitive at
> low history / cold start, the regime most relevant to real deployments where
> few prior ratings are available per new user.

### Add A New Section 3.3

**New section title**

> 3.3 Ordinal agreement and within-participant rank signal

**New text**

> QWK and within-participant Spearman correlation answered different questions.
> QWK evaluated ordinal agreement across all held-out ratings, whereas Spearman
> evaluated whether models ranked held-out messages correctly within the same
> participant. In the strict `k_train=7` Demographics block, supervised RF had
> the highest mean QWK (0.527), with LLM-PP close behind (0.488).
> Within-participant Spearman values were smaller and more sensitive to tied
> predictions. Critically, the demographics-only supervised RF is message-blind:
> it assigns one value per participant, so its within-participant Spearman is
> undefined (tied predictions prevent rank correlation estimation). LLM-PP, by
> contrast, can rank a participant's messages and yielded a small but defined
> within-participant Spearman (~0.06). This is a qualitative advantage of LLM-PP
> for message ranking, even though supervised RF leads on aggregate QWK.
> Confusion matrices are therefore reported alongside QWK and Spearman to show
> the full ordinal error pattern.

### Replace Old Section 3.2 About Score Distributions

**Current section title**

> 3.2 Distribution of predicted scores across models

**Replace with**

> 3.4 Rating distributions and confusion matrices

**Replace text with**

> The observed rating distributions were concentrated in the neutral and
> favorable categories, with fewer low ratings. This imbalance helps explain
> why accuracy alone can overstate performance and why macro-F1, QWK, and
> confusion matrices are necessary. The revised histograms are presented as
> data-context figures rather than evidence that any method captures individual
> differences by producing a wider score distribution.

### Replace Old Section 3.3 About LLM Selection

**Current section title**

> 3.3 Contrasting messages selected by LLM versus human raters

**Replace with**

> 3.5 Message-selection gain over random

**Replace text with**

> In the revised message-selection analysis, both methods selected messages with
> higher mean human ratings than expected under random selection, but which
> method won depended on the domain. Using the fixed
> `Demographics + History + Message Embedding` feature block, the Content gain at
> `K=5` was larger for supervised RF (0.389) than for LLM-PP (0.184). For Coping,
> LLM-PP had the larger `K=5` gain (0.570) versus supervised RF (0.245). For
> Quitting, LLM-PP again led (0.540) versus supervised RF (0.123), and at `K=10`
> LLM-PP remained high (0.521). These results show that the two methods are
> complementary for message selection: supervised RF is best for message
> quality (Content), whereas LLM-PP selects substantially better messages in the
> helpfulness domains (Coping, Quitting), which is where the LLM earns its keep.

### Move Old Generic LLM/Prompt-Family Claims To Supplement

Move the following kinds of statements out of the main Results:

- "LLM-based digital twins outperformed zero-/few-shot by +12 percentage
  points."
- "Supervised baselines were <=0.38."
- "Macro-F1 exceeded supervised baselines by 8-12 points."
- "Digital twin models produced predictions that were more spread across all
  five categories, indicating improved sensitivity to individual-level
  heterogeneity."

If retained, use only this caveated supplemental sentence:

> In the LLM-only prompt-family comparison, history-augmented prompting
> improved over generic zero-/few-shot prompting; however, these results are
> contextual because the revised primary AI-vs-ML analysis uses strict
> shared-row comparisons between supervised RF and LLM-PP.

## Discussion

### Replace The First Discussion Paragraph

**Find text beginning**

> In this study, we evaluated the capacity of LLMs to predict...

**Replace with**

> In this study, we evaluated two approaches, supervised RF and a
> persona-conditioned LLM (LLM-PP), for predicting and selecting
> smoking-cessation messages based on perceived message effectiveness. The
> stricter shared-row benchmark revised the central interpretation of the
> original analysis. LLM-PP did not dominate supervised learning on aggregate
> PME classification. Instead, on this within-participant benchmark, supervised
> RF was the strongest aggregate classifier (QWK 0.527), acting as a strong
> participant-calibrated reference, while LLM-PP was competitive and contributed
> complementary value: it was most competitive at low history / cold start, it
> can rank a participant's messages (unlike the message-blind RF), and it
> selected better messages in the coping and quitting domains.

### Replace The "Digital Twin Opportunity" Paragraph

**Find text beginning**

> Digital twin-based modeling presents substantial opportunities...

**Replace with**

> These findings support a more cautious and operationally useful view of
> LLM-based personalization. Rather than replacing supervised learning,
> persona-conditioned LLM scores were complementary to a calibrated supervised
> baseline. LLM-PP was competitive with supervised RF on aggregate accuracy and
> macro-F1 and was strongest at low history and for coping/quitting message
> selection, whereas the participant-calibrated RF led on aggregate ordinal
> agreement (QWK). For mHealth message development, this suggests that LLM-based
> personalization should be evaluated as one component of a
> prediction-and-selection pipeline, not as a standalone substitute for
> supervised modeling.

### Replace The "Population-Level Trends" Paragraph

**Find text beginning**

> Our findings also highlight potential limitations of using models focusing
> on population level trends...

**Replace with**

> The revised results also show why strong supervised baselines are essential.
> On the within-participant benchmark, supervised RF was difficult to outperform
> on aggregate metrics. However, because all 301 participants appear in both
> train and test, the demographics-only RF largely reproduces each
> participant's most-frequent prior rating; a trivial modal-rating rule matches
> RF accuracy, and on a strict unseen-participant split RF falls to ~0.35. Thus
> supervised RF should be read as a strong participant-calibrated reference, not
> as evidence that demographics predict PME or that the model would generalize
> to new individuals. It does not negate the value of persona-conditioned LLMs;
> rather, it clarifies their role.
> LLM-based scores were not uniformly superior for exact rating prediction, but
> they remained competitive in message-selection analyses and could be useful
> where flexible natural-language reasoning, explanation, or rapid scoring of
> new messages is needed.

### Replace The Directional Accuracy Paragraph

**Find text beginning**

> Evaluated on directional consistency rather than exact category agreement...

**Replace with**

> Because PME ratings are ordinal and imbalanced, the revised analysis places
> greater emphasis on QWK and confusion matrices than on exact accuracy alone.
> QWK showed that supervised RF had the strongest aggregate ordinal agreement
> (0.527), with LLM-PP close behind (0.488). Within-participant Spearman
> correlation was modest and sensitive to tied predictions, which is expected
> when each participant contributes only a small number of held-out messages;
> the message-blind RF's within-participant Spearman was undefined, whereas
> LLM-PP yielded a small but defined value. We therefore interpret Spearman as a
> secondary diagnostic of within-participant ranking rather than as definitive
> evidence of individual-level simulation.

### Replace The Score-Dispersion Paragraph

**Find text beginning**

> The distribution of predicted scores further helps contextualize why digital
> twin models outperformed...

**Replace with**

> The revised distributional figures are best interpreted as context for class
> imbalance and model calibration, not as proof that wider prediction
> dispersion reflects better personalization. Because observed ratings were
> concentrated in neutral and favorable categories, models could achieve
> reasonable accuracy while still failing to represent rare low-rating classes.
> This motivated the addition of macro-F1, QWK, and confusion matrices in the
> revised analysis.

### Replace The Limitations Paragraph

**Find text beginning**

> Despite their promising performance, several limitations...

**Replace with**

> Several limitations should guide interpretation. First, the data came from
> 301 young adult smokers recruited through an online research panel, which may
> limit generalizability. Second, PME is a proximal perceptual endpoint and
> should not be interpreted as observed smoking behavior or cessation outcome.
> Third, within-participant rank estimates were based on a small number of
> held-out messages per participant and were sensitive to tied predictions.
> Fourth, generic zero-/few-shot LLM results were retained as contextual
> analyses but should not be interpreted as the primary apples-to-apples
> comparison against supervised RF. Fifth, the benchmark is within-participant by
> design: all 301 participants appear in both train and test, so the
> demographics-only supervised RF largely reproduces each participant's prior
> ratings and should be read as a participant-calibrated reference, not as
> evidence that demographics generalize to new individuals; on a strict
> unseen-participant split RF accuracy falls to ~0.35. Sixth, the
> message-selection analysis evaluates gain over random but is not a full
> recommender-system benchmark. Future work should evaluate these methods
> prospectively, include repeated PME assessments and behavioral endpoints, and
> test whether combined prediction-and-selection pipelines improve real-world
> intervention outcomes.

## Conclusion

### Replace The Entire Conclusion

**Current conclusion begins**

> In conclusion, this study evaluated the efficacy of LLM-based approaches...

**Replace with**

> In conclusion, this study benchmarked two approaches, supervised RF and a
> persona-conditioned LLM (LLM-PP), for personalized prediction and selection of
> smoking-cessation messages. Under strict within-participant shared-row
> evaluation, supervised RF was the strongest aggregate benchmark (QWK 0.527),
> functioning as a strong participant-calibrated reference, while LLM-PP was
> competitive and contributed complementary value, especially at low history and
> for coping/quitting message selection. These findings support a calibrated
> role for persona-conditioned LLMs in mHealth message evaluation, but they do
> not support broad claims that LLM personalization uniformly outperforms
> supervised learning. Future prospective studies should test whether these
> prediction-and-selection methods improve message engagement and
> smoking-cessation outcomes.

## Figure Captions

### Figure 1 Caption

**Replace with**

> Figure 1. Revised study design and benchmark framework. Participants rated
> smoking-cessation messages across content, coping, and quitting domains.
> Within-participant history ratings were the central personalization input
> used to evaluate supervised RF and persona-conditioned LLM prediction
> (LLM-PP) on the same shared held-out rows (dt10 split; 301 participants;
> N=898 at k_train=7). The revised analysis emphasizes strict apples-to-apples
> comparison, ordinal metrics, and message-selection gain over random.

### Figure 2 Caption

**Replace Figure 2 caption with**

> Figure 2. Within-participant (dt10) performance by method and domain. The main
> panel compares supervised RF (Demographics block) and persona-conditioned LLM
> prediction (LLM-PP) on the same held-out rows (N=898 at k_train=7), showing
> overall accuracy, macro-F1, and quadratic weighted kappa (QWK) for Content,
> Coping, and Quitting. A separate context panel shows generic zero-/few-shot
> LLM prompting (mean of five models) on a smaller, non-comparable coverage
> subset (N ≈ 87), explicitly labeled as different coverage and not merged into
> the 898-row comparison. A supplementary panel reports the remaining diagnostic
> metrics: Cohen's kappa, directional accuracy, directional macro-F1, and
> Kendall's tau.

### Figure 3 Caption

**Replace with**

> Figure 3. Observed and predicted rating distributions across Content,
> Coping, and Quitting. Rows compare zero-shot (select), a combined
> supervised-learning row, few-shot (select), and PP. In the supervised row, RF
> uses demographics and LR uses demographics plus history and message
> embeddings. Gray bars show observed human ratings; colored grouped bars show
> model or supervised predictions on the five-point PME scale. The revised
> Results should interpret these distributions as class-imbalance and
> calibration context, not as proof that any model simulates individual
> behavior better than the alternatives.

### Figure 4 Caption

**Replace with**

> Figure 4. Top-K message-selection quality. For each domain and method, messages
> were ranked by predicted PME score, and the mean human rating of the top
> `K` selected messages was compared with random selection and a human-rating
> oracle. The revised manuscript should preserve this original figure slot and
> report the supervised RF versus LLM-PP gain-over-random comparison as a
> supporting method-level selection benchmark. In that supporting benchmark,
> supervised RF provides the response-history-informed comparator for
> model-assisted selection, using the fixed `Demographics + History + Message Embedding`
> feature block across all domains. Selection winners differ by domain: RF for
> Content, LLM-PP for Coping and Quitting.

## Appendix And Supplement Edits

### Appendix A2

**Current language to remove**

> Across prompting strategies, the CIs reveal a clear hierarchy. Digital Twin
> prompting achieved statistically significant agreement...

**Replace with**

> The bootstrap confidence intervals summarize uncertainty for the LLM-only
> prompt-family analyses. These analyses contextualize the effect of
> history-augmented prompting, but the main revised AI-vs-ML claims are based
> on strict shared-row comparisons between supervised RF and LLM-PP.

### Appendix A3

**Current language to soften**

> These preliminary findings suggest that a lightweight onboarding procedure
> may be sufficient for initializing reliable personalized digital-twin
> predictions...

**Replace with**

> These sensitivity analyses suggest that a small number of prior ratings can
> improve model performance for both supervised RF and LLM-PP. Notably, LLM-PP
> is most competitive relative to RF at low history (it ties RF on QWK at k=1
> and slightly exceeds it at k=3), whereas RF's aggregate advantage grows as more
> history accumulates. Because the rank-sensitive metrics remain modest, these
> results should be interpreted as evidence that history is useful, not as proof
> that a lightweight onboarding procedure is sufficient for reliable
> individual-level simulation.

## Highest-Priority Deletes

Delete or replace every occurrence of these claims in the main paper:

1. "LLM-based digital twins outperformed zero-/few-shot (+12 percentage
   points) LLMs and supervised baselines (+13 percentage points)."
2. "Personalized LLM-based digital twins had the best performance."
3. "These results outperformed both zero-/few-shot prompting and supervised
   baselines."
4. "Macro-F1 scores ... exceeded supervised baselines by 8-12 points."
5. "Personalized digital twin models substantially outperformed all other
   approaches."
6. "The distribution of predicted scores explains why digital twin models
   outperformed supervised learning."
7. "Digital twin models demonstrated the highest accuracy."

The replacement message should be consistent everywhere:

> On the within-participant benchmark, supervised RF was the strongest aggregate
> classifier (QWK 0.527), functioning as a strong participant-calibrated
> reference, while persona-conditioned LLM (LLM-PP) prediction was competitive
> and provided complementary value, especially at low history / cold start and
> for coping/quitting message selection.
