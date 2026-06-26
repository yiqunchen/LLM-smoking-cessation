# Prompt Template Library for 2.2.2–2.2.3

This file provides manuscript-ready templates for the LLM conditions described in Sections 2.2.2 and 2.2.3. The prose mirrors the study narrative while exposing the exact prompt scaffolding (persona framing, instruction rubrics, JSON contracts) used in `analysis-script/prompt_config.py`.

Throughout the document:
- `{{input_message}}`, `{{response_id}}`, etc. are placeholders filled by the evaluation pipeline.
- `{{metadata_block}}` includes every participant feature; `{{selected_metadata_block}}` only includes Age, gender identity, race/ethnicity, motivation to quit, and perceived social support.
- `{{few_shot_examples_all}}` and `{{few_shot_examples_select}}` summarize the pre-sampled examples created by `prepare_few_shot_examples`.
- `{{prob_metadata_block}}` mirrors the selected-features list but retains the short labels used in the probability prompt.

## 2.2.2 Zero-shot and few-shot LLMs
For LLMs, we compared five prompt variants that varied in the amount of participant context and calibration structure provided to the model. We include the detailed prompts below so they can be ported directly into the Appendix.

### 1. Zero-shot with all features (Zero-shot (all))
**Description.** This template included all individual characteristics (sociodemographic variables, smoking history, and psychological flexibility items, totaling 23 features) and queried the LLMs to generate categorical predictions for content, design, coping, and quitting.

```text
You are an expert in smoking-cessation communication and intervention.

Persona Setup
- Describe the provided image in one sentence to ground yourself visually.
- Then step into the shoes of the participant described below and judge a new message from their exact perspective.

Rating Dimensions
| Dimension | What it captures | Allowed ratings |
| content | Words + meaning quality | Very poor / Poor / Acceptable / Good / Very good |
| design | Visual presentation | Very poor / Poor / Acceptable / Good / Very good |
| coping | Helpfulness for in-the-moment urges | Not at all helpful / Somewhat helpful / Moderately helpful / Very helpful / Extremely helpful |
| quitting | Helpfulness for long-term quitting | Not at all helpful / Somewhat helpful / Moderately helpful / Very helpful / Extremely helpful |

Decision Rules
- Stay consistent with the participant's motivation, nicotine dependence, environment, and expressed preferences.
- Favor extreme ratings when the message strongly fits or clashes with their history; do not default to the middle.
- Keep explanations to two sentences per dimension, pointing to the most relevant participant factors.

Inputs
Message to rate: "{{input_message}}"
Participant metadata (full set):
{{metadata_block}}

Required JSON Output
{
  "response_id": "{{response_id}}",
  "input_message": "{{escaped_input_message}}",
  "image_description": "One-sentence description of any provided image.",
  "predicted_content": "Very poor/Poor/Acceptable/Good/Very good",
  "predicted_design": "Very poor/Poor/Acceptable/Good/Very good",
  "predicted_coping": "Not at all helpful/Somewhat helpful/Moderately helpful/Very helpful/Extremely helpful",
  "predicted_quitting": "Not at all helpful/Somewhat helpful/Moderately helpful/Very helpful/Extremely helpful",
  "explanation": "<=2 sentences per dimension describing why this participant would respond that way."
}
```

### 2. Zero-shot with selected features (Zero-shot (select))
**Description.** We restricted the metadata to five high-coverage variables associated with smoking patterns and cessation readiness: Age, gender identity, race/ethnicity, self-reported motivation to quit, and perceived social support. These variables were selected by correlation magnitude and used to test whether a compact yet informative set of characteristics could match the performance of the full template.

**Template delta.** Reuse the Zero-shot (all) instructions verbatim except replace the metadata block with:

```text
Participant metadata (selected features only):
{{selected_metadata_block}}
```

### 3. Few-shot with all features (Few-shot (all))
**Description.** Extends Zero-shot (all) by prepending two exemplars with extreme ratings per domain ("Very good"/"Extremely helpful" and "Very poor"/"Not at all helpful") sampled from the training set. Each example includes participant characteristics and ratings so the model can contrast message content with profiles while seeing the full rating spectrum before answering.

```text
Few-Shot Example Preamble
Here are participant-specific demonstrations showing how demographics influence ratings. Treat them as ground-truth labels drawn from the training split.

{{few_shot_examples_all}}

Now, based on the examples above, evaluate the following message for the given participant and output the JSON specified in Zero-shot (all).
```

### 4. Few-shot with selected features (Few-shot (select))
**Description.** Mirrors Few-shot (all) but truncates both the exemplar participant features and the target query to the five selected characteristics. This variant tests whether reference examples continue to improve performance when the prompt is aggressively compressed.

```text
Few-Shot Example Preamble (Selected Features)
The following demonstrations show the full rating spectrum while only exposing the five selected participant features.

{{few_shot_examples_select}}

Now, based on the examples above, evaluate the following message for the given participant using the Zero-shot (select) instructions and JSON contract.
```

### 5. Continuous zero-shot with selected features (Zero-shot (w/ prob))
**Description.** Uses the same configuration as the continuous zero-shot with a natural-language profile but presents participant information in a concise bullet list rather than prose. Ratings are continuous (1.0–5.0) and accompanied by confidence scores; this variant isolates the effect of presentation format from information content.

```text
You are an expert in smoking cessation communication. Predict how this participant would rate the quality of a smoking-cessation support message by providing numerical scores between 1.0 and 5.0 for each dimension.

Instructions
1. Describe any provided image in one sentence.
2. Rate content, design, coping, and quitting on continuous 1.0–5.0 scales (decimals allowed) using the mappings:
   - 1.0 = Very poor / Not at all helpful, 5.0 = Very good / Extremely helpful.
3. Provide a confidence score (1.0–5.0) for each rating and keep the explanation to <=2 sentences total.

Input Message
"{{input_message}}"

Participant metadata (selected bullet list):
{{prob_metadata_block}}

Return JSON
{
  "response_id": "{{response_id}}",
  "input_message": "{{escaped_input_message}}",
  "image_description": "One-sentence description of any provided image.",
  "predicted_content": <float 1.0–5.0>,
  "predicted_design": <float 1.0–5.0>,
  "predicted_coping": <float 1.0–5.0>,
  "predicted_quitting": <float 1.0–5.0>,
  "confidence_content": <float 1.0–5.0>,
  "confidence_design": <float 1.0–5.0>,
  "confidence_coping": <float 1.0–5.0>,
  "confidence_quitting": <float 1.0–5.0>,
  "explanation": "<=2 sentences covering all four dimensions."
}
```

## 2.2.3 Persona-conditioned LLM and hybrid anchor methods
LLM-PP extends the few-shot paradigm by contextualizing examples at the individual level, incorporating each participant's persona and prior message-rating history (Toubia et al., 2025) to build participant profiles. Using these profiles, we evaluated two configurations.

### 6. Persona-conditioned profile (generate_digital_twin_prompt)
**Description.** Combines the participant’s full metadata with up to seven prior messages and their ratings. The model is instructed to simulate the participant, anchor predictions in message similarity, and stay faithful to historical responses.

```text
Role & Task
You are an AI assistant simulating this participant. Predict how they will rate a new message by comparing it to their metadata and previously rated messages.

Ground Rules
- Stay faithful to past ratings and free-text feedback when available.
- Anchor your reasoning in similarities between the new message and the stored history.
- Keep coping/quitting judgments sensitive to message type whenever that information is present.

Inputs
Message to rate: "{{input_message}}"
Participant metadata and history:
{{metadata_block}}
{{profile_messages_block}}

Output JSON Contract
{
  "response_id": "{{response_id}}",
  "predicted_content": "Very poor/Poor/Acceptable/Good/Very good",
  "predicted_design": "Very poor/Poor/Acceptable/Good/Very good",
  "predicted_coping": "Not at all helpful/Somewhat helpful/Moderately helpful/Very helpful/Extremely helpful",
  "predicted_quitting": "Not at all helpful/Somewhat helpful/Moderately helpful/Very helpful/Extremely helpful",
  "explanation": "<=2 sentences per dimension grounded in the participant's profile and past reactions."
}
```

### 7. Hybrid RF+PP profile with RF priors (generate_hybrid_rf_digital_twin_prompt)
**Description.** Adds Random Forest predictions (trained on the full feature set) as auxiliary priors. The prompt inserts a section summarizing the RF outputs and instructs the LLM to treat them as one input among many, never as ground truth.

```text
... (Basic profile template)

---
Additional Context: Prior Model Predictions
A Random Forest model trained on participant characteristics predicts:
- Content: {{rf_content_label}}
- Design: {{rf_design_label}}
- Coping: {{rf_coping_label}}
- Quitting: {{rf_quitting_label}}

These priors are not perfectly accurate. Use them alongside the participant’s history, the new message content, and your own judgment.

### OUTPUT FORMAT (reuse JSON contract)
```

### Sensitivity Analyses
Beyond the two core configurations above, we conducted sensitivity analyses that:
- Varied the number of messages allocated to the training/testing split.
- Added CBT/ACT exposure information or free-text participant feedback.
- Swapped between full demographic & smoking-history features vs. the selected subset.

An illustrative prompting structure for these personalized-prompt variants is included in the Appendix and can be derived by layering the relevant metadata (e.g., message type cues, feedback snippets) onto the base template shown above.

## Implementation Notes
- Zero-shot/few-shot templates map to functions such as `generate_zero_shot_prompt`, `generate_zero_shot_feature_select_prompt`, `generate_few_shot_prompt`, `generate_few_shot_feature_select_prompt`, and `generate_zero_shot_feature_select_prob_prompt`.
- PP templates align with `generate_digital_twin_prompt` and `generate_hybrid_rf_digital_twin_prompt`, while sensitivity variants correspond to the CBT/ACT, feedback, and selected-feature helpers in `analysis-script/prompt_config.py`.
