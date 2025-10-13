import random
import pandas as pd

# --- Configuration for Selected Features ---

# A subset of features to be used in feature-selected and few-shot prompts
SELECTED_FEATURES = [
    "age_years", "gender_identity", "race_ethnicity", 
    "quit_motivation_level", "social_support_to_quit"
]

# --- Zero-Shot Prompt (Original) ---

def generate_zero_shot_prompt(data: dict) -> str:
    """Generates a prompt with all available metadata (original behavior)."""
    return _generate_prompt_base(data, use_all_features=True)

# --- Zero-Shot Prompt (Feature-Selected) ---

def generate_zero_shot_feature_select_prompt(data: dict) -> str:
    """Generates a prompt with only a selected subset of metadata."""
    return _generate_prompt_base(data, use_all_features=False)

# --- Zero-Shot Prompt (Feature-Selected with Balanced Instructions) ---

def generate_zero_shot_feature_select_balanced_prompt(data: dict) -> str:
    """Generates a prompt with only a selected subset of metadata and explicit instructions to use the full rating range."""
    return _generate_prompt_base(data, use_all_features=False, is_balanced=True)

# --- Few-Shot Prompt Preparation and Generation ---

FEW_SHOT_EXAMPLES = {}
FEW_SHOT_EXAMPLES_BALANCED = {}

def prepare_few_shot_examples(all_question_data: dict):
    """
    Pre-samples and stores high/low rated examples for each dimension to be used in few-shot prompts.
    This should be called once before starting the evaluation.
    """
    global FEW_SHOT_EXAMPLES, FEW_SHOT_EXAMPLES_BALANCED
    if FEW_SHOT_EXAMPLES and FEW_SHOT_EXAMPLES_BALANCED:  # Don't re-prepare if already done
        return

    print("Preparing examples for few-shot prompts...")
    
    # Convert dict to list of tuples for easier processing
    dataset = [(qid, qdata) for qid, qdata in all_question_data.items()]
    
    # Define rating dimensions and their high/low values
    dimensions = {
        'content': {"high": "Very good", "low": "Very poor"},
        'design': {"high": "Very good", "low": "Very poor"},
        'coping': {"high": "Extremely helpful", "low": "Not at all helpful"},
        'quitting': {"high": "Extremely helpful", "low": "Not at all helpful"}
    }

    # Prepare standard few-shot examples (high/low only)
    for dim, values in dimensions.items():
        high_val, low_val = values['high'], values['low']
        
        # Find one example for the high rating
        high_example = next((d for _, d in dataset if d['ratings'].get(dim) == high_val), None)
        
        # Find one example for the low rating
        low_example = next((d for _, d in dataset if d['ratings'].get(dim) == low_val), None)

        if high_example and low_example:
            FEW_SHOT_EXAMPLES[dim] = {'high': high_example, 'low': low_example}
        else:
            print(f"Warning: Could not find examples for dimension '{dim}'")

    # Prepare balanced few-shot examples (all 5 categories)
    all_ratings = {
        'content': ["Very poor", "Poor", "Acceptable", "Good", "Very good"],
        'design': ["Very poor", "Poor", "Acceptable", "Good", "Very good"],
        'coping': ["Not at all helpful", "Somewhat helpful", "Moderately helpful", "Very helpful", "Extremely helpful"],
        'quitting': ["Not at all helpful", "Somewhat helpful", "Moderately helpful", "Very helpful", "Extremely helpful"]
    }

    for dim, rating_values in all_ratings.items():
        examples_by_rating = {}
        for rating in rating_values:
            # Find examples for each rating level
            examples = [d for _, d in dataset if d['ratings'].get(dim) == rating]
            if examples:
                # Randomly select one example for this rating
                examples_by_rating[rating] = random.choice(examples)
            else:
                print(f"Warning: No examples found for {dim} = {rating}")
        
        if examples_by_rating:
            FEW_SHOT_EXAMPLES_BALANCED[dim] = examples_by_rating

    print(f"Standard few-shot examples prepared for {len(FEW_SHOT_EXAMPLES)} dimensions.")
    print(f"Balanced few-shot examples prepared for {len(FEW_SHOT_EXAMPLES_BALANCED)} dimensions.")


def generate_few_shot_prompt(data: dict) -> str:
    """
    Generates a prompt that includes pre-sampled high/low examples for each dimension,
    using all available metadata.
    """
    if not FEW_SHOT_EXAMPLES:
        raise ValueError("Few-shot examples have not been prepared. Call prepare_few_shot_examples() first.")

    # Start with examples
    examples_text = "Here are some examples of how participants with certain demographics have rated other messages:\n\n"
    
    for dim, examples in FEW_SHOT_EXAMPLES.items():
        for example_type, example_data in examples.items():
            examples_text += f"--- Example ({dim.capitalize()} - {example_type.capitalize()}) ---\n"
            examples_text += "Participant Demographics:\n"
            for key, value in example_data['metadata'].items():
                if pd.notna(value):
                    examples_text += f"- {key}: {value}\n"
            
            examples_text += f"\nMessage: \"{example_data['input_message']}\"\n"
            examples_text += f"Participant's Ground Truth Rating for {dim.capitalize()}: {example_data['ratings'][dim]}\n"
            examples_text += "---\n\n"

    # Generate the base prompt for the actual question to evaluate
    question_prompt = _generate_prompt_base(data, use_all_features=True, is_few_shot=True)
    
    return examples_text + question_prompt


def generate_few_shot_feature_select_prompt(data: dict) -> str:
    """
    Generates a prompt that includes pre-sampled high/low examples for each dimension,
    using only a selected subset of metadata.
    """
    if not FEW_SHOT_EXAMPLES:
        raise ValueError("Few-shot examples have not been prepared. Call prepare_few_shot_examples() first.")

    # Start with examples
    examples_text = "Here are some examples of how participants with certain demographics have rated other messages:\n\n"
    
    for dim, examples in FEW_SHOT_EXAMPLES.items():
        for example_type, example_data in examples.items():
            examples_text += f"--- Example ({dim.capitalize()} - {example_type.capitalize()}) ---\n"
            examples_text += "Participant Demographics:\n"
            for key, value in example_data['metadata'].items():
                if key in SELECTED_FEATURES and pd.notna(value):
                    examples_text += f"- {key}: {value}\n"
            
            examples_text += f"\nMessage: \"{example_data['input_message']}\"\n"
            examples_text += f"Participant's Ground Truth Rating for {dim.capitalize()}: {example_data['ratings'][dim]}\n"
            examples_text += "---\n\n"

    # Generate the base prompt for the actual question to evaluate
    question_prompt = _generate_prompt_base(data, use_all_features=False, is_few_shot=True)
    
    return examples_text + question_prompt


def generate_few_shot_feature_select_balanced_prompt(data: dict) -> str:
    """
    Generates a prompt that includes examples from ALL 5 rating categories for each dimension,
    using only selected metadata, with explicit instructions to use the full rating range.
    """
    if not FEW_SHOT_EXAMPLES_BALANCED:
        raise ValueError("Balanced few-shot examples have not been prepared. Call prepare_few_shot_examples() first.")

    # Start with examples from all 5 categories
    examples_text = """IMPORTANT: The examples below show the FULL RANGE of possible ratings. Notice that participants DO rate messages at ALL extremes - from "Very poor" to "Very good" and from "Not at all helpful" to "Extremely helpful". DO NOT shy away from predicting extreme ratings when they are warranted.

Here are examples showing how participants with various demographics have rated messages across ALL rating categories:

"""
    
    for dim, examples_by_rating in FEW_SHOT_EXAMPLES_BALANCED.items():
        examples_text += f"=== {dim.capitalize()} Examples (All Rating Levels) ===\n"
        for rating, example_data in examples_by_rating.items():
            examples_text += f"--- {rating} ---\n"
            examples_text += "Participant Demographics:\n"
            for key, value in example_data['metadata'].items():
                if key in SELECTED_FEATURES and pd.notna(value):
                    examples_text += f"- {key}: {value}\n"
            
            examples_text += f"\nMessage: \"{example_data['input_message']}\"\n"
            examples_text += f"Participant's Rating for {dim.capitalize()}: {rating}\n"
            examples_text += "---\n"
        examples_text += "\n"

    examples_text += """
REMEMBER: Based on these examples, you can see that:
1. Participants DO give extreme ratings ("Very poor", "Very good", "Not at all helpful", "Extremely helpful")
2. Consider ALL rating categories - don't avoid the extremes
3. Match your predictions to what similar participants would actually rate
"""

    # Generate the base prompt for the actual question to evaluate
    question_prompt = _generate_prompt_base(data, use_all_features=False, is_few_shot=True, is_balanced=True)
    
    return examples_text + question_prompt


# --- Continuous Rating Prompt (for calibration) ---

def generate_continuous_rating_prompt(data: dict) -> str:
    """
    Generates a prompt that asks for continuous numerical ratings between 1-5 
    instead of categorical ratings, enabling calibration.
    """
    return _generate_continuous_prompt_base(data, use_all_features=False)

def _generate_continuous_prompt_base(data: dict, use_all_features: bool) -> str:
    """
    A helper to generate prompts for continuous rating prediction.
    """
    prompt = """
You are an expert in smoking cessation communication and intervention. Your first task is to describe the image provided in a single sentence.
After that, you are tasked with predicting how a participant would rate the quality of a smoking cessation support message. 
There are four different dimensions: 
1. content (How would you rate the content, that is, the words and meaning of this message), 
2. design (How would you rate the design, that is, how the message looks),
3. coping (How helpful would this message be to support you in coping with a smoking urge or craving), 
4. quitting (How helpful would this message be to support you in quitting or reducing smoking).

IMPORTANT: Provide your ratings as CONTINUOUS NUMERICAL SCORES between 1.0 and 5.0 (with decimals allowed).

For content and design quality:
- 1.0 = Very poor
- 2.0 = Poor  
- 3.0 = Acceptable
- 4.0 = Good
- 5.0 = Very good
- Use any decimal between these values (e.g., 2.3, 3.7, 4.1)

For coping and quitting helpfulness:
- 1.0 = Not at all helpful
- 2.0 = Somewhat helpful
- 3.0 = Moderately helpful  
- 4.0 = Very helpful
- 5.0 = Extremely helpful
- Use any decimal between these values (e.g., 1.8, 3.2, 4.6)

Here is the message provided to the participant:

"{message}"

And here is the demographics for the participant, use these information to embed yourself as
a member of the group:

Participant metadata:
""".format(message=data['input_message'])

    # Determine which metadata features to include
    features_to_include = data['metadata'].keys() if use_all_features else SELECTED_FEATURES
    
    for key in features_to_include:
        value = data['metadata'].get(key)
        if pd.notna(value):
            prompt += f"- {key}: {value}\n"

    # End of the prompt
    prompt += """
Return your response in the following JSON format:
{{
"response_id": "{response_id}",
"input_message": "{input_message}",
"image_description": "List the text shown on the image here.",
"predicted_content": [numerical score between 1.0 and 5.0],
"predicted_design": [numerical score between 1.0 and 5.0],
"predicted_coping": [numerical score between 1.0 and 5.0],
"predicted_quitting": [numerical score between 1.0 and 5.0],
"confidence_content": [how confident are you in this rating, scale 1.0-5.0],
"confidence_design": [how confident are you in this rating, scale 1.0-5.0],
"confidence_coping": [how confident are you in this rating, scale 1.0-5.0],
"confidence_quitting": [how confident are you in this rating, scale 1.0-5.0],
"explanation": "Replace this with a very brief explanation (AT MOST 2 sentences!) for each predicted dimension."
}}
""".format(response_id=data['response_id'], input_message=data['input_message'].replace('"', '\\"'))

    return prompt


# --- Helper Function for Prompt Generation ---

def _generate_prompt_base(data: dict, use_all_features: bool, is_few_shot: bool = False, is_balanced: bool = False) -> str:
    """
    A helper to generate the core prompt structure.
    """
    # Start of the prompt
    if is_few_shot:
        if is_balanced:
            prompt = "\nNow, based on the examples above showing the FULL RANGE of ratings, evaluate the following message for the given participant. Remember to consider ALL rating categories and use extreme ratings when appropriate:\n\n"
        else:
            prompt = "Now, based on the examples above, evaluate the following message for the given participant:\n\n"
    else:
        prompt = ""
        
    prompt += """
You are an expert in smoking cessation communication and intervention. Your first task is to describe the image provided in a single sentence.
After that, you are tasked with predicting how a participant would rate the quality of a smoking cessation support message. 
There are four different dimensions: 
1. content (How would you rate the content, that is, the words and meaning of this message), 
2. design (How would you rate the design, that is, how the message looks),
3. coping (How helpful would this message be to support you in coping with a smoking urge or craving), 
4. quitting (How helpful would this message be to support you in quitting or reducing smoking).

The possible ratings (from lowest to highest) for content and design quality are:
- Very poor
- Poor
- Acceptable
- Good
- Very good

The possible ratings (from lowest to highest) for coping and quitting are:
- Not at all helpful
- Somewhat helpful
- Moderately helpful
- Very helpful
- Extremely helpful 
"""

    if is_balanced:
        prompt += """
IMPORTANT: Use the FULL RANGE of ratings. Do not avoid extreme ratings like "Very poor", "Very good", "Not at all helpful", or "Extremely helpful" - these ratings exist in the data and should be used when appropriate for the participant and message.
"""

    prompt += f"""
Here is the message provided to the participant:

"{data['input_message']}"

And here is the demographics for the participant, use these information to embed yourself as
a member of the group:

Participant metadata:
"""

    # Determine which metadata features to include
    features_to_include = data['metadata'].keys() if use_all_features else SELECTED_FEATURES
    
    for key in features_to_include:
        value = data['metadata'].get(key)
        if pd.notna(value):
            prompt += f"- {key}: {value}\n"

    # End of the prompt
    prompt += """
Return your response in the following JSON format:
{{
"response_id": "{response_id}",
"input_message": "{input_message}",
"image_description": "List the text shown on the image here.",
"predicted_content": "Choose one of the following options: \\"Very poor/Poor/Acceptable/Good/Very good\\"",
"predicted_design": "Choose one of the following options: \\"Very poor/Poor/Acceptable/Good/Very good\\"",
"predicted_coping": "Choose one of the following options: \\"Not at all helpful/Somewhat helpful/Moderately helpful/Very helpful/Extremely helpful\\"",
"predicted_quitting": "Choose one of the following options: \\"Not at all helpful/Somewhat helpful/Moderately helpful/Very helpful/Extremely helpful\\"",
"explanation": "Replace this with a very brief explanation (AT MOST 2 sentences!) for each predicted dimension. Your explanation should reflect your internal reasoning—consider what latent beliefs, inferred motivations, or psychological traits (e.g., readiness to quit, affective response, perceived relevance) might influence the participant's ratings."
}}
""".format(response_id=data['response_id'], input_message=data['input_message'].replace('"', '\\"'))

    return prompt 

def generate_enhanced_zero_shot_prompt(data: dict) -> str:
    """Generate the enhanced zero-shot prompt with detailed rubrics"""
    
    # Extract participant metadata
    # Using all metadata available in this version
    metadata_text = ""
    if 'metadata' in data and data['metadata']:
        for key, value in data['metadata'].items():
            if pd.notna(value):
                metadata_text += f"- {key}: {value}\n"
    
    # Escape the message for JSON
    escaped_message = data['input_message'].replace('"', '\\"')
    
    prompt = f"""
You are an **expert in smoking-cessation communication and intervention**.

Your first task is to **describe the image** provided in **one sentence**.

After that, **predict how *this participant* will rate** a smoking-cessation support message.  
**CRITICAL → Evaluate strictly from the participant’s perspective, not as a general expert.**

---

## RATING DIMENSIONS
1. **content** – words and meaning  
2. **design** – visual presentation  
3. **coping** – helpfulness for handling an urge or craving *in the moment*  
4. **quitting** – helpfulness for quitting or reducing smoking *long-term*  

### Allowed rating categories  
**Content / Design** → Very poor · Poor · Acceptable · Good · Very good  
**Coping / Quitting** → Not at all helpful · Somewhat helpful · Moderately helpful · Very helpful · Extremely helpful  

---

## PARTICIPANT-CENTRIC GUIDELINES  
Before rating, weigh these factors **in the participant's metadata**:  
1. **Psychological barriers** – worry, fear of feelings, emotional problems.  
2. **Social environment** – household/friend smokers, second-hand cues.  
3. **Quit history** – number & outcome of prior attempts; past use of advice given.  
4. **Motivation & support** – readiness and availability of help.  
5. **Nicotine dependence** – cigarettes/day, time to first cigarette.  
6. **Context realism** – can the suggested action work in their daily life and socioeconomic setting?  

---

## GLOBAL RATING RUBRIC  
• **Very poor** – generic, patronising, no new actionable idea; likely ignored / reactance.  
• **Poor** – some idea but unclear, controlling, or badly tailored.  
• **Acceptable** – clear & neutral but not notably novel or motivating.  
• **Good** – clear + novel/motivating *and* respects autonomy; supportive tone.  
• **Very good** – all of "Good" plus culturally/relevantly tailored, emotionally engaging, concrete, feasible action.  

---

## COPING-SPECIFIC RUBRIC  
| Level | Likely participant reaction | Message traits |
|-------|-----------------------------|----------------|
| **Not at all helpful** | "This won't help my cravings." | Only slogans / self-talk, no concrete skill, unrealistic. |
| **Somewhat helpful** | "Maybe okay but basic." | A single generic tactic (e.g., "count to 10"), limited tailoring. |
| **Moderately helpful** | "Could work sometimes." | One evidence-based tactic + brief rationale, minor feasibility gaps. |
| **Very helpful** | "I can do this when cravings hit." | Specific, context-aware tactic addressing barriers; autonomy-supportive. |
| **Extremely helpful** | "Gives me multiple tools I trust." | Combines behavioral + cognitive skills; boosts self-efficacy; usable anywhere. |

✔ **Coping checklist** → Novelty · Feasibility · Barrier fit · Self-efficacy  

---

## QUITTING-SPECIFIC RUBRIC (Enhanced)  
| Level | Likely participant reaction | Message traits |
|-------|-----------------------------|----------------|
| **Not at all helpful** | "This won't get me closer to quitting." | Pure affirmation or vague mindfulness; **no plan, resources, or timeline**; ignores nicotine dependence. |
| **Somewhat helpful** | "Nice idea but thin." | Single motivational step (e.g., write a note) **without** guidance on evidence-based aids (NRT, quitline, Rx) or social support. |
| **Moderately helpful** | "A decent starting point." | Introduces **one** evidence-based step (set quit date, call quitline, use NRT) + rationale, yet only lightly addresses dependence/barriers. |
| **Very helpful** | "This feels like a workable plan." | Blends **behavioral + pharmacological** advice, includes resource link / support prompt, tailors to dependence (cigs/day, TTFC) **and** quit history. |
| **Extremely helpful** | "Clear, comprehensive roadmap I believe in." | Provides a **multi-step, tailored strategy** (set date, remove triggers, select NRT/Rx, enlist support), anticipates barriers (stress, environment, low income), offers concrete resources (quitline number, free NRT program), and reinforces efficacy with success cues. |

✔ **Quitting checklist** → Evidence-based components · Tailoring to dependence & history · Resource inclusion · Socio-economic fit · Self-efficacy boost  

### Common pitfalls that **must lower** the quitting rating  
- Relies only on positive self-talk or mindfulness metaphors without next steps.  
- Ignores heavy dependence indicators (≥10 CPD, TTFC ≤30 min).  
- Suggests difficult or costly actions without acknowledging participant's income or support constraints.  

---

### ADJUSTMENT RULES  
• **Penalty** – Down-grade ≥1 level if message omits evidence-based aids, offers only generic motivation, or is unrealistic for the participant's context.  
• **Bonus** – Up-grade only when strategy is **fresh, specific, evidence-based, low-cost**, and actionable **today** for this participant.  

### EXPLANATION LIMIT  
For each dimension write **≤ 2 sentences** citing at least one checklist factor (e.g., "generic self-talk; no NRT advice for highly dependent smoker").

---

### INPUTS  

Here is the message provided to the participant:  

\\"{data['input_message']}\\"  

Participant metadata (embed yourself as a member of this group):  
{metadata_text}
---

### OUTPUT FORMAT  
Return **exactly** this JSON object:  

{{
  "response_id": "{data['response_id']}",
  "input_message": "{escaped_message}",
  "image_description": "List the text shown on the image here.",
  "predicted_content": "Very poor/Poor/Acceptable/Good/Very good",
  "predicted_design": "Very poor/Poor/Acceptable/Good/Very good",
  "predicted_coping": "Not at all helpful/Somewhat helpful/Moderately helpful/Very helpful/Extremely helpful",
  "predicted_quitting": "Not at all helpful/Somewhat helpful/Moderately helpful/Very helpful/Extremely helpful",
  "explanation": "≤ 2 sentences per dimension reflecting the participant's likely view (psychological barriers, social context, quit history, dependence, motivation)."
}}
"""
    return prompt 

def _create_natural_language_profile(metadata: dict) -> str:
    """Creates a natural language paragraph from participant metadata."""
    
    parts = []
    
    # Sentence 1: Demographics
    age = metadata.get('age_years')
    gender = metadata.get('gender_identity')
    if age and gender:
        parts.append(f"The participant is a {age}-year-old {gender}.")
    
    # Sentence 2: Smoking Behavior
    cigs_per_day = metadata.get('cigs_per_day', 'an unspecified number of')
    smoking_status = metadata.get('smoking_status', 'smokes')
    quit_attempts = metadata.get('quit_attempts_count', 'an unknown number of')
    
    behavior_parts = []
    if smoking_status:
        behavior_parts.append(f"who currently smokes '{smoking_status}'")
    if cigs_per_day:
        behavior_parts.append(f"and reports smoking around {cigs_per_day} cigarettes per day")
    if quit_attempts:
        behavior_parts.append(f"They have attempted to quit {quit_attempts} times in the past.")
        
    if behavior_parts:
        # A bit of grammar correction for sentence flow
        full_sentence = " ".join(behavior_parts)
        if full_sentence.startswith("who currently"):
            full_sentence = "Currently, they" + full_sentence[13:]
        parts.append(full_sentence.strip() + ".")
        
    # Sentence 3: Motivation and Support
    motivation = metadata.get('quit_motivation_level')
    support = metadata.get('social_support_to_quit')
    
    motivation_parts = []
    if motivation:
        motivation_parts.append(f"Their motivation to quit is '{motivation}'.")
    if support:
        motivation_parts.append(f"In terms of social support, they describe it as '{support}'.")
        
    if motivation_parts:
        parts.append(" ".join(motivation_parts))
        
    if not parts:
        return "No participant metadata available."
        
    return "\n".join(parts)


def generate_zero_shot_natural_lang_prob_prompt(data: dict) -> str:
    """
    Generates a zero-shot prompt with a natural language participant profile
    and probability output.
    """
    metadata = data.get('metadata', {})
    participant_profile = _create_natural_language_profile(metadata)

    prompt = f"""
You are an expert in smoking cessation communication. Your task is to evaluate a support message based on a specific participant's characteristics, which are described below in a natural language profile.

**INSTRUCTIONS:**
1.  Read the **Participant Profile** to understand the person receiving the message.
2.  Analyze the provided **Input Message**.
3.  For each of the four dimensions (content, design, coping, quitting), perform two steps:
    a.  **Predict Probabilities**: Estimate the probability for each possible rating. The probabilities for each dimension must sum to 1.0.
    b.  **Make a Final Prediction**: Choose the rating with the highest estimated probability.
    c.  **State Confidence**: Provide a confidence score from 0.0 to 1.0 for your final prediction.

**PARTICIPANT PROFILE:**
{participant_profile}

**INPUT MESSAGE:**
{data['input_message']}

**OUTPUT FORMAT (JSON ONLY):**
You must return a valid JSON object with the following structure. Provide the final prediction, the estimated probabilities, and your confidence for each of the four dimensions.

{{
  "predicted_content": "...",
  "predicted_content_probabilities": {{
    "Very poor": 0.0,
    "Poor": 0.0,
    "Acceptable": 0.0,
    "Good": 0.0,
    "Very good": 0.0
  }},
  "predicted_content_confidence": 0.0,
  "predicted_design": "...",
  "predicted_design_probabilities": {{
    "Very poor": 0.0,
    "Poor": 0.0,
    "Acceptable": 0.0,
    "Good": 0.0,
    "Very good": 0.0
  }},
  "predicted_design_confidence": 0.0,
  "predicted_coping": "...",
  "predicted_coping_probabilities": {{
    "Not at all helpful": 0.0,
    "Somewhat helpful": 0.0,
    "Moderately helpful": 0.0,
    "Very helpful": 0.0,
    "Extremely helpful": 0.0
  }},
  "predicted_coping_confidence": 0.0,
  "predicted_quitting": "...",
  "predicted_quitting_probabilities": {{
    "Not at all helpful": 0.0,
    "Somewhat helpful": 0.0,
    "Moderately helpful": 0.0,
    "Very helpful": 0.0,
    "Extremely helpful": 0.0
  }},
  "predicted_quitting_confidence": 0.0,
  "image_description": "A brief, one-sentence description of the image content if one is provided.",
  "explanation": "A brief explanation of your reasoning, considering both the message and the participant."
}}
"""
    return prompt


def generate_zero_shot_feature_select_prob_prompt(data: dict) -> str:
    """
    Generates a zero-shot prompt with feature selection and probability output.
    """
    metadata = data.get('metadata', {})
    
    # Feature selection
    selected_features = {
        'Age': metadata.get('age_years'),
        'Gender': metadata.get('gender_identity'),
        'Smoking Status': metadata.get('smoking_status'),
        'Cigarettes per Day': metadata.get('cigs_per_day'),
        'Quit Attempts in Past Year': metadata.get('quit_attempts_count'),
        'Motivation to Quit': metadata.get('quit_motivation_level'),
        'Social Support': metadata.get('social_support_to_quit')
    }
    
    # Filter out any features that are None or empty strings
    participant_metadata = "\n".join([f"- {key}: {value}" for key, value in selected_features.items() if value is not None and str(value).strip()])

    prompt = f"""
You are an expert in smoking cessation communication. Your task is to evaluate a support message based on a specific participant's characteristics.

**INSTRUCTIONS:**
1.  Analyze the provided **Input Message**.
2.  Review the **Participant Metadata** to understand the person receiving the message.
3.  For each of the four dimensions (content, design, coping, quitting), perform two steps:
    a.  **Predict Probabilities**: Estimate the probability for each possible rating. The probabilities for each dimension must sum to 1.0.
    b.  **Make a Final Prediction**: Choose the rating with the highest estimated probability.

**INPUT MESSAGE:**
{data['input_message']}

**PARTICIPANT METADATA:**
{participant_metadata}

**OUTPUT FORMAT (JSON ONLY):**
You must return a valid JSON object with the following structure. Provide both the final prediction and the estimated probabilities for each of the four dimensions.

{{
  "predicted_content": "...",
  "predicted_content_probabilities": {{
    "Very poor": 0.0,
    "Poor": 0.0,
    "Acceptable": 0.0,
    "Good": 0.0,
    "Very good": 0.0
  }},
  "predicted_design": "...",
  "predicted_design_probabilities": {{
    "Very poor": 0.0,
    "Poor": 0.0,
    "Acceptable": 0.0,
    "Good": 0.0,
    "Very good": 0.0
  }},
  "predicted_coping": "...",
  "predicted_coping_probabilities": {{
    "Not at all helpful": 0.0,
    "Somewhat helpful": 0.0,
    "Moderately helpful": 0.0,
    "Very helpful": 0.0,
    "Extremely helpful": 0.0
  }},
  "predicted_quitting": "...",
  "predicted_quitting_probabilities": {{
    "Not at all helpful": 0.0,
    "Somewhat helpful": 0.0,
    "Moderately helpful": 0.0,
    "Very helpful": 0.0,
    "Extremely helpful": 0.0
  }},
  "image_description": "A brief, one-sentence description of the image content if one is provided.",
  "explanation": "A brief explanation of your reasoning, considering both the message and the participant."
}}
"""
    return prompt 

def generate_digital_twin_prompt(data):
    """Generate prompts for digital twin method"""

    # Read the traing messages & ratings (first 7 messages) for each participant
    df = pd.read_excel("data/digitalTwin_msg.xlsx")
    
    # Extract participant metadata
    # Using all metadata available in this version
    metadata_text = ""
    if 'metadata' in data and data['metadata']:
        for key, value in data['metadata'].items():
            if pd.notna(value):
                metadata_text += f"- {key}: {value}\n"
    
    # Add training messages and ratings for each participant
    row = df[df["response_id"] == data['response_id']]
    
    for col in df.columns:
            if col.startswith("message_"):
                msg_text = str(row[col].iloc[0]).strip()

                # Map rating column names to short labels
                label_map = {
                    "content": "content",
                    "design": "design",
                    "coping": "coping",
                    "quitting": "quitting"
                }
                
                # Find rating columns (they follow the message column)
                ratings = []
                col_index = df.columns.get_loc(col)
                # Ratings usually are next 4 columns after each message
                for rcol in df.columns[col_index+1 : col_index+5]:
                    if rcol.startswith("how"):
                        # Pick a short label based on keyword in column name
                        for key, short_label in label_map.items():
                            if key in rcol.lower():
                                value = row[rcol].iloc[0]
                                ratings.append(f"{short_label}: {value}")
                                break

                ratings_text = "\n".join(ratings)

                metadata_text += f"Past message:\n{msg_text}\nRatings:\n{ratings_text}\n\n---\n"

      
    prompt = f"""
You are an AI assistant simulating this participant. Your task is to predict how the participant will rate a new smoking-cessation support message, using their Participant metadata (which includes their characteristics and past message ratings).
Base your prediction on how similar the new message is to the participant’s previously rated messages. Remain consistent with the participant’s prior ratings and stated characteristics, as if you are that person.
Be sure to carefully follow all provided Instructions for formatting your answer to the new message.
---
Instructions:
### RATING DIMENSIONS
1. **content** – How would you rate the content (that is, the words and meaning) of this message?  
2. **design** – How would you rate the design (that is, how the message looks) of this message? 
3. **coping** – How helpful would this message be to support you in coping with a smoking urge or craving?  
4. **quitting** – How helpful would this message be to support you in quitting or reducing smoking? 

### Allowed rating categories  
**content / design** → Very poor · Poor · Acceptable · Good · Very good  
**coping / quitting** → Not at all helpful · Somewhat helpful · Moderately helpful · Very helpful · Extremely helpful  



### INPUTS  

Here is the message provided to the participant to be rated:  

\\"{data['input_message']}\\"  

Participant metadata:  
{metadata_text}
---

### OUTPUT FORMAT  
Return **exactly** this JSON object:  

{{
  "response_id": "{data['response_id']}",
  "predicted_content": "Very poor/Poor/Acceptable/Good/Very good",
  "predicted_design": "Very poor/Poor/Acceptable/Good/Very good",
  "predicted_coping": "Not at all helpful/Somewhat helpful/Moderately helpful/Very helpful/Extremely helpful",
  "predicted_quitting": "Not at all helpful/Somewhat helpful/Moderately helpful/Very helpful/Extremely helpful",
  "explanation": "≤ 2 sentences per dimension reflecting the participant's likely view (psychological barriers, social context, quit history, dependence, motivation)."
}}
"""
    #print("=== Prompt Sent to GPT ===")
    #print(prompt)
    #print("==========================")
    return prompt 

def generate_digital_twin_select_prompt(data):
    """Generate prompts for digital twin method with selected features"""

    # Read the traing messages & ratings (first 7 messages) for each participant
    df = pd.read_excel("data/digitalTwin_msg.xlsx")
    
    # Extract participant metadata
    # Using all metadata available in this version
    metadata_text = ""
    if 'metadata' in data and data['metadata']:
        for key, value in data['metadata'].items():
            if key in SELECTED_FEATURES and pd.notna(value):
                metadata_text += f"- {key}: {value}\n"
    
    # Add training messages and ratings for each participant
    row = df[df["response_id"] == data['response_id']]
    
    for col in df.columns:
            if col.startswith("message_"):
                msg_text = str(row[col].iloc[0]).strip()

                # Map rating column names to short labels
                label_map = {
                    "content": "content",
                    "design": "design",
                    "coping": "coping",
                    "quitting": "quitting"
                }
                
                # Find rating columns (they follow the message column)
                ratings = []
                col_index = df.columns.get_loc(col)
                # Ratings usually are next 4 columns after each message
                for rcol in df.columns[col_index+1 : col_index+5]:
                    if rcol.startswith("how"):
                        # Pick a short label based on keyword in column name
                        for key, short_label in label_map.items():
                            if key in rcol.lower():
                                value = row[rcol].iloc[0]
                                ratings.append(f"{short_label}: {value}")
                                break

                ratings_text = "\n".join(ratings)

                metadata_text += f"Past message:\n{msg_text}\nRatings:\n{ratings_text}\n\n---\n"

      
    prompt = f"""
You are an AI assistant simulating this participant. Your task is to predict how the participant will rate a new smoking-cessation support message, using their Participant metadata (which includes their characteristics and past message ratings).
Base your prediction on how similar the new message is to the participant’s previously rated messages. Remain consistent with the participant’s prior ratings and stated characteristics, as if you are that person.
Be sure to carefully follow all provided Instructions for formatting your answer to the new message.
---
Instructions:
### RATING DIMENSIONS
1. **content** – How would you rate the content (that is, the words and meaning) of this message?  
2. **design** – How would you rate the design (that is, how the message looks) of this message? 
3. **coping** – How helpful would this message be to support you in coping with a smoking urge or craving?  
4. **quitting** – How helpful would this message be to support you in quitting or reducing smoking? 

### Allowed rating categories  
**content / design** → Very poor · Poor · Acceptable · Good · Very good  
**coping / quitting** → Not at all helpful · Somewhat helpful · Moderately helpful · Very helpful · Extremely helpful  



### INPUTS  

Here is the message provided to the participant to be rated:  

\\"{data['input_message']}\\"  

Participant metadata:  
{metadata_text}
---

### OUTPUT FORMAT  
Return **exactly** this JSON object:  

{{
  "response_id": "{data['response_id']}",
  "predicted_content": "Very poor/Poor/Acceptable/Good/Very good",
  "predicted_design": "Very poor/Poor/Acceptable/Good/Very good",
  "predicted_coping": "Not at all helpful/Somewhat helpful/Moderately helpful/Very helpful/Extremely helpful",
  "predicted_quitting": "Not at all helpful/Somewhat helpful/Moderately helpful/Very helpful/Extremely helpful",
  "explanation": "≤ 2 sentences per dimension reflecting the participant's likely view (psychological barriers, social context, quit history, dependence, motivation)."
}}
"""
    #print("=== Prompt Sent to GPT ===")
    #print(prompt)
    #print("==========================")
    return prompt 

def generate_digital_twin_feedback_prompt(data):
    """Generate prompts for digital twin method with feedback in prompts"""

    # Read the traing messages & ratings (first 7 messages) for each participant
    df = pd.read_excel("data/digitalTwin_msg.xlsx")
    feedback = pd.read_csv("data/Message testing data with participant characteristics_02.27.csv")
    
    # Extract participant metadata
    # Using all metadata available in this version
    metadata_text = ""
    if 'metadata' in data and data['metadata']:
        for key, value in data['metadata'].items():
            if pd.notna(value):
                metadata_text += f"- {key}: {value}\n"
    
    # Add training messages and ratings for each participant
    row = df[df["response_id"] == data['response_id']]
    
    for col in df.columns:
            if col.startswith("message_"):
                msg_text = str(row[col].iloc[0]).strip()

                # Map rating column names to short labels
                label_map = {
                    "content": "content",
                    "design": "design",
                    "coping": "coping",
                    "quitting": "quitting"
                }
                
                # Find rating columns (they follow the message column)
                ratings = []
                col_index = df.columns.get_loc(col)
                # Ratings usually are next 4 columns after each message
                for rcol in df.columns[col_index+1 : col_index+5]:
                    if rcol.startswith("how"):
                        # Pick a short label based on keyword in column name
                        for key, short_label in label_map.items():
                            if key in rcol.lower():
                                value = row[rcol].iloc[0]
                                ratings.append(f"{short_label}: {value}")
                                break

                ratings_text = "\n".join(ratings)

                 # --- Add feedback ---
                fb_row = feedback[
                    (feedback["response_id"] == data['response_id']) &
                    (feedback["message_num"] == col)   
                ]
        
                if not fb_row.empty and "additional_thoughts_quantative_resonses" in fb_row:
                    fb_text = str(fb_row["additional_thoughts_quantative_resonses"].iloc[0]).strip()
                else:
                    fb_text = None
        
                # Build section text
                metadata_text += f"Past message:\n{msg_text}\n"
                if ratings_text:
                    metadata_text += f"Ratings:\n{ratings_text}\n"
                if fb_text:
                    metadata_text += f"Feedback:\n{fb_text}\n"
                metadata_text += "\n---\n"
      
    prompt = f"""
You are an AI assistant simulating this participant. Your task is to predict how the participant will rate a new smoking-cessation support message. Use the Participant metadata, which includes their characteristics, past message ratings, and any feedback they have provided on those messages.

When forming your prediction:

Base your judgment primarily on how similar the new message is to the participant’s previously rated messages and the patterns in their past ratings.
Use the participant’s feedback as additional context to refine your understanding of their preferences (e.g., comments about what they liked or disliked).
Ensure that your prediction reflects both the quantitative ratings and the qualitative feedback, while keeping ratings as the main anchor.
Stay consistent with the participant’s overall profile, including their characteristics, past ratings, and feedback.
Be sure to carefully follow all provided Instructions for formatting your answer to the new message.

---
Instructions:
### RATING DIMENSIONS
1. **content** – How would you rate the content (that is, the words and meaning) of this message?  
2. **design** – How would you rate the design (that is, how the message looks) of this message? 
3. **coping** – How helpful would this message be to support you in coping with a smoking urge or craving?  
4. **quitting** – How helpful would this message be to support you in quitting or reducing smoking? 

### Allowed rating categories  
**content / design** → Very poor · Poor · Acceptable · Good · Very good  
**coping / quitting** → Not at all helpful · Somewhat helpful · Moderately helpful · Very helpful · Extremely helpful  



### INPUTS  

Here is the message provided to the participant to be rated:  

\\"{data['input_message']}\\"  

Participant metadata:  
{metadata_text}
---

### OUTPUT FORMAT  
Return **exactly** this JSON object:  

{{
  "response_id": "{data['response_id']}",
  "predicted_content": "Very poor/Poor/Acceptable/Good/Very good",
  "predicted_design": "Very poor/Poor/Acceptable/Good/Very good",
  "predicted_coping": "Not at all helpful/Somewhat helpful/Moderately helpful/Very helpful/Extremely helpful",
  "predicted_quitting": "Not at all helpful/Somewhat helpful/Moderately helpful/Very helpful/Extremely helpful",
  "explanation": "≤ 2 sentences per dimension reflecting the participant's likely view (psychological barriers, social context, quit history, dependence, motivation)."
}}
"""
    #print("=== Prompt Sent to GPT ===")
    #print(prompt)
    #print("==========================")
    return prompt 

def generate_digital_twin_cbtact_prompt(data):
    """Generate prompts for digital twin method with CBT/ACT labels for messages"""

    # Read the traing messages & ratings (first 7 messages) for each participant
    df = pd.read_excel("data/digitalTwin_msg.xlsx")
    feedback = pd.read_csv("data/Message testing data with participant characteristics_02.27.csv")
    
    # Extract participant metadata
    # Using all metadata available in this version
    metadata_text = ""
    msg_type_text = ""
    if 'metadata' in data and data['metadata']:
        for key, value in data['metadata'].items():
            if pd.notna(value):
                metadata_text += f"- {key}: {value}\n"

        # Check if "Image ID" exists in metadata
        image_id = data['metadata'].get("Image ID")
        if image_id and pd.notna(image_id):
            # Find matching row in feedback by photo_no
            match = feedback.loc[feedback['photo_no'] == image_id, 'l_category']
            if not match.empty:
                l_category_value = match.iloc[0]
                msg_type_text += f"\nMessage type: {l_category_value}\n"
    
    # Add training messages and ratings for each participant
    row = df[df["response_id"] == data['response_id']]
    
    for col in df.columns:
            if col.startswith("message_"):
                msg_text = str(row[col].iloc[0]).strip()

                # Map rating column names to short labels
                label_map = {
                    "content": "content",
                    "design": "design",
                    "coping": "coping",
                    "quitting": "quitting"
                }
                
                # Find rating columns (they follow the message column)
                ratings = []
                col_index = df.columns.get_loc(col)
                # Ratings usually are next 4 columns after each message
                for rcol in df.columns[col_index+1 : col_index+5]:
                    if rcol.startswith("how"):
                        # Pick a short label based on keyword in column name
                        for key, short_label in label_map.items():
                            if key in rcol.lower():
                                value = row[rcol].iloc[0]
                                ratings.append(f"{short_label}: {value}")
                                break

                ratings_text = "\n".join(ratings)

                 # --- Add message type ---
                fb_row = feedback[
                    (feedback["response_id"] == data['response_id']) &
                    (feedback["message_num"] == col)   
                ]
        
                if not fb_row.empty and "l_category" in fb_row:
                    fb_text = str(fb_row["l_category"].iloc[0]).strip()
                else:
                    fb_text = None
        
                # Build section text
                metadata_text += f"Past message:\n{msg_text}\n"
                if ratings_text:
                    metadata_text += f"Ratings:\n{ratings_text}\n"
                if fb_text:
                    metadata_text += f"Message type:\n{fb_text}\n"
                metadata_text += "\n---\n"
      
    prompt = f"""
You are an AI assistant simulating this participant. Your task is to predict how the participant will rate a new smoking-cessation support message, using their Participant metadata (which includes their characteristics and past message ratings).
Base your prediction on how similar the new message is to the participant’s previously rated messages. Remain consistent with the participant’s prior ratings and stated characteristics, as if you are that person.
When predicting coping and quitting, use an additional factor - message type.
Be sure to carefully follow all provided Instructions for formatting your answer to the new message.
---
Instructions:
### RATING DIMENSIONS
1. **content** – How would you rate the content (that is, the words and meaning) of this message?  
2. **design** – How would you rate the design (that is, how the message looks) of this message? 
3. **coping** – How helpful would this message be to support you in coping with a smoking urge or craving?  
4. **quitting** – How helpful would this message be to support you in quitting or reducing smoking? 

### Allowed rating categories  
**content / design** → Very poor · Poor · Acceptable · Good · Very good  
**coping / quitting** → Not at all helpful · Somewhat helpful · Moderately helpful · Very helpful · Extremely helpful  



### INPUTS  

Here is the message provided to the participant to be rated:  

\\"{data['input_message']}\\"  

{msg_type_text}

Participant metadata:  
{metadata_text}
---

### OUTPUT FORMAT  
Return **exactly** this JSON object:  

{{
  "response_id": "{data['response_id']}",
  "predicted_content": "Very poor/Poor/Acceptable/Good/Very good",
  "predicted_design": "Very poor/Poor/Acceptable/Good/Very good",
  "predicted_coping": "Not at all helpful/Somewhat helpful/Moderately helpful/Very helpful/Extremely helpful",
  "predicted_quitting": "Not at all helpful/Somewhat helpful/Moderately helpful/Very helpful/Extremely helpful",
  "explanation": "≤ 2 sentences per dimension reflecting the participant's likely view (psychological barriers, social context, quit history, dependence, motivation)."
}}
"""
    #print("=== Prompt Sent to GPT ===")
    #print(prompt)
    #print("==========================")
    return prompt
