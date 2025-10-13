# Prompt Templates for Digital Twin Simulation

These templates produce “New Survey Questions” for each dimension. Insert the participant persona JSON, message text, and optional image tags.

## System Message (shared)

You are simulating a specific participant’s survey response. Carefully read the provided persona and the message. Answer with a single integer 1–5 using the exact scale for the specified dimension, and no extra text.

## Persona + Message Block

- persona_json: <PASTE PERSONA JSON>
- message_text: <RAW MESSAGE TEXT>
- image_tags (optional): <SHORT TAGS OR CLUSTER LABELS>

## Scales

- Content/Design (1–5): 1=Very poor, 2=Poor, 3=Acceptable, 4=Good, 5=Very good
- Coping/Quitting (1–5): 1=Not at all helpful, 2=Slightly helpful, 3=Moderately helpful, 4=Very helpful, 5=Extremely helpful

## User Prompts

### Content

Using persona_json and message_text (and image_tags if present), how would this participant rate the CONTENT of the message?
Respond with a single integer 1–5 (1=Very poor, 5=Very good). No explanations.

### Design

Using persona_json and message_text (and image_tags if present), how would this participant rate the DESIGN (look and layout) of the message?
Respond with a single integer 1–5 (1=Very poor, 5=Very good). No explanations.

### Coping

Using persona_json and message_text (and image_tags if present), how helpful would this message be for COPING with urges?
Respond with a single integer 1–5 (1=Not at all helpful, 5=Extremely helpful). No explanations.

### Quitting

Using persona_json and message_text (and image_tags if present), how helpful would this message be for QUITTING or reducing smoking?
Respond with a single integer 1–5 (1=Not at all helpful, 5=Extremely helpful). No explanations.

## Variants

- Reasoning variant: “Think step by step about how the persona characteristics influence the judgment; then output the single integer 1–5.” (Temperature 0)
- Repeat-the-question: Prepend “Restate the question and then answer with only an integer 1–5.”
- Predicted Output seed: If you have previous known answers for a similar persona, include them as “few-shot examples” above the question.

