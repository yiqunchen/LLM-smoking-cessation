"""
Adaptive Prompt Template System
This module handles system prompt optimization while keeping data injection separate
"""

SYSTEM_PROMPT_TEMPLATE = """
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

Focus on predicting from the participant's perspective, considering their demographics and personal context.
"""

DATA_INJECTION_TEMPLATE = """
Here is the message provided to the participant:

"{input_message}"

And here is the demographics for the participant, use these information to embed yourself as
a member of the group:

Participant metadata:
{participant_metadata}

Return your response in the following JSON format:
{{
"response_id": "{response_id}",
"input_message": "{input_message_escaped}",
"image_description": "List the text shown on the image here.",
"predicted_content": "Choose one of the following options: \\"Very poor/Poor/Acceptable/Good/Very good\\"",
"predicted_design": "Choose one of the following options: \\"Very poor/Poor/Acceptable/Good/Very good\\"",
"predicted_coping": "Choose one of the following options: \\"Not at all helpful/Somewhat helpful/Moderately helpful/Very helpful/Extremely helpful\\"",
"predicted_quitting": "Choose one of the following options: \\"Not at all helpful/Somewhat helpful/Moderately helpful/Very helpful/Extremely helpful\\"",
"explanation": "Replace this with a very brief explanation (AT MOST 2 sentences!) for each predicted dimension."
}}
"""

class AdaptivePromptTemplate:
    """Manages system prompt template optimization separately from data injection"""
    
    def __init__(self, initial_system_prompt: str = None):
        self.system_prompt = initial_system_prompt or SYSTEM_PROMPT_TEMPLATE
        self.data_template = DATA_INJECTION_TEMPLATE
        
    def generate_full_prompt(self, data: dict, selected_features: list) -> str:
        """Combine system prompt with actual data"""
        # Build participant metadata
        metadata_text = ""
        for key in selected_features:
            value = data['metadata'].get(key)
            if value is not None and str(value).strip():
                import pandas as pd
                if pd.notna(value):
                    metadata_text += f"- {key}: {value}\n"
        
        # Fill in the data template
        data_section = self.data_template.format(
            input_message=data['input_message'],
            participant_metadata=metadata_text,
            response_id=data['response_id'],
            input_message_escaped=data['input_message'].replace('"', '\\"')
        )
        
        # Combine system prompt with data
        return self.system_prompt + "\n" + data_section
    
    def update_system_prompt(self, new_system_prompt: str):
        """Update only the system prompt portion"""
        self.system_prompt = new_system_prompt
        
    def get_system_prompt(self) -> str:
        """Get the current system prompt without data"""
        return self.system_prompt