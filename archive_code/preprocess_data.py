import pandas as pd
import json
import os
import re

def get_first_sentence(text):
    """Extracts the first sentence from a block of text."""
    if pd.isna(text):
        return None
    # Strip leading/trailing whitespace and newlines
    cleaned_text = str(text).strip()
    # Find the first full sentence. This looks for text ending in ., !, or ?
    match = re.search(r'([^.!?]*[.!?])', cleaned_text)
    if match:
        return match.group(1).strip()
    # If no sentence-ending punctuation is found, return the whole cleaned string
    return cleaned_text

def preprocess_data(output_path='data/processed_llm_data.json'):
    """
    Processes two XLSX files to generate a JSON file for model training.
    """
    # Load the datasets
    try:
        message_summary_df = pd.read_excel('data/R01 Message Summary for message testing paper.xlsx')
        testing_data_df = pd.read_excel('data/Messaging_Testing_Data.xlsx')
    except FileNotFoundError as e:
        print(f"Error loading Excel files: {e}")
        return

    # 1. Define explicit column mappings
    meta_cols = {
        "age_years": "how_old_are_you_years",
        "gender_identity": "how_do_you_describe_yourself",
        "race_ethnicity": "what_is_your_race_select_all_that_apply_selected_choice",
        "is_hispanic_latino": "are_you_hispanic_latino_or_latina_or_of_spanish_origin",
        "quit_intention": "what_best_describes_your_intentions_regarding_quitting_smoking_would_you_say_you",
        "sexual_orientation_identity": "do_you_think_of_yourself_as_please_check_all_that_apply_selected_choice",
        "education_level": "what_is_your_highest_level_of_education",
        "household_income": "what_is_your_total_annual_household_income",
        "days_smoked_past_30d": "during_the_past_30_days_how_many_days_did_you_smoke_at_least_one_cigarette_days",
        "cigs_per_day": "when_you_smoke_on_average_how_many_cigarettes_do_you_smoke_per_day_cigarettes_per_days",
        "time_to_first_cig": "how_soon_after_you_wake_up_in_the_morning_do_you_usually_smoke_your_first_cigarette",
        "household_smokers": "does_anyone_who_lives_with_you_smoke_tobacco",
        "friends_smoke_level": "how_many_of_your_friends_smoke_tobacco_would_you_say",
        "quit_attempt_past_year": "during_the_past_12_months_have_you_stopped_smoking_tobacco_for_one_day_or_longer_because_you_were_trying_to_quit",
        "quit_attempts_count": "how_many_times_have_you_tried_to_quit_in_the_past_12_months",
        "pain_blocks_valued_life": "below_you_will_find_a_list_of_statements_please_rate_how_true_each_statement_is_for_you_by_selecting_a_response_use_the_scale_below_to_make_your_choice_my_painful_experiences_and_memories_make_it_difficult_for_me_to_live_a_life_that_i_would_value",
        "fear_of_feelings": "below_you_will_find_a_list_of_statements_please_rate_how_true_each_statement_is_for_you_by_selecting_a_response_use_the_scale_below_to_make_your_choice_i_m_afraid_of_my_feelings",
        "worry_about_control": "below_you_will_find_a_list_of_statements_please_rate_how_true_each_statement_is_for_you_by_selecting_a_response_use_the_scale_below_to_make_your_choice_i_worry_about_not_being_able_to_control_my_worries_and_feelings",
        "memories_block_fulfillment": "below_you_will_find_a_list_of_statements_please_rate_how_true_each_statement_is_for_you_by_selecting_a_response_use_the_scale_below_to_make_your_choice_my_painful_memories_prevent_me_from_having_a_fulfilling_life",
        "emotions_cause_problems": "below_you_will_find_a_list_of_statements_please_rate_how_true_each_statement_is_for_you_by_selecting_a_response_use_the_scale_below_to_make_your_choice_emotions_cause_problems_in_my_life",
        "others_handle_life_better": "below_you_will_find_a_list_of_statements_please_rate_how_true_each_statement_is_for_you_by_selecting_a_response_use_the_scale_below_to_make_your_choice_it_seems_like_most_people_are_handling_their_lives_better_than_i_am",
        "worry_blocks_success": "below_you_will_find_a_list_of_statements_please_rate_how_true_each_statement_is_for_you_by_selecting_a_response_use_the_scale_below_to_make_your_choice_worries_get_in_the_way_of_my_success",
        "smoking_status": "do_you_now_smoke_cigarettes_every_day_some_days_or_not_at_all",
        "quit_motivation_level": "how_motivated_are_you_to_quit_smoking",
        "social_support_to_quit": "if_you_tried_to_quit_smoking_how_supportive_would_your_friends_and_family_be"
    }
    
    message_blocks = [
        {"message": "message_1_of_10", "content": "how_would_you_rate_the_content_that_is_the_words_and_meaning_of_this_message_55", "design": "how_would_you_rate_the_design_that_is_how_the_message_looks_of_this_message_56", "coping": "how_helpful_would_this_message_be_to_support_you_in_coping_with_a_smoking_urge_or_craving_57", "quitting": "how_helpful_would_this_message_be_to_support_you_in_quitting_or_reducing_smoking_58"},
        {"message": "message_2_of_10", "content": "how_would_you_rate_the_content_that_is_the_words_and_meaning_of_this_message_61", "design": "how_would_you_rate_the_design_that_is_how_the_message_looks_of_this_message_62", "coping": "how_helpful_would_this_message_be_to_support_you_in_coping_with_a_smoking_urge_or_craving_63", "quitting": "how_helpful_would_this_message_be_to_support_you_in_quitting_or_reducing_smoking_64"},
        {"message": "message_3_of_10", "content": "how_would_you_rate_the_content_that_is_the_words_and_meaning_of_this_message_67", "design": "how_would_you_rate_the_design_that_is_how_the_message_looks_of_this_message_68", "coping": "how_helpful_would_this_message_be_to_support_you_in_coping_with_a_smoking_urge_or_craving_69", "quitting": "how_helpful_would_this_message_be_to_support_you_in_quitting_or_reducing_smoking_70"},
        {"message": "message_4_of_10", "content": "how_would_you_rate_the_content_that_is_the_words_and_meaning_of_the_this_message_72", "design": "how_would_you_rate_the_design_that_is_how_the_message_looks_of_this_message_73", "coping": "how_helpful_would_this_message_be_to_support_you_in_coping_with_a_smoking_urge_or_craving_74", "quitting": "how_helpful_would_this_message_be_to_support_you_in_quitting_or_reducing_smoking_75"},
        {"message": "message_5_of_10", "content": "how_would_you_rate_the_content_that_is_the_words_and_meaning_of_this_message_77", "design": "how_would_you_rate_the_design_that_is_how_the_message_looks_of_this_message_78", "coping": "how_helpful_would_this_message_be_to_support_you_in_coping_with_a_smoking_urge_or_craving_79", "quitting": "how_helpful_would_this_message_be_to_support_you_in_quitting_or_reducing_smoking_80"},
        {"message": "message_6_of_10", "content": "how_would_you_rate_the_content_that_is_the_words_and_meaning_of_this_message_82", "design": "how_would_you_rate_the_design_that_is_how_the_message_looks_of_this_message_83", "coping": "how_helpful_would_this_message_be_to_support_you_in_coping_with_a_smoking_urge_or_craving_84", "quitting": "how_helpful_would_this_message_be_to_support_you_in_quitting_or_reducing_smoking_85"},
        {"message": "message_7_of_10", "content": "how_would_you_rate_the_content_that_is_the_words_and_meaning_of_this_message_87", "design": "how_would_you_rate_the_design_that_is_how_the_message_looks_of_this_message_88", "coping": "how_helpful_would_this_message_be_to_support_you_in_coping_with_a_smoking_urge_or_craving_89", "quitting": "how_helpful_would_this_message_be_to_support_you_in_quitting_or_reducing_smoking_90"},
        {"message": "message_8_of_10", "content": "how_would_you_rate_the_content_that_is_the_words_and_meaning_of_this_message_92", "design": "how_would_you_rate_the_design_that_is_how_the_message_looks_of_this_message_93", "coping": "how_helpful_would_this_message_be_to_support_you_in_coping_with_a_smoking_urge_or_craving_94", "quitting": "how_helpful_would_this_message_be_to_support_you_in_quitting_or_reducing_smoking_95"},
        {"message": "message_9_of_10", "content": "how_would_you_rate_the_content_that_is_the_words_and_meaning_of_this_message_97", "design": "how_would_you_rate_the_design_that_is_how_the_message_looks_of_this_message_98", "coping": "how_helpful_would_this_message_be_to_support_you_in_coping_with_a_smoking_urge_or_craving_99", "quitting": "how_helpful_would_this_message_be_to_support_you_in_quitting_or_reducing_smoking_100"},
        {"message": "message_10_of_10", "content": "how_would_you_rate_the_content_that_is_the_words_and_meaning_of_this_message_102", "design": "how_would_you_rate_the_design_that_is_how_the_message_looks_of_this_message_103", "coping": "how_helpful_would_this_message_be_to_support_you_in_coping_with_a_smoking_urge_or_craving_104", "quitting": "how_helpful_would_this_message_be_to_support_you_in_quitting_or_reducing_smoking_105"},
    ]

    # 2. Restructure the data from wide to long format
    structured_data = []
    for _, row in testing_data_df.iterrows():
        # Extract metadata for the participant
        metadata = {simple_name: row.get(original_name) for simple_name, original_name in meta_cols.items()}
        
        # Loop through each message block
        for block in message_blocks:
            msg_text = row.get(block["message"])
            if pd.notnull(msg_text):
                structured_data.append({
                    "response_id": row.get("response_id"),
                    "input_message": msg_text.strip(),
                    "ratings": {
                        "content": row.get(block["content"]),
                        "design": row.get(block["design"]),
                        "coping": row.get(block["coping"]),
                        "quitting": row.get(block["quitting"])
                    },
                    "metadata": metadata
                })

    long_df = pd.DataFrame(structured_data)

    # 3. Apply Hardcoded Matches FIRST
    hardcode_map = {
        'Instead of lighting up a cigarette, light a scented candle or burn some incense.': 'D47',
        'No need to get rid of these feelings or control them with a smoke. They are not permanent.': 'A45',
        'You don’t need to get rid of or change your thoughts about smoking—just let them pass by like a cloud in the sky.': 'A47'
    }
    
    # Temporarily create a column for the match key in the long_df
    long_df['match_key_temp'] = long_df['input_message'].str.strip()
    
    for msg, img_id in hardcode_map.items():
        long_df.loc[long_df['match_key_temp'] == msg.strip(), 'Image ID'] = img_id

    # Drop the temporary key
    long_df = long_df.drop('match_key_temp', axis=1)

    # 4. Merge with message summary to get remaining Image IDs
    message_summary_df.rename(columns={'Photo No.': 'Image ID'}, inplace=True)
    long_df['match_key'] = long_df['input_message'].apply(get_first_sentence)
    message_summary_df['match_key'] = message_summary_df['Message'].apply(get_first_sentence)
    
    # Use a left merge to preserve the hardcoded IDs and add others
    merged_df = pd.merge(long_df, message_summary_df[['Image ID', 'match_key']], on='match_key', how='left', suffixes=('_hardcoded', ''))
    
    # Coalesce the Image ID columns - prioritize hardcoded, then fill with summary matches
    merged_df['Image ID'] = merged_df['Image ID_hardcoded'].fillna(merged_df['Image ID'])
    merged_df = merged_df.drop(['Image ID_hardcoded', 'Image ID_x', 'Image ID_y'], axis=1, errors='ignore')


    # 5. Finalize JSON creation
    image_dir = 'data/downloaded_smoke_images'
    data_for_json = []

    for _, row in merged_df.iterrows():
        # Skip rows with no message or no ratings
        if pd.isna(row['input_message']) or pd.isna(row['Image ID']):
            continue

        # Add image path
        row_dict = row.to_dict()
        row_dict['image_path'] = next((os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.startswith(str(row_dict['Image ID']))), None)
        
        # Ensure all metadata is properly handled
        final_metadata = {k: v for k, v in row['metadata'].items() if pd.notna(v)}
        if 'Image ID' in row and pd.notna(row['Image ID']):
            final_metadata['Image ID'] = row['Image ID']

        data_for_json.append({
            'response_id': row_dict['response_id'],
            'input_message': row_dict['input_message'],
            'image_path': row_dict['image_path'],
            'metadata': final_metadata,
            'ratings': row_dict['ratings']
        })
        
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(data_for_json, f, indent=4)

    print(f"Successfully generated JSON file at {output_path}")
    print(f"Total items processed: {len(data_for_json)}")


if __name__ == '__main__':
    preprocess_data() 