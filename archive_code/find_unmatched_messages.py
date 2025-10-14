import pandas as pd
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

def find_unmatched_messages():
    """
    Finds messages in the testing data that cannot be matched
    to an image in the message summary file.
    """
    try:
        message_summary_df = pd.read_excel('data/R01 Message Summary for message testing paper.xlsx')
        testing_data_df = pd.read_excel('data/Messaging_Testing_Data.xlsx')
    except FileNotFoundError as e:
        print(f"Error loading Excel files: {e}")
        return

    # Create a set of match keys from the summary file for efficient lookup
    summary_match_keys = set(message_summary_df['Message'].apply(get_first_sentence).dropna())

    # Find messages in the testing data that don't have a match
    unmatched_messages = set()
    message_cols = [f"message_{i}_of_10" for i in range(1, 11)]
    
    # Manually add the irregular column name for message 4
    if "how_would_you_rate_the_content_that_is_the_words_and_meaning_of_the_this_message_72" in testing_data_df.columns:
         # This seems to be a typo in the original file, check if other message columns are similar
         pass # No, the mapping from the user code is better

    # Using the manually defined message columns is more robust
    message_block_cols = [
        "message_1_of_10", "message_2_of_10", "message_3_of_10",
        "message_4_of_10", "message_5_of_10", "message_6_of_10",
        "message_7_of_10", "message_8_of_10", "message_9_of_10",
        "message_10_of_10"
    ]


    for _, row in testing_data_df.iterrows():
        for col in message_block_cols:
            msg_text = row.get(col)
            if pd.notna(msg_text):
                match_key = get_first_sentence(msg_text)
                if match_key not in summary_match_keys:
                    unmatched_messages.add(msg_text.strip())

    # Print the results
    if unmatched_messages:
        print(f"Found {len(unmatched_messages)} unique messages that could not be matched to an image:")
        for msg in sorted(list(unmatched_messages)):
            # Use repr() to make hidden characters like newlines visible
            print(f"- {repr(msg)}")
    else:
        print("All messages in the testing data were successfully matched to an image.")

if __name__ == '__main__':
    find_unmatched_messages() 