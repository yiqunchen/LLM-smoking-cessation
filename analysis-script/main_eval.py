import json
import csv
import asyncio
import tqdm.asyncio
from datetime import datetime
from openai import AsyncOpenAI
import random
import pickle
from tqdm.asyncio import tqdm
import asyncio
from typing import Callable, Dict
import os
import time
import signal
import sys
import argparse
import base64
import pandas as pd

# Replace with your own API key
api_key = os.environ['CHEN_OPENAI_API_KEY']

# Default file paths (will be overridden by command line arguments)
DEFAULT_CHECKPOINT_FILE = "checkpoint_results_{model}_{mode}.json"
DEFAULT_OUTPUT_FILE = "evaluation_results_{model}_{mode}.json"
DEFAULT_CHECKPOINT_INTERVAL = 100  # Save every 100 completed items

# Global variable to track if we're shutting down
shutting_down = False

# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    global shutting_down
    print("\nReceived interrupt signal. Finishing current tasks and saving progress...")
    shutting_down = True
    # Don't exit immediately, let the main loop handle the shutdown

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def encode_image(image_path):
    """Encodes an image to a base64 string."""
    if not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

async def get_response_async(client, prompt_messages, model, semaphore, temperature):
    """Asynchronously get response from OpenAI API with semaphore control"""
    async with semaphore:
        completion = await client.chat.completions.create(
            model=model,
            messages=prompt_messages,
            response_format={"type": "json_object"},
            temperature=temperature,
        )
        return completion.choices[0].message.content


async def evaluate_questions_parallel(
    question_data: Dict[str, dict],
    client,
    model: str,
    mode: str,
    temperature: float,
    max_concurrent: int = 5,
    checkpoint_file: str = DEFAULT_CHECKPOINT_FILE,
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL
) -> Dict[str, dict]:
    """
    Evaluate a set of smoking cessation message ratings using an LLM asynchronously with tqdm progress bar.
    Includes checkpointing to save progress periodically.

    Args:
        question_data (dict): Dict where keys are question IDs and values contain:
            - input_message, metadata, ratings (ground truth)
        client: OpenAI client
        model (str): Name of the model (e.g., "gpt-4o", "gpt-4-turbo")
        mode (str): 'text-only' or 'vision'
        max_concurrent (int): Max concurrent API calls allowed
        checkpoint_file (str): Path to save intermediate results
        checkpoint_interval (int): How often to save results (every N completed items)

    Returns:
        dict: Dictionary keyed by question ID, containing both model predictions and ground truth
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    results = {}
    completed_count = 0
    global shutting_down

    # Load previous results if they exist
    if os.path.exists(checkpoint_file):
        print(f"Loading checkpoint from {checkpoint_file}")
        with open(checkpoint_file, "r") as f:
            results = json.load(f)
        print(f"Loaded {len(results)} previously completed items")
    
    # Filter out questions that have already been processed
    pending_questions = {k: v for k, v in question_data.items() if k not in results}
    print(f"Processing {len(pending_questions)} remaining items out of {len(question_data)} total")

    if not pending_questions:
        print("All items already processed!")
        return results

    async def query_llm(qid: str, data: dict) -> Dict[str, dict]:
        nonlocal completed_count
        
        if shutting_down:
            return {}  # Return empty if we're shutting down
            
        text_prompt = f"""
        You are an AI assistant. Your first task is to describe the image provided in a single sentence.
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

        Here is the message provided to the participant:

        "{data['input_message']}"

        And here is the demographics for the participant, use these information to embed yourself as
        a member of the group:

        Participant metadata:
        """
        
        # Add all metadata fields to the prompt
        for key, value in data['metadata'].items():
            if pd.notna(value):  # Only include non-null values
                text_prompt += f"- {key}: {value}\n"
        
        text_prompt += f"""
        Return your response in the following JSON format:
        {{
        "response_id": "{data['response_id']}",
        "input_message": "{data['input_message']}",
        "image_description": "List the text shown on the image here.",
        "predicted_content": Choose one of the following options: "Very poor/Poor/Acceptable/Good/Very good",
        "predicted_design": Choose one of the following options: "Very poor/Poor/Acceptable/Good/Very good",
        "predicted_coping": Choose one of the following options: "Not at all helpful/Somewhat helpful/Moderately helpful/Very helpful/Extremely helpful",
        "predicted_quitting": Choose one of the following options: "Not at all helpful/Somewhat helpful/Moderately helpful/Very helpful/Extremely helpful",
        "explanation": Replace this with a very brief explanation (AT MOST 2 sentences!) for each predicted dimension. 
        Your explanation should reflect your internal reasoning—consider what latent beliefs, inferred motivations, or psychological traits 
        (e.g., readiness to quit, affective response, perceived relevance) might influence the participant's ratings.
        }}
        """
        
        prompt_messages = [{"role": "user", "content": []}]
        prompt_messages[0]["content"].append({"type": "text", "text": text_prompt})

        if mode == 'vision':
            if data.get('image_path') and os.path.exists(data['image_path']):
                base64_image = encode_image(data['image_path'])
                if base64_image:
                    # Add a message to the text prompt indicating an image is included
                    text_prompt += "\nAn image is included in this message for your review."
                    prompt_messages[0]["content"].append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    })
            else:
                # Handle case where image is missing for a vision task
                print(f"Warning: Image path not found for qid {qid}, running as text-only.")

        # Finalize the user prompt content
        prompt_messages[0]["content"][0]['text'] = text_prompt

        try:
            response_json_str = await get_response_async(client, prompt_messages, model, semaphore, temperature)
            response = json.loads(response_json_str)
        except Exception as e:
            response = {
                "response_id": data['response_id'],
                "input_message": data['input_message'],
                "ground_truth_content": data['ratings'].get('content', "ERROR"),
                "ground_truth_design": data['ratings'].get('design', "ERROR"),
                "ground_truth_coping": data['ratings'].get('coping', "ERROR"),
                "ground_truth_quitting": data['ratings'].get('quitting', "ERROR"),
                "predicted_content": "ERROR",
                "predicted_design": "ERROR",
                "predicted_coping": "ERROR",
                "predicted_quitting": "ERROR",
                "explanation": "ERROR",
                "error": str(e)
            }

        result = {
            qid: {
                "response_id": data['response_id'],
                "input_message": data['input_message'],
                "metadata": data['metadata'],
                "ground_truth_content": data['ratings'].get('content', ""),
                "ground_truth_design": data['ratings'].get('design', ""),
                "ground_truth_coping": data['ratings'].get('coping', ""),
                "ground_truth_quitting": data['ratings'].get('quitting', ""),
                "predicted_content": response.get("predicted_content", ""),
                "predicted_design": response.get("predicted_design", ""),
                "predicted_coping": response.get("predicted_coping", ""),
                "predicted_quitting": response.get("predicted_quitting", ""),
                "image_description": response.get("image_description", ""),
                "explanation": response.get("explanation", "")
            }
        }
        
        return result

    # Create a list to store tasks
    tasks = []
    for qid, item in pending_questions.items():
        # Create task and add callback
        task = asyncio.create_task(query_llm(qid, item))
        tasks.append(task)

    # Track progress
    pbar = tqdm(total=len(tasks), desc=f"Processing {len(tasks)} items")
    
    # Process tasks as they complete
    for task in asyncio.as_completed(tasks):
        if shutting_down:
            # Cancel pending tasks when shutdown is requested
            for t in tasks:
                if not t.done():
                    t.cancel()
            break
            
        result = await task
        if result:  # Only update if we got a non-empty result
            results.update(result)
            completed_count += 1
            pbar.update(1)
            
            # Save checkpoint periodically
            if completed_count % checkpoint_interval == 0:
                print(f"\nSaving checkpoint after {completed_count} completed items...")
                with open(checkpoint_file, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"Checkpoint saved to {checkpoint_file}")
                
            # Check if we need to shut down
            if shutting_down:
                print("Shutting down after current batch...")
                break

    pbar.close()
    
    # Final save
    print(f"Saving final results with {len(results)} items...")
    with open(checkpoint_file, "w") as f:
        json.dump(results, f, indent=2)
    
    return results

async def main():
    parser = argparse.ArgumentParser(description="Evaluate smoking cessation messages using an LLM.")
    parser.add_argument('--mode', type=str, choices=['text-only', 'vision'], required=True, help="Evaluation mode: 'text-only' or 'vision'")
    parser.add_argument('--model', type=str, default="gpt-4o-mini", help="Name of the OpenAI model to use.")
    parser.add_argument('--checkpoint-file', type=str, default=DEFAULT_CHECKPOINT_FILE, help="Path template for checkpoint file (use {model} and {mode} placeholders).")
    parser.add_argument('--output-file', type=str, default=DEFAULT_OUTPUT_FILE, help="Path template for final output file (use {model} and {mode} placeholders).")
    parser.add_argument('--max-concurrent', type=int, default=10, help="Maximum concurrent API calls.")
    parser.add_argument('--checkpoint-interval', type=int, default=DEFAULT_CHECKPOINT_INTERVAL, help="How often to save checkpoint (every N completed items).")
    parser.add_argument('--temperature', type=float, default=None, help="Set the model temperature. Overrides default logic (0.2, or 1.0 for 'o3-' models).")
    args = parser.parse_args()

    # Format file paths with model and mode
    checkpoint_file = args.checkpoint_file.format(model=args.model, mode=args.mode)
    final_output_file = args.output_file.format(model=args.model, mode=args.mode)

    # Determine temperature based on model name or user override
    temperature = args.temperature
    if temperature is None:  # If user did not provide a temperature
        if 'o3-' in args.model:
            temperature = 1.0
            print(f"Model name contains 'o3-', setting temperature to 1.0")
        else:
            temperature = 0.2
            print(f"Using default temperature: 0.2")
    else:
        print(f"User override: using temperature {temperature}")

    # Load data
    with open("data/processed_llm_data.json", "r") as f:
        question_data = json.load(f)

    random.seed(42)
    question_data = {f"{i}": data for i, data in enumerate(question_data)}
    
    # For testing with a sample, these lines are now commented out to run on the full dataset
    # sampled_keys = random.sample(list(question_data.keys()), 5)
    # question_data = {k: question_data[k] for k in sampled_keys}

    client = AsyncOpenAI(api_key=api_key)
    
    try:
        # Run the evaluation
        results = await evaluate_questions_parallel(
            question_data, 
            client, 
            model=args.model,
            mode=args.mode,
            temperature=temperature,
            max_concurrent=args.max_concurrent,
            checkpoint_file=checkpoint_file,
            checkpoint_interval=args.checkpoint_interval
        )
        
        # Save to final file
        with open(final_output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Final results saved to {final_output_file}")
            
    except Exception as e:
        print(f"ERROR: {str(e)}")
        # Even on exception, we'll have our checkpoint
    finally:
        print("Process completed or interrupted. Progress has been saved.")

# Entry point
if __name__ == "__main__":
    asyncio.run(main())