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
from prompt_config import (
    generate_zero_shot_prompt, 
    generate_zero_shot_feature_select_prompt, 
    generate_zero_shot_feature_select_balanced_prompt,
    generate_few_shot_prompt,
    generate_few_shot_feature_select_prompt,
    generate_few_shot_feature_select_balanced_prompt,
    generate_enhanced_zero_shot_prompt,
    generate_zero_shot_feature_select_prob_prompt,
    generate_zero_shot_natural_lang_prob_prompt,
    generate_digital_twin_prompt,
    generate_digital_twin_select_prompt,
    generate_digital_twin_feedback_prompt,
    generate_digital_twin_cbtact_prompt,
    prepare_few_shot_examples
)

# Replace with your own API key
api_key = os.environ['OPENAI_API_KEY']

# Default file paths (will be overridden by command line arguments)
DEFAULT_CHECKPOINT_FILE = "checkpoint_results_{model}_{mode}_{prompt_config}.json"
DEFAULT_OUTPUT_FILE = "evaluation_results_{model}_{mode}_{prompt_config}.json"
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

async def get_response_async(client, prompt_messages, model, semaphore, temperature, max_retries=5):
    """Asynchronously get response from OpenAI API with semaphore control and exponential backoff"""
    async with semaphore:
        for attempt in range(max_retries):
            try:
                completion = await client.chat.completions.create(
                    model=model,
                    messages=prompt_messages,
                    response_format={"type": "json_object"},
                    temperature=temperature,
                )
                return completion.choices[0].message.content
            except Exception as e:
                wait_time = (2 ** attempt) + random.uniform(0, 1)  # Exponential backoff with jitter
                if attempt < max_retries - 1:
                    print(f"\n⚠️  API error (attempt {attempt + 1}/{max_retries}): {str(e)[:100]}")
                    print(f"   Retrying in {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"\n❌ API error after {max_retries} attempts: {str(e)[:100]}")
                    raise  # Re-raise after all retries exhausted


async def evaluate_questions_parallel(
    question_data: Dict[str, dict],
    client,
    model: str,
    mode: str,
    prompt_config: str,
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
        prompt_config (str): 'zero-shot', 'zero-shot-feature-select', 'few-shot', 'digital-twin', 'digital-twin-select', 'digital-twin-feedback', 'digital-twin-cbtact'
        temperature (float): Temperature for model generation
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

    # Prepare few-shot examples if needed
    if prompt_config in ['few-shot', 'few-shot-feature-select', 'few-shot-feature-select-balanced']:
        prepare_few_shot_examples(question_data)

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
        
        # Select the appropriate prompt generation function
        if prompt_config == 'zero-shot':
            text_prompt = generate_zero_shot_prompt(data)
        elif prompt_config == 'zero-shot-feature-select':
            text_prompt = generate_zero_shot_feature_select_prompt(data)
        elif prompt_config == 'zero-shot-feature-select-balanced':
            text_prompt = generate_zero_shot_feature_select_balanced_prompt(data)
        elif prompt_config == 'few-shot':
            text_prompt = generate_few_shot_prompt(data)
        elif prompt_config == 'few-shot-feature-select':
            text_prompt = generate_few_shot_feature_select_prompt(data)
        elif prompt_config == 'few-shot-feature-select-balanced':
            text_prompt = generate_few_shot_feature_select_balanced_prompt(data)
        elif prompt_config == 'enhanced-zero-shot':
            text_prompt = generate_enhanced_zero_shot_prompt(data)
        elif prompt_config == 'zero-shot-prob':
            text_prompt = generate_zero_shot_feature_select_prob_prompt(data)
        elif prompt_config == 'zero-shot-natural-lang':
            text_prompt = generate_zero_shot_natural_lang_prob_prompt(data)
        elif prompt_config == 'digital-twin':
            text_prompt = generate_digital_twin_prompt(data)
        elif prompt_config == 'digital-twin-select':
            text_prompt = generate_digital_twin_select_prompt(data)
        elif prompt_config == 'digital-twin-feedback':
            text_prompt = generate_digital_twin_feedback_prompt(data)
        elif prompt_config == 'digital-twin-cbtact':
            text_prompt = generate_digital_twin_cbtact_prompt(data)
        else:
            raise ValueError(f"Unknown prompt config: {prompt_config}")
        
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
                "predicted_content_probabilities": response.get("predicted_content_probabilities", {}),
                "predicted_design_probabilities": response.get("predicted_design_probabilities", {}),
                "predicted_coping_probabilities": response.get("predicted_coping_probabilities", {}),
                "predicted_quitting_probabilities": response.get("predicted_quitting_probabilities", {}),
                "predicted_content_confidence": response.get("predicted_content_confidence", None),
                "predicted_design_confidence": response.get("predicted_design_confidence", None),
                "predicted_coping_confidence": response.get("predicted_coping_confidence", None),
                "predicted_quitting_confidence": response.get("predicted_quitting_confidence", None),
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
    parser.add_argument('--mode', type=str, choices=['text-only', 'vision'], default='text-only', help="Evaluation mode: 'text-only' or 'vision'")
    parser.add_argument('--model', type=str, default="gpt-4o-mini", help="Name of the OpenAI model to use.")
    parser.add_argument('--prompt-config', type=str, choices=['zero-shot', 'zero-shot-feature-select', 'zero-shot-feature-select-balanced', 'few-shot', 'few-shot-feature-select', 'few-shot-feature-select-balanced', 'enhanced-zero-shot', 'zero-shot-prob', 'zero-shot-natural-lang', 'digital-twin', 'digital-twin-select', 'digital-twin-feedback', 'digital-twin-cbtact'], default='zero-shot', help="Prompt configuration")
    parser.add_argument('--sample-size', type=int, default=None, help="Number of samples to process for testing (if not specified, processes all data)")
    parser.add_argument('--adaptive', action='store_true', help="Run adaptive prompt optimization instead of regular evaluation")
    parser.add_argument('--checkpoint-file', type=str, default=DEFAULT_CHECKPOINT_FILE, help="Path template for checkpoint file (use {model}, {mode}, and {prompt_config} placeholders).")
    parser.add_argument('--output-file', type=str, default=DEFAULT_OUTPUT_FILE, help="Path template for final output file (use {model}, {mode}, and {prompt_config} placeholders).")
    parser.add_argument('--max-concurrent', type=int, default=10, help="Maximum concurrent API calls.")
    parser.add_argument('--checkpoint-interval', type=int, default=DEFAULT_CHECKPOINT_INTERVAL, help="How often to save checkpoint (every N completed items).")
    parser.add_argument('--temperature', type=float, default=None, help="Set the model temperature. Overrides default logic (0.2, or 1.0 for 'o3-' models).")
    parser.add_argument('--data-file', type=str, default='data/processed_llm_data.json', help="Path to data file (default: full dataset). Use canonical splits for manuscript experiments.")
    parser.add_argument('--train-file', type=str, default=None, help="(Digital-twin) Path to TRAIN split for building participant profiles (e.g., data_splits/canonical/train_digital_twin_5050.json)")
    args = parser.parse_args()

    # Check if adaptive optimization is requested
    if args.adaptive:
        print("🔄 Redirecting to adaptive prompt optimization...")
        import subprocess
        import sys
        
        cmd = [
            sys.executable, 
            "analysis-script/run_adaptive_optimization.py",
            "--model", args.model,
            "--mode", args.mode,
            "--initial-prompt-config", "zero-shot-feature-select",
            "--iterations", "15",
            "--test-sample-size", "50",
            "--max-concurrent", str(args.max_concurrent)
        ]
        
        if args.temperature is not None:
            cmd.extend(["--temperature", str(args.temperature)])
        
        subprocess.run(cmd)
        return

    # Format file paths with model, mode, and prompt_config
    checkpoint_file = args.checkpoint_file.format(model=args.model, mode=args.mode, prompt_config=args.prompt_config)
    final_output_file = args.output_file.format(model=args.model, mode=args.mode, prompt_config=args.prompt_config)

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

    print(f"Using prompt configuration: {args.prompt_config}")
    print(f"Loading data from: {args.data_file}")

    # Load data
    with open(args.data_file, "r") as f:
        question_data = json.load(f)

    random.seed(42)
    # If digital-twin, optionally attach profile messages from train split to each test item
    if args.prompt_config.startswith('digital-twin') and args.train_file:
        print(f"Loading digital-twin TRAIN profiles from: {args.train_file}")
        try:
            with open(args.train_file, "r") as f:
                train_items = json.load(f)
        except Exception as e:
            print(f"WARNING: Failed to load train file '{args.train_file}': {e}. Proceeding without profiles.")
            train_items = []

        # Build map: response_id -> list of prior messages with ratings
        response_id_to_profile = {}
        for it in train_items:
            rid = it.get('response_id')
            if rid is None:
                continue
            profile_entry = {
                'input_message': it.get('input_message'),
                'ratings': it.get('ratings', {})
            }
            response_id_to_profile.setdefault(rid, []).append(profile_entry)

        # Attach to test items
        for it in question_data:
            rid = it.get('response_id')
            if rid in response_id_to_profile:
                it['profile_messages'] = response_id_to_profile[rid]
        print("Attached profile_messages to test items where available.")

    # Re-key data for async processing
    question_data = {f"{i}": data for i, data in enumerate(question_data)}
    
    # Sample data if specified
    if args.sample_size:
        print(f"Sampling {args.sample_size} items from {len(question_data)} total items")
        sampled_keys = random.sample(list(question_data.keys()), min(args.sample_size, len(question_data)))
        question_data = {k: question_data[k] for k in sampled_keys}
        print(f"Using {len(question_data)} samples for evaluation")

    client = AsyncOpenAI(api_key=api_key)
    
    try:
        # Run the evaluation
        results = await evaluate_questions_parallel(
            question_data, 
            client, 
            model=args.model,
            mode=args.mode,
            prompt_config=args.prompt_config,
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