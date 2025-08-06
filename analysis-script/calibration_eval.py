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
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from prompt_config import generate_continuous_rating_prompt

# Replace with your own API key
api_key = os.environ['CHEN_OPENAI_API_KEY']

# Default file paths
DEFAULT_EVAL_FILE = "evaluation_continuous_{model}_{mode}.json"
DEFAULT_CALIBRATION_FILE = "calibration_continuous_{model}_{mode}.json" 
DEFAULT_CALIBRATED_FILE = "calibrated_results_{model}_{mode}.json"
DEFAULT_CHECKPOINT_INTERVAL = 50

# Global variable to track if we're shutting down
shutting_down = False

# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    global shutting_down
    print("\nReceived interrupt signal. Finishing current tasks and saving progress...")
    shutting_down = True

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

def convert_categorical_to_continuous(categorical_rating, dimension):
    """Convert categorical ratings to continuous scale 1-5"""
    if dimension in ['content', 'design']:
        mapping = {
            "Very poor": 1.0,
            "Poor": 2.0,
            "Acceptable": 3.0,
            "Good": 4.0,
            "Very good": 5.0
        }
    else:  # coping, quitting
        mapping = {
            "Not at all helpful": 1.0,
            "Somewhat helpful": 2.0,
            "Moderately helpful": 3.0,
            "Very helpful": 4.0,
            "Extremely helpful": 5.0
        }
    return mapping.get(categorical_rating, np.nan)

async def evaluate_continuous_ratings(
    question_data: Dict[str, dict],
    client,
    model: str,
    mode: str,
    temperature: float,
    max_concurrent: int = 5,
    checkpoint_file: str = None,
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL
) -> Dict[str, dict]:
    """
    Evaluate questions with continuous rating predictions.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    results = {}
    completed_count = 0
    global shutting_down

    # Load previous results if they exist
    if checkpoint_file and os.path.exists(checkpoint_file):
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
            return {}
            
        # Generate continuous rating prompt
        text_prompt = generate_continuous_rating_prompt(data)
        
        prompt_messages = [{"role": "user", "content": []}]
        prompt_messages[0]["content"].append({"type": "text", "text": text_prompt})

        if mode == 'vision':
            if data.get('image_path') and os.path.exists(data['image_path']):
                base64_image = encode_image(data['image_path'])
                if base64_image:
                    prompt_messages[0]["content"].append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    })
            else:
                print(f"Warning: Image path not found for qid {qid}, running as text-only.")

        try:
            response_json_str = await get_response_async(client, prompt_messages, model, semaphore, temperature)
            response = json.loads(response_json_str)
            
            # Ensure numerical values
            for dim in ['content', 'design', 'coping', 'quitting']:
                pred_key = f'predicted_{dim}'
                conf_key = f'confidence_{dim}'
                if pred_key in response:
                    try:
                        response[pred_key] = float(response[pred_key])
                    except (ValueError, TypeError):
                        response[pred_key] = np.nan
                if conf_key in response:
                    try:
                        response[conf_key] = float(response[conf_key])
                    except (ValueError, TypeError):
                        response[conf_key] = np.nan
                        
        except Exception as e:
            print(f"Error processing {qid}: {e}")
            response = {
                "predicted_content": np.nan,
                "predicted_design": np.nan, 
                "predicted_coping": np.nan,
                "predicted_quitting": np.nan,
                "confidence_content": np.nan,
                "confidence_design": np.nan,
                "confidence_coping": np.nan,
                "confidence_quitting": np.nan,
                "error": str(e)
            }

        # Convert ground truth categorical to continuous
        result = {
            qid: {
                "response_id": data['response_id'],
                "input_message": data['input_message'],
                "metadata": data['metadata'],
                "ground_truth_content_categorical": data['ratings'].get('content', ""),
                "ground_truth_design_categorical": data['ratings'].get('design', ""),
                "ground_truth_coping_categorical": data['ratings'].get('coping', ""),
                "ground_truth_quitting_categorical": data['ratings'].get('quitting', ""),
                "ground_truth_content": convert_categorical_to_continuous(data['ratings'].get('content'), 'content'),
                "ground_truth_design": convert_categorical_to_continuous(data['ratings'].get('design'), 'design'),
                "ground_truth_coping": convert_categorical_to_continuous(data['ratings'].get('coping'), 'coping'),
                "ground_truth_quitting": convert_categorical_to_continuous(data['ratings'].get('quitting'), 'quitting'),
                "predicted_content": response.get("predicted_content", np.nan),
                "predicted_design": response.get("predicted_design", np.nan),
                "predicted_coping": response.get("predicted_coping", np.nan),
                "predicted_quitting": response.get("predicted_quitting", np.nan),
                "confidence_content": response.get("confidence_content", np.nan),
                "confidence_design": response.get("confidence_design", np.nan),
                "confidence_coping": response.get("confidence_coping", np.nan),
                "confidence_quitting": response.get("confidence_quitting", np.nan),
                "image_description": response.get("image_description", ""),
                "explanation": response.get("explanation", "")
            }
        }
        
        return result

    # Create tasks
    tasks = []
    for qid, item in pending_questions.items():
        task = asyncio.create_task(query_llm(qid, item))
        tasks.append(task)

    # Track progress
    pbar = tqdm(total=len(tasks), desc=f"Processing {len(tasks)} items")
    
    # Process tasks as they complete
    for task in asyncio.as_completed(tasks):
        if shutting_down:
            for t in tasks:
                if not t.done():
                    t.cancel()
            break
            
        result = await task
        if result:
            results.update(result)
            completed_count += 1
            pbar.update(1)
            
            # Save checkpoint periodically
            if checkpoint_file and completed_count % checkpoint_interval == 0:
                print(f"\nSaving checkpoint after {completed_count} completed items...")
                with open(checkpoint_file, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"Checkpoint saved to {checkpoint_file}")
                
            if shutting_down:
                break

    pbar.close()
    
    # Final save
    if checkpoint_file:
        print(f"Saving final results with {len(results)} items...")
        with open(checkpoint_file, "w") as f:
            json.dump(results, f, indent=2)
    
    return results

def calibrate_predictions(eval_results, calibration_results):
    """
    Calibrate predictions using isotonic regression.
    """
    calibrated_results = {}
    dimensions = ['content', 'design', 'coping', 'quitting']
    
    print("Performing calibration...")
    
    for dim in dimensions:
        print(f"Calibrating {dim} dimension...")
        
        # Extract calibration data
        cal_predictions = []
        cal_ground_truth = []
        
        for item in calibration_results.values():
            pred = item.get(f'predicted_{dim}')
            truth = item.get(f'ground_truth_{dim}')
            if not (np.isnan(pred) or np.isnan(truth)):
                cal_predictions.append(pred)
                cal_ground_truth.append(truth)
        
        if len(cal_predictions) < 10:
            print(f"Warning: Not enough calibration data for {dim} ({len(cal_predictions)} samples)")
            continue
            
        # Fit isotonic regression
        iso_reg = IsotonicRegression(out_of_bounds='clip')
        iso_reg.fit(cal_predictions, cal_ground_truth)
        
        # Apply calibration to evaluation data
        for qid, item in eval_results.items():
            pred = item.get(f'predicted_{dim}')
            if not np.isnan(pred):
                calibrated_pred = iso_reg.predict([pred])[0]
                if qid not in calibrated_results:
                    calibrated_results[qid] = item.copy()
                calibrated_results[qid][f'calibrated_{dim}'] = calibrated_pred
            else:
                if qid not in calibrated_results:
                    calibrated_results[qid] = item.copy()
                calibrated_results[qid][f'calibrated_{dim}'] = np.nan
    
    return calibrated_results

async def main():
    parser = argparse.ArgumentParser(description="Evaluate smoking cessation messages with continuous ratings and calibration.")
    parser.add_argument('--mode', type=str, choices=['text-only', 'vision'], required=True, help="Evaluation mode")
    parser.add_argument('--model', type=str, default="gpt-4o-mini", help="OpenAI model to use")
    parser.add_argument('--eval-samples', type=int, default=500, help="Number of samples for evaluation")
    parser.add_argument('--cal-samples', type=int, default=500, help="Number of samples for calibration")
    parser.add_argument('--max-concurrent', type=int, default=10, help="Maximum concurrent API calls")
    parser.add_argument('--temperature', type=float, default=None, help="Model temperature")
    args = parser.parse_args()

    # Determine temperature
    temperature = args.temperature
    if temperature is None:
        if 'o3-' in args.model:
            temperature = 1.0
        else:
            temperature = 0.2
    
    print(f"Using model: {args.model}, mode: {args.mode}, temperature: {temperature}")
    print(f"Evaluation samples: {args.eval_samples}, Calibration samples: {args.cal_samples}")

    # Load data
    with open("data/processed_llm_data.json", "r") as f:
        all_data = json.load(f)

    # Set fixed seed for reproducible splits
    random.seed(42)
    all_indices = list(range(len(all_data)))
    random.shuffle(all_indices)
    
    # Split data
    eval_indices = all_indices[:args.eval_samples]
    cal_indices = all_indices[args.eval_samples:args.eval_samples + args.cal_samples]
    
    eval_data = {f"eval_{i}": all_data[i] for i in eval_indices}
    cal_data = {f"cal_{i}": all_data[i] for i in cal_indices}
    
    print(f"Evaluation set: {len(eval_data)} samples")
    print(f"Calibration set: {len(cal_data)} samples")

    client = AsyncOpenAI(api_key=api_key)
    
    # File paths
    eval_file = DEFAULT_EVAL_FILE.format(model=args.model, mode=args.mode)
    cal_file = DEFAULT_CALIBRATION_FILE.format(model=args.model, mode=args.mode)
    calibrated_file = DEFAULT_CALIBRATED_FILE.format(model=args.model, mode=args.mode)
    
    try:
        # Run evaluation
        print("Running evaluation...")
        eval_results = await evaluate_continuous_ratings(
            eval_data, client, args.model, args.mode, temperature,
            max_concurrent=args.max_concurrent, checkpoint_file=eval_file
        )
        
        # Run calibration data collection
        print("Running calibration data collection...")
        cal_results = await evaluate_continuous_ratings(
            cal_data, client, args.model, args.mode, temperature,
            max_concurrent=args.max_concurrent, checkpoint_file=cal_file
        )
        
        # Perform calibration
        calibrated_results = calibrate_predictions(eval_results, cal_results)
        
        # Save calibrated results
        with open(calibrated_file, "w") as f:
            json.dump(calibrated_results, f, indent=2)
        
        print(f"Evaluation results saved to: {eval_file}")
        print(f"Calibration data saved to: {cal_file}")
        print(f"Calibrated results saved to: {calibrated_file}")
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
    finally:
        print("Process completed.")

if __name__ == "__main__":
    asyncio.run(main()) 