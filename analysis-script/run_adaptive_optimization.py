#!/usr/bin/env python3
"""
Adaptive Prompt Optimization for Smoking Cessation Message Evaluation

This script implements a mini-batch SGD approach to optimize prompts using LLM feedback.
It trains on training data and evaluates on test data with iterative prompt improvements.
"""

import argparse
import asyncio
import json
import sys
import os
from typing import Dict, Any, Callable
from datetime import datetime

# Import existing evaluation components
from main_eval import evaluate_questions_parallel, encode_image
from prompt_config import (
    generate_zero_shot_prompt,
    generate_zero_shot_feature_select_prompt,
    generate_zero_shot_feature_select_balanced_prompt,
    generate_few_shot_prompt,
    generate_few_shot_feature_select_prompt,
    generate_few_shot_feature_select_balanced_prompt,
    generate_enhanced_zero_shot_prompt,
    prepare_few_shot_examples
)
from adaptive_prompt_optimizer import AdaptivePromptOptimizer, OptimizationConfig
from openai import AsyncOpenAI

# Configuration mapping
PROMPT_GENERATORS = {
    'zero-shot': generate_zero_shot_prompt,
    'zero-shot-feature-select': generate_zero_shot_feature_select_prompt,
    'zero-shot-feature-select-balanced': generate_zero_shot_feature_select_balanced_prompt,
    'few-shot': generate_few_shot_prompt,
    'few-shot-feature-select': generate_few_shot_feature_select_prompt,
    'few-shot-feature-select-balanced': generate_few_shot_feature_select_balanced_prompt,
    'enhanced-zero-shot': generate_enhanced_zero_shot_prompt,
}

class AdaptiveEvaluationWrapper:
    """Wrapper to integrate adaptive optimization with existing evaluation system"""
    
    def __init__(self, model: str, mode: str, temperature: float = 0.7, max_concurrent: int = 5):
        self.model = model
        self.mode = mode
        self.temperature = temperature
        self.max_concurrent = max_concurrent
        self.client = AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'])
    
    async def evaluate_with_prompt_generator(self, 
                                           data: Dict[str, Any], 
                                           prompt_generator: Callable) -> Dict[str, Any]:
        """Evaluate data using a custom prompt generator"""
        
        # Convert prompt generator to format expected by evaluate_questions_parallel
        async def custom_evaluation(question_data, client, model, mode, prompt_config, temperature, max_concurrent=5, **kwargs):
            """Custom evaluation function using the prompt generator"""
            
            # Create prompt messages for each question
            results = {}
            semaphore = asyncio.Semaphore(max_concurrent)
            
            # Prepare tasks
            tasks = []
            for question_id, question in question_data.items():
                task = self._evaluate_single_question(
                    question_id, question, prompt_generator, semaphore
                )
                tasks.append(task)
            
            # Execute tasks with progress tracking
            completed_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(completed_results):
                question_id = list(question_data.keys())[i]
                if isinstance(result, Exception):
                    print(f"Error evaluating {question_id}: {result}")
                    continue
                results[question_id] = result
            
            return results
        
        return await custom_evaluation(
            data, self.client, self.model, self.mode, "adaptive", 
            self.temperature, self.max_concurrent
        )
    
    async def _evaluate_single_question(self, 
                                       question_id: str, 
                                       question: Dict[str, Any], 
                                       prompt_generator: Callable,
                                       semaphore: asyncio.Semaphore) -> Dict[str, Any]:
        """Evaluate a single question using the prompt generator"""
        
        async with semaphore:
            # Generate prompt
            prompt_text = prompt_generator(question)
            
            # Prepare messages
            if self.mode == 'vision' and 'image_path' in question:
                # Vision mode
                image_base64 = encode_image(question['image_path'])
                if image_base64:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_text},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_base64}"
                                    }
                                }
                            ]
                        }
                    ]
                else:
                    # Fallback to text-only if image not found
                    messages = [{"role": "user", "content": prompt_text}]
            else:
                # Text-only mode
                messages = [{"role": "user", "content": prompt_text}]
            
            # Get response from API
            try:
                completion = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=self.temperature,
                )
                
                response_content = completion.choices[0].message.content
                parsed_response = json.loads(response_content)
                
                # Combine with ground truth for evaluation
                result = {
                    **question,  # Include original data
                    **parsed_response,  # Include model predictions
                    'raw_response': response_content
                }
                
                return result
                
            except Exception as e:
                print(f"Error in API call for {question_id}: {e}")
                return {
                    **question,
                    'error': str(e)
                }

async def run_adaptive_optimization(args):
    """Main function to run adaptive prompt optimization"""
    
    print("🔄 Starting Adaptive Prompt Optimization")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Mode: {args.mode}")
    print(f"Initial config: {args.initial_prompt_config} (zero-shot with selected features baseline)")
    print(f"Iterations: {args.iterations}")
    print(f"Temperature: {args.temperature}")
    print(f"Mini-batch sizes: {args.correct_samples} correct / {args.incorrect_samples} incorrect")
    print(f"Test sample size: {args.test_sample_size}")
    print(f"Test evaluations at iterations: 0, 5, 10, 15")
    
    # Try to find checkpoint file if not provided
    if not args.checkpoint_results:
        checkpoint_pattern = f"checkpoint_results_{args.model}_{args.mode}_{args.initial_prompt_config}.json"
        if os.path.exists(checkpoint_pattern):
            args.checkpoint_results = checkpoint_pattern
            print(f"Found existing checkpoint: {checkpoint_pattern}")
        else:
            print("No existing checkpoint found - will run full evaluation for iteration 0")
    else:
        print(f"Using provided checkpoint: {args.checkpoint_results}")
    
    print("=" * 50)
    
    # Setup optimization configuration
    config = OptimizationConfig(
        n_iterations=args.iterations,
        minibatch_size_correct=args.correct_samples,
        minibatch_size_incorrect=args.incorrect_samples,
        max_prompt_change_pct=args.max_change_pct,
        temperature=args.temperature,
        max_concurrent=args.max_concurrent,
        checkpoint_interval=args.checkpoint_interval
    )
    
    # Initialize optimizer
    optimizer = AdaptivePromptOptimizer(
        train_data_path=args.train_data,
        test_data_path=args.test_data,
        initial_prompt_config=args.initial_prompt_config,
        model=args.model,
        mode=args.mode,
        config=config,
        test_sample_size=args.test_sample_size,
        checkpoint_results_path=args.checkpoint_results
    )
    
    # Initialize evaluation wrapper
    evaluator = AdaptiveEvaluationWrapper(
        model=args.model,
        mode=args.mode,
        temperature=args.temperature,
        max_concurrent=args.max_concurrent
    )
    
    # Get initial prompt generator
    initial_prompt_generator = PROMPT_GENERATORS[args.initial_prompt_config]
    
    # Prepare few-shot examples if needed
    if 'few-shot' in args.initial_prompt_config:
        all_data = {**optimizer.train_data, **optimizer.test_data}
        prepare_few_shot_examples(all_data)
        print(f"✅ Prepared few-shot examples for {len(all_data)} total samples")
    
    # Run optimization
    try:
        results = await optimizer.optimize(
            evaluate_fn=evaluator.evaluate_with_prompt_generator,
            initial_prompt_generator=initial_prompt_generator
        )
        
        # Save final results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"adaptive_optimization_results_{args.model}_{args.mode}_{timestamp}.json"
        
        final_results = {
            'experiment_config': {
                'model': args.model,
                'mode': args.mode,
                'initial_prompt_config': args.initial_prompt_config,
                'optimization_config': config.__dict__,
                'train_data_size': len(optimizer.train_data),
                'test_data_size': len(optimizer.test_data)
            },
            'results': results,
            'timestamp': timestamp
        }
        
        with open(output_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print("\n🎉 Optimization Complete!")
        print("=" * 50)
        if results['initial_performance'] is not None:
            print(f"Initial performance (iter 0): {results['initial_performance']:.3f}")
        if results['best_performance'] is not None:
            print(f"Best performance: {results['best_performance']:.3f}")
        if results['improvement'] is not None:
            print(f"Total improvement: {results['improvement']:+.3f}")
        print(f"Test evaluations at: {results['test_eval_iterations']}")
        if results['test_performances']:
            print(f"Test performance progression: {[f'{p:.3f}' for p in results['test_performances']]}")
        print(f"Results saved to: {output_file}")
        
        return results
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description="Adaptive Prompt Optimization for Smoking Cessation Message Evaluation")
    
    # Model and mode arguments
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                       choices=["gpt-4o-mini", "gpt-4o", "o3-2025-04-16"],
                       help="Model to use for evaluation")
    parser.add_argument("--mode", type=str, default="vision",
                       choices=["vision", "text-only"],
                       help="Evaluation mode")
    
    # Data paths
    parser.add_argument("--train-data", type=str, 
                       default="data_splits/train_data.json",
                       help="Path to training data")
    parser.add_argument("--test-data", type=str,
                       default="data_splits/test_data.json", 
                       help="Path to test data")
    parser.add_argument("--test-sample-size", type=int, default=50,
                       help="Number of test samples to evaluate (for speed)")
    parser.add_argument("--checkpoint-results", type=str, default=None,
                       help="Path to existing checkpoint results to reuse (avoids re-evaluation)")
    
    # Optimization parameters
    parser.add_argument("--initial-prompt-config", type=str,
                       default="zero-shot-feature-select",
                       choices=list(PROMPT_GENERATORS.keys()),
                       help="Initial prompt configuration")
    parser.add_argument("--iterations", type=int, default=15,
                       help="Number of optimization iterations")
    parser.add_argument("--correct-samples", type=int, default=10,
                       help="Number of correct samples per mini-batch")
    parser.add_argument("--incorrect-samples", type=int, default=10,
                       help="Number of incorrect samples per mini-batch")
    parser.add_argument("--max-change-pct", type=float, default=0.10,
                       help="Maximum prompt change percentage per iteration")
    
    # Technical parameters  
    parser.add_argument("--temperature", type=float, default=0.5,
                       help="Temperature for model generation")
    parser.add_argument("--max-concurrent", type=int, default=5,
                       help="Maximum concurrent API calls")
    parser.add_argument("--checkpoint-interval", type=int, default=1,
                       help="Save checkpoint every N iterations")
    
    args = parser.parse_args()
    
    # Validate data files exist
    if not os.path.exists(args.train_data):
        print(f"❌ Training data not found: {args.train_data}")
        sys.exit(1)
    
    if not os.path.exists(args.test_data):
        print(f"❌ Test data not found: {args.test_data}")
        sys.exit(1)
    
    # Check API key
    if 'OPENAI_API_KEY' not in os.environ:
        print("❌ OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    # Run optimization
    asyncio.run(run_adaptive_optimization(args))

if __name__ == "__main__":
    main()