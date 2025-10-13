import json
import random
import asyncio
import os
from typing import Dict, List, Tuple, Any
from openai import AsyncOpenAI
import logging
from dataclasses import dataclass
from datetime import datetime

@dataclass
class OptimizationConfig:
    """Configuration for adaptive prompt optimization"""
    n_iterations: int = 15
    minibatch_size_correct: int = 10
    minibatch_size_incorrect: int = 10
    max_prompt_change_pct: float = 0.25  # Max 10% change per iteration
    temperature: float = 0.5
    max_concurrent: int = 5
    checkpoint_interval: int = 1  # Save after each iteration
    test_eval_iterations: list = None  # Which iterations to evaluate on test set
    
    def __post_init__(self):
        if self.test_eval_iterations is None:
            self.test_eval_iterations = [0, 5, 10, 15]

class AdaptivePromptOptimizer:
    """
    Adaptive prompt optimization using mini-batch SGD approach with LLM-guided prompt rewriting.
    
    The system:
    1. Trains on training examples using mini-batches
    2. Samples correct/incorrect predictions 
    3. Uses LLM to rewrite prompts with constraints
    4. Evaluates on test set after each iteration
    """
    
    def __init__(self, 
                 train_data_path: str,
                 test_data_path: str,
                 initial_prompt_config: str,
                 model: str,
                 mode: str,
                 config: OptimizationConfig = None,
                 test_sample_size: int = None,
                 checkpoint_results_path: str = None):
        
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.initial_prompt_config = initial_prompt_config
        self.model = model
        self.mode = mode
        self.config = config or OptimizationConfig()
        self.test_sample_size = test_sample_size
        self.checkpoint_results_path = checkpoint_results_path
        
        # Setup logging first
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'])
        
        # Load data
        self.train_data = self._load_data(train_data_path)
        self.test_data = self._load_data(test_data_path)
        
        # Load existing checkpoint results if provided
        self.baseline_results = None
        if checkpoint_results_path and os.path.exists(checkpoint_results_path):
            self.baseline_results = self._load_data(checkpoint_results_path)
            self.logger.info(f"Loaded {len(self.baseline_results)} baseline evaluation results from {checkpoint_results_path}")
        
        # Sample test data if requested
        if self.test_sample_size and len(self.test_data) > self.test_sample_size:
            import random
            random.seed(42)  # For reproducibility
            test_keys = list(self.test_data.keys())
            sampled_keys = random.sample(test_keys, self.test_sample_size)
            self.test_data = {k: self.test_data[k] for k in sampled_keys}
            self.logger.info(f"Sampled {len(self.test_data)} test cases from {len(test_keys)} total")
        
        # Track optimization history
        self.optimization_history = []
        self.current_prompt = None
        self.iteration = 0
        
    def _load_data(self, path: str) -> Dict[str, Any]:
        """Load data from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        # Convert list to dict keyed by response_id if needed
        if isinstance(data, list):
            return {item['response_id']: item for item in data}
        return data
    
    async def optimize(self, 
                      evaluate_fn: callable,
                      initial_prompt_generator: callable) -> Dict[str, Any]:
        """
        Main optimization loop
        
        Args:
            evaluate_fn: Function to evaluate prompts on data
            initial_prompt_generator: Function to generate initial prompt
            
        Returns:
            Optimization results including best prompt and performance history
        """
        
        self.logger.info(f"Starting adaptive prompt optimization for {self.config.n_iterations} iterations")
        self.logger.info(f"Train data: {len(self.train_data)} samples")
        self.logger.info(f"Test data: {len(self.test_data)} samples")
        
        # Initialize with baseline prompt
        self.current_prompt = initial_prompt_generator
        
        # Use checkpoint results for initial performance if available
        if self.baseline_results and 0 in self.config.test_eval_iterations:
            self.logger.info("Using baseline results for initial performance...")
            initial_performance = self._calculate_performance(self.baseline_results)
            
            # Get initial prompt text
            sample_data = list(self.train_data.values())[0] if self.train_data else {}
            initial_prompt_text = initial_prompt_generator(sample_data) if sample_data else "No sample data available"
            
            self.optimization_history.append({
                'iteration': 0,
                'prompt_config': self.initial_prompt_config,
                'test_performance': initial_performance,
                'prompt_text': initial_prompt_text,
                'test_evaluated': True
            })
            
            self.logger.info(f"Initial test performance (from checkpoint): {initial_performance}")
        elif 0 in self.config.test_eval_iterations:
            self.logger.info("Evaluating initial performance on test set...")
            initial_results = await evaluate_fn(self.test_data, self.current_prompt)
            initial_performance = self._calculate_performance(initial_results)
            
            # Get initial prompt text
            sample_data = list(self.train_data.values())[0] if self.train_data else {}
            initial_prompt_text = initial_prompt_generator(sample_data) if sample_data else "No sample data available"
            
            self.optimization_history.append({
                'iteration': 0,
                'prompt_config': self.initial_prompt_config,
                'test_performance': initial_performance,
                'prompt_text': initial_prompt_text,
                'test_evaluated': True
            })
            
            self.logger.info(f"Initial test performance: {initial_performance}")
        else:
            # Just record initial state without test evaluation
            # Get initial prompt text
            sample_data = list(self.train_data.values())[0] if self.train_data else {}
            initial_prompt_text = initial_prompt_generator(sample_data) if sample_data else "No sample data available"
            
            self.optimization_history.append({
                'iteration': 0,
                'prompt_config': self.initial_prompt_config,
                'test_performance': None,
                'prompt_text': initial_prompt_text,
                'test_evaluated': False
            })
        
        # Main optimization loop
        for iteration in range(1, self.config.n_iterations + 1):
            self.iteration = iteration
            self.logger.info(f"\n=== Iteration {iteration}/{self.config.n_iterations} ===")
            
            # Step 1: Get training samples for this iteration
            if self.baseline_results and iteration == 1:
                # For iteration 1, use baseline results to sample from
                self.logger.info("Using baseline results to sample training examples...")
                correct_samples, incorrect_samples = self._sample_minibatch(self.baseline_results)
            else:
                # For later iterations, evaluate on small training sample (10 examples)
                self.logger.info("Evaluating current prompt on small training sample...")
                small_train_sample = self._get_small_train_sample(10)
                train_results = await evaluate_fn(small_train_sample, self.current_prompt)
                correct_samples, incorrect_samples = self._sample_minibatch(train_results)
            
            # Step 3: Generate improved prompt using LLM
            new_prompt_generator = await self._generate_improved_prompt(
                correct_samples, incorrect_samples, self.current_prompt
            )
            
            # Step 4: Evaluate on test set only at specified iterations
            test_performance = None
            test_evaluated = False
            
            if iteration in self.config.test_eval_iterations:
                self.logger.info(f"📊 Evaluating test set performance at iteration {iteration}...")
                test_results = await evaluate_fn(self.test_data, new_prompt_generator)
                test_performance = self._calculate_performance(test_results)
                test_evaluated = True
                
                # Update current prompt if improvement (only when we have test results)
                last_test_result = None
                for hist in reversed(self.optimization_history):
                    if hist.get('test_evaluated', False):
                        last_test_result = hist['test_performance']
                        break
                
                if last_test_result and test_performance['overall_accuracy'] > last_test_result['overall_accuracy']:
                    self.current_prompt = new_prompt_generator
                    self.logger.info(f"✅ Iteration {iteration}: Improved! New accuracy: {test_performance['overall_accuracy']:.3f}")
                elif last_test_result:
                    self.logger.info(f"❌ Iteration {iteration}: No improvement. Accuracy: {test_performance['overall_accuracy']:.3f}")
                else:
                    # First test evaluation, always accept
                    self.current_prompt = new_prompt_generator
                    self.logger.info(f"📈 Iteration {iteration}: First test evaluation. Accuracy: {test_performance['overall_accuracy']:.3f}")
            else:
                # Always update prompt when not evaluating (exploratory)
                self.current_prompt = new_prompt_generator
                self.logger.info(f"🔄 Iteration {iteration}: Updated prompt (no test evaluation)")
            
            # Step 5: Get sample prompt text to save
            sample_data = list(self.train_data.values())[0] if self.train_data else {}
            current_prompt_text = self.current_prompt(sample_data) if self.current_prompt else "No prompt available"
            
            # Step 6: Record history
            self.optimization_history.append({
                'iteration': iteration,
                'test_performance': test_performance,
                'test_evaluated': test_evaluated,
                'train_correct_samples': len(correct_samples),
                'train_incorrect_samples': len(incorrect_samples),
                'prompt_text': current_prompt_text,
                'prompt_analysis': {
                    'correct_sample_count': len(correct_samples),
                    'incorrect_sample_count': len(incorrect_samples)
                }
            })
            
            # Step 7: Save checkpoint
            if iteration % self.config.checkpoint_interval == 0:
                await self._save_checkpoint(iteration)
        
        # Return optimization results
        test_performances = [h['test_performance']['overall_accuracy'] for h in self.optimization_history 
                           if h.get('test_evaluated', False) and h['test_performance'] is not None]
        
        initial_perf = None
        best_perf = None
        improvement = None
        
        if test_performances:
            best_perf = max(test_performances)
            initial_perf = test_performances[0] if test_performances else None
            improvement = best_perf - initial_perf if initial_perf is not None else None
        
        return {
            'optimization_history': self.optimization_history,
            'best_performance': best_perf,
            'initial_performance': initial_perf,
            'improvement': improvement,
            'test_eval_iterations': self.config.test_eval_iterations,
            'test_performances': test_performances,
            'final_prompt_text': self.current_prompt(list(self.train_data.values())[0]) if self.current_prompt and self.train_data else "No final prompt available"
        }
    
    def _get_small_train_sample(self, sample_size: int) -> Dict[str, Any]:
        """Get a small random sample from training data"""
        import random
        train_keys = list(self.train_data.keys())
        sampled_keys = random.sample(train_keys, min(sample_size, len(train_keys)))
        return {k: self.train_data[k] for k in sampled_keys}
    
    def _sample_minibatch(self, train_results: Dict[str, Any]) -> Tuple[List[Any], List[Any]]:
        """Sample mini-batch of correct and incorrect predictions"""
        
        correct_predictions = []
        incorrect_predictions = []
        
        for result_id, result in train_results.items():
            if self._is_prediction_correct(result):
                correct_predictions.append(result)
            else:
                incorrect_predictions.append(result)
        
        # Sample mini-batches
        correct_sample = random.sample(
            correct_predictions, 
            min(len(correct_predictions), self.config.minibatch_size_correct)
        )
        
        incorrect_sample = random.sample(
            incorrect_predictions,
            min(len(incorrect_predictions), self.config.minibatch_size_incorrect)  
        )
        
        self.logger.info(f"Sampled {len(correct_sample)} correct, {len(incorrect_sample)} incorrect predictions")
        return correct_sample, incorrect_sample
    
    def _is_prediction_correct(self, result: Dict[str, Any]) -> bool:
        """Check if prediction matches ground truth across all dimensions"""
        dimensions = ['content', 'design', 'coping', 'quitting']
        
        for dim in dimensions:
            predicted = result.get(f'predicted_{dim}', '').strip()
            ground_truth = result.get('ratings', {}).get(dim, '').strip()
            if predicted != ground_truth:
                return False
        return True
    
    async def _generate_improved_prompt(self, 
                                       correct_samples: List[Any], 
                                       incorrect_samples: List[Any],
                                       current_prompt_generator: callable) -> callable:
        """Generate improved prompt using LLM analysis of correct/incorrect predictions"""
        
        # Extract patterns from samples
        analysis_prompt = self._create_analysis_prompt(correct_samples, incorrect_samples, current_prompt_generator)
        
        # Get LLM analysis and suggested improvements
        messages = [{"role": "user", "content": analysis_prompt}]
        
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",  # Use cheaper model for prompt optimization
            messages=messages,
            response_format={"type": "json_object"},
            temperature=self.config.temperature
        )
        
        analysis_result = json.loads(response.choices[0].message.content)
        
        # Create new prompt generator based on analysis
        return self._create_optimized_prompt_generator(analysis_result, current_prompt_generator)
    
    def _create_analysis_prompt(self, 
                               correct_samples: List[Any], 
                               incorrect_samples: List[Any],
                               current_prompt_generator: callable) -> str:
        """Create prompt for LLM to analyze and improve the evaluation prompt"""
        
        # Use ALL examples for analysis - no truncation!
        correct_examples = correct_samples
        incorrect_examples = incorrect_samples
        
        # Get a sample of the current prompt to show structure
        sample_data = list(self.train_data.values())[0] if self.train_data else {}
        current_prompt_example = current_prompt_generator(sample_data) if sample_data else "No sample available"
        
        analysis_prompt = f"""
You are an expert in prompt engineering for smoking cessation message evaluation.

TASK: Analyze the prediction patterns and provide an improved prompt template that will maximize accuracy.

CURRENT PROMPT EXAMPLE (showing structure):
{current_prompt_example}

PERFORMANCE OBSERVATIONS:

CORRECT PREDICTIONS (ALL {len(correct_samples)} examples):
{json.dumps(correct_examples, indent=2) if correct_examples else "None"}

INCORRECT PREDICTIONS (ALL {len(incorrect_samples)} examples):
{json.dumps(incorrect_examples, indent=2) if incorrect_examples else "None"}

ANALYSIS TASK:
1. Analyze ALL the correct and incorrect predictions above
2. Identify patterns in what leads to correct vs incorrect predictions
3. Consider participant demographics, message types, and rating patterns
4. Provide a new prompt function that will improve accuracy

IMPORTANT: Return a Python function string that takes 'data' parameter and returns the full prompt.
The function should:
- Access data['input_message'] for the message
- Access data['metadata'] for participant info
- Access data['response_id'] for the ID
- Use only these selected features from metadata: age_years, gender_identity, race_ethnicity, quit_motivation_level, social_support_to_quit

Return your response in this JSON format:
{{
    "analysis": "Detailed analysis of patterns in correct vs incorrect predictions",
    "improved_prompt_function": "def generate_prompt(data):\\n    # Your improved prompt generation code here\\n    return prompt"
}}
"""
        return analysis_prompt
    
    def _create_optimized_prompt_generator(self, 
                                          analysis_result: Dict[str, Any],
                                          current_prompt_generator: callable) -> callable:
        """Create new prompt generator from the LLM's function string"""
        
        # Get the improved prompt function from the LLM
        improved_function_str = analysis_result.get('improved_prompt_function', '')
        
        if not improved_function_str:
            # Fallback to current prompt if no improvement provided
            self.logger.warning("No improved prompt function provided, keeping current prompt")
            return current_prompt_generator
        
        try:
            # Execute the function string to create the actual function
            exec_globals = {}
            exec(improved_function_str, exec_globals)
            
            # Get the function from the executed code
            if 'generate_prompt' in exec_globals:
                optimized_generator = exec_globals['generate_prompt']
                self.logger.info("Successfully created new prompt generator from LLM function")
                return optimized_generator
            else:
                self.logger.warning("Could not find generate_prompt function in LLM response")
                return current_prompt_generator
                
        except Exception as e:
            self.logger.error(f"Error creating prompt generator: {e}")
            return current_prompt_generator
    
    def _calculate_performance(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate performance metrics from evaluation results"""
        
        if not results:
            return {'overall_accuracy': 0.0}
        
        dimensions = ['content', 'design', 'coping', 'quitting']
        correct_counts = {dim: 0 for dim in dimensions}
        total_count = len(results)
        
        for result in results.values():
            for dim in dimensions:
                predicted = result.get(f'predicted_{dim}', '').strip()
                ground_truth = result.get('ratings', {}).get(dim, '').strip()
                if predicted == ground_truth:
                    correct_counts[dim] += 1
        
        # Calculate accuracies
        accuracies = {dim: correct_counts[dim] / total_count for dim in dimensions}
        overall_accuracy = sum(accuracies.values()) / len(dimensions)
        
        return {
            'overall_accuracy': overall_accuracy,
            **{f'{dim}_accuracy': acc for dim, acc in accuracies.items()},
            'total_samples': total_count
        }
    
    async def _save_checkpoint(self, iteration: int):
        """Save optimization checkpoint with prompt evolution tracking"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_file = f"adaptive_optimization_checkpoint_{self.model}_{self.mode}_{iteration}_{timestamp}.json"
        
        # Create prompt evolution summary
        prompt_evolution = []
        for hist in self.optimization_history:
            prompt_evolution.append({
                'iteration': hist['iteration'],
                'test_evaluated': hist.get('test_evaluated', False),
                'test_performance': hist.get('test_performance'),
                'prompt_length': len(hist.get('prompt_text', '')),
                'prompt_preview': hist.get('prompt_text', '')[:300] + "..." if len(hist.get('prompt_text', '')) > 300 else hist.get('prompt_text', ''),
                'train_samples': {
                    'correct': hist.get('train_correct_samples', 0),
                    'incorrect': hist.get('train_incorrect_samples', 0)
                }
            })
        
        checkpoint_data = {
            'iteration': iteration,
            'timestamp': timestamp,
            'config': {
                'n_iterations': self.config.n_iterations,
                'minibatch_size_correct': self.config.minibatch_size_correct,
                'minibatch_size_incorrect': self.config.minibatch_size_incorrect,
                'max_prompt_change_pct': self.config.max_prompt_change_pct,
                'test_eval_iterations': self.config.test_eval_iterations
            },
            'optimization_history': self.optimization_history,  # Full history with complete prompts
            'prompt_evolution_summary': prompt_evolution,  # Condensed view
            'model': self.model,
            'mode': self.mode,
            'initial_prompt_config': self.initial_prompt_config,
            'current_iteration': iteration,
            'data_info': {
                'train_size': len(self.train_data),
                'test_size': len(self.test_data),
                'checkpoint_results_used': self.checkpoint_results_path is not None
            }
        }
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        self.logger.info(f"💾 Checkpoint saved: {checkpoint_file}")
        self.logger.info(f"   - Full prompt texts saved for all {len(self.optimization_history)} iterations")
        self.logger.info(f"   - Prompt evolution summary included")