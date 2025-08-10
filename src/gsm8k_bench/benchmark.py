"""
Core benchmark runner for GSM8K prompting techniques.
"""

import time
import logging
from typing import Dict, List, Optional, Any
from collections import defaultdict
import numpy as np

import dspy
from .utils import (
    BenchmarkResult, math_accuracy, safe_execute_with_timeout,
    create_prediction_record, ProgressTracker, setup_dspy_with_config
)
from .techniques import (
    ZeroShotModule, FewShotModule, CoTModule, 
    SelfConsistencyModule, PrologModule
)

logger = logging.getLogger(__name__)


class GSM8KBenchmark:
    """Main benchmark runner for GSM8K prompting techniques"""
    
    def __init__(self, test_dataset: List[dspy.Example], 
                 selected_techniques: Optional[List[str]] = None,
                 model_config: Optional[Dict[str, Any]] = None):
        """
        Initialize benchmark with dataset and configuration.
        
        Args:
            test_dataset: List of DSPy examples to test on
            selected_techniques: List of technique names to run (None for all)
            model_config: Model configuration dictionary
        """
        
        self.test_set = test_dataset
        self.model_config = model_config or {}
        
        # Setup DSPy
        if not setup_dspy_with_config(self.model_config):
            raise RuntimeError("Failed to configure DSPy")
        
        # Initialize techniques
        self.all_techniques = {
            "zero_shot": ("1. Zero-Shot", ZeroShotModule),
            "few_shot": ("2. Few-Shot", FewShotModule),
            "cot": ("3. Chain-of-Thought", CoTModule),
            "self_consistency": ("4. Self-Consistency", SelfConsistencyModule),
            "prolog_style": ("5. Prolog-Style", PrologModule),
        }
        
        # Filter techniques if specified
        if selected_techniques:
            self.modules = {}
            for tech_name in selected_techniques:
                if tech_name in self.all_techniques:
                    display_name, module_class = self.all_techniques[tech_name]
                    self.modules[display_name] = self._create_module(module_class)
                else:
                    logger.warning(f"Unknown technique: {tech_name}")
        else:
            # Use all techniques
            self.modules = {}
            for tech_name, (display_name, module_class) in self.all_techniques.items():
                self.modules[display_name] = self._create_module(module_class)
        
        logger.info(f"Initialized benchmark with {len(self.modules)} techniques")
    
    def _create_module(self, module_class):
        """Create a technique module with proper configuration"""
        try:
            if module_class == SelfConsistencyModule:
                # Configure self-consistency with reasonable parameters
                return module_class(n_samples=5)
            else:
                return module_class()
        except Exception as e:
            logger.error(f"Error creating module {module_class.__name__}: {e}")
            raise
    
    def evaluate_module(self, module, name: str, 
                       timeout_seconds: int = 30) -> BenchmarkResult:
        """
        Evaluate a single technique module.
        
        Args:
            module: The technique module to evaluate
            name: Display name for the technique
            timeout_seconds: Timeout for each prediction
            
        Returns:
            BenchmarkResult with evaluation metrics
        """
        
        logger.info(f" Evaluating {name}...")
        
        correct = 0
        total = len(self.test_set)
        errors = 0
        times = []
        predictions = []
        
        # Progress tracking
        progress = ProgressTracker(total, name)
        
        for i, example in enumerate(self.test_set):
            try:
                # Execute with timeout protection
                result, exec_time, error = safe_execute_with_timeout(
                    module, timeout_seconds, question=example.question
                )
                
                times.append(exec_time)
                
                if error:
                    # Execution failed
                    errors += 1
                    predictions.append(create_prediction_record(
                        example.question, example.answer, f"ERROR: {error}", False
                    ))
                    logger.debug(f"Error on example {i}: {error}")
                
                else:
                    # Check accuracy
                    is_correct = math_accuracy(example, result)
                    if is_correct:
                        correct += 1
                    
                    # Extract predicted answer and reasoning
                    predicted_answer = result.answer if hasattr(result, 'answer') else str(result)
                    reasoning = getattr(result, 'reasoning', '')
                    confidence = getattr(result, 'confidence', None)
                    
                    predictions.append(create_prediction_record(
                        example.question, example.answer, predicted_answer,
                        is_correct, reasoning, confidence
                    ))
                
                progress.update()
            
            except Exception as e:
                errors += 1
                times.append(0)
                logger.error(f"Unexpected error on example {i}: {e}")
                predictions.append(create_prediction_record(
                    example.question, example.answer, f"EXCEPTION: {e}", False
                ))
                progress.update()
        
        # Calculate final metrics
        accuracy = correct / total if total > 0 else 0
        avg_time = np.mean(times) if times else 0
        error_rate = errors / total if total > 0 else 0
        
        logger.info(f" {name}: {correct}/{total} correct ({accuracy*100:.2f}%)")
        logger.info(f" Avg time: {avg_time:.2f}s, Error rate: {error_rate*100:.1f}%")
        
        return BenchmarkResult(
            technique=name,
            accuracy=accuracy,
            correct=correct,
            total=total,
            avg_time=avg_time,
            error_rate=error_rate,
            predictions=predictions
        )
    
    def run_benchmark(self, timeout_seconds: int = 30) -> Dict[str, BenchmarkResult]:
        """
        Run benchmark on all configured modules.
        
        Args:
            timeout_seconds: Timeout for each prediction
            
        Returns:
            Dictionary mapping technique names to results
        """
        
        logger.info("=" * 60)
        logger.info(" GSM8K PROMPTING TECHNIQUES BENCHMARK")
        logger.info("=" * 60)
        logger.info(f" Test set size: {len(self.test_set)} problems")
        logger.info(f" Model: {self.model_config.get('name', 'gpt-3.5-turbo')}")
        
        results = {}
        
        for name, module in self.modules.items():
            try:
                result = self.evaluate_module(module, name, timeout_seconds)
                results[name] = result
            except Exception as e:
                logger.error(f"Failed to evaluate {name}: {e}")
                # Create a failed result
                results[name] = BenchmarkResult(
                    technique=name, accuracy=0.0, correct=0, 
                    total=len(self.test_set), avg_time=0.0, 
                    error_rate=1.0, predictions=[]
                )
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict[str, BenchmarkResult]):
        """Print benchmark summary"""
        
        logger.info("\n" + "=" * 60)
        logger.info(" BENCHMARK SUMMARY")
        logger.info("=" * 60)
        
        # Sort by accuracy
        sorted_results = sorted(results.items(), key=lambda x: x[1].accuracy, reverse=True)
        
        for name, result in sorted_results:
            logger.info(f"{name}: {result.accuracy*100:.2f}% "
                       f"({result.correct}/{result.total}) "
                       f"[{result.avg_time:.2f}s avg]")
        
        if sorted_results:
            best_name, best_result = sorted_results[0]
            logger.info(f"\n Best technique: {best_name} "
                       f"({best_result.accuracy*100:.2f}%)")


def run_ab_test(technique1_name: str, technique2_name: str, 
                dataset: List[dspy.Example], test_size: int = 30,
                alpha: float = 0.05) -> Optional[Dict[str, Any]]:
    """
    Run A/B test between two techniques.
    
    Args:
        technique1_name: Name of first technique
        technique2_name: Name of second technique  
        dataset: Dataset to test on
        test_size: Number of problems per technique
        alpha: Significance level
        
    Returns:
        Dictionary with test results or None if failed
    """
    
    # Map technique names to classes
    technique_map = {
        "zero_shot": ZeroShotModule,
        "few_shot": FewShotModule,
        "cot": CoTModule,
        "self_consistency": SelfConsistencyModule,
        "prolog_style": PrologModule,
    }
    
    if technique1_name not in technique_map:
        logger.error(f"Unknown technique: {technique1_name}")
        return None
    
    if technique2_name not in technique_map:
        logger.error(f"Unknown technique: {technique2_name}")
        return None
    
    # Ensure sufficient data
    if len(dataset) < test_size * 2:
        test_size = len(dataset) // 2
        logger.warning(f"Reduced test size to {test_size} per technique")
    
    if test_size < 5:
        logger.error("Insufficient data for meaningful test")
        return None
    
    # Split dataset randomly
    import random
    random.seed(42)
    shuffled_indices = list(range(len(dataset)))
    random.shuffle(shuffled_indices)
    
    group_a_indices = shuffled_indices[:test_size]
    group_b_indices = shuffled_indices[test_size:test_size*2]
    
    group_a = [dataset[i] for i in group_a_indices]
    group_b = [dataset[i] for i in group_b_indices]
    
    # Initialize techniques
    technique_a = technique_map[technique1_name]()
    technique_b = technique_map[technique2_name]()
    
    # Test technique A
    logger.info(f"Testing {technique1_name}...")
    correct_a = 0
    for example in group_a:
        try:
            pred = technique_a(question=example.question)
            if math_accuracy(example, pred):
                correct_a += 1
        except Exception as e:
            logger.debug(f"Error in technique A: {e}")
    
    # Test technique B  
    logger.info(f"Testing {technique2_name}...")
    correct_b = 0
    for example in group_b:
        try:
            pred = technique_b(question=example.question)
            if math_accuracy(example, pred):
                correct_b += 1
        except Exception as e:
            logger.debug(f"Error in technique B: {e}")
    
    # Calculate results
    accuracy_a = correct_a / len(group_a) if len(group_a) > 0 else 0
    accuracy_b = correct_b / len(group_b) if len(group_b) > 0 else 0
    
    # Statistical test
    try:
        from scipy.stats import fisher_exact
        contingency_table = [
            [correct_a, len(group_a) - correct_a],
            [correct_b, len(group_b) - correct_b]
        ]
        odds_ratio, p_value = fisher_exact(contingency_table)
    except ImportError:
        logger.warning("scipy not available, skipping statistical test")
        p_value = 1.0
    except Exception as e:
        logger.warning(f"Statistical test failed: {e}")
        p_value = 1.0
    
    is_significant = p_value < alpha
    
    result = {
        'technique_a': technique1_name,
        'technique_b': technique2_name,
        'accuracy_a': accuracy_a,
        'accuracy_b': accuracy_b,
        'correct_a': correct_a,
        'correct_b': correct_b,
        'total_a': len(group_a),
        'total_b': len(group_b),
        'p_value': p_value,
        'is_significant': is_significant,
        'alpha': alpha
    }
    
    # Log results
    logger.info(f"\n A/B Test Results:")
    logger.info(f"  {technique1_name}: {accuracy_a*100:.1f}% ({correct_a}/{len(group_a)})")
    logger.info(f"  {technique2_name}: {accuracy_b*100:.1f}% ({correct_b}/{len(group_b)})")
    logger.info(f"  Difference: {(accuracy_b - accuracy_a)*100:+.1f} percentage points")
    logger.info(f"  P-value: {p_value:.4f}")
    
    if is_significant:
        winner = technique2_name if accuracy_b > accuracy_a else technique1_name
        logger.info(f"  Winner: {winner}  (statistically significant)")
    else:
        logger.info(f"  Result: No significant difference ")
    
    return result


if __name__ == "__main__":
    # Test the benchmark
    import logging
    from .data import load_gsm8k_dataset
    
    logging.basicConfig(level=logging.INFO)
    
    # Load small dataset for testing
    dataset = load_gsm8k_dataset(n_samples=5)
    
    # Test benchmark
    benchmark = GSM8KBenchmark(dataset, selected_techniques=["zero_shot", "cot"])
    results = benchmark.run_benchmark()
    
    print(f"Test completed with {len(results)} techniques")
