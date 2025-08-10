"""
Utility functions for answer extraction, validation, and evaluation.
"""

import re
import time
import dspy
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from collections import Counter
import logging

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    technique: str
    accuracy: float
    correct: int
    total: int
    avg_time: float
    error_rate: float
    predictions: Optional[List[Dict]] = None
    confidence: Optional[float] = None


def extract_answer(text: str) -> str:
    """
    Extract numerical answer from text using multiple strategies.
    
    Args:
        text: Text containing the answer
        
    Returns:
        Extracted numerical answer as string
    """
    
    if not text:
        return "0"
    
    # Clean the text
    text = str(text).replace(',', '').replace('$', '')
    
    # Strategy 1: Look for explicit answer patterns
    answer_patterns = [
        r'(?:answer|result|solution)[\s:=]+([+-]?\d+\.?\d*)',
        r'([+-]?\d+\.?\d*)[\s]*(?:is the answer|is the result)',
        r'=[\s]*([+-]?\d+\.?\d*)',
        r'ANSWER:\s*([+-]?\d+\.?\d*)',  # For Prolog-style
        r'([+-]?\d+\.?\d*)$'  # Number at the end
    ]
    
    for pattern in answer_patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            return matches[-1]
    
    # Strategy 2: Find all numbers and return the last one
    numbers = re.findall(r'[+-]?\d+\.?\d*', text)
    return numbers[-1] if numbers else "0"


def math_accuracy(example: dspy.Example, prediction: Any, 
                 tolerance: float = 0.01) -> bool:
    """
    Evaluate if prediction matches expected answer within tolerance.
    
    Args:
        example: DSPy example with expected answer
        prediction: Model prediction (can have .answer attribute or be string)
        tolerance: Numerical tolerance for comparison
        
    Returns:
        True if prediction is correct, False otherwise
    """
    
    try:
        # Extract predicted answer
        if hasattr(prediction, 'answer'):
            predicted = float(extract_answer(prediction.answer))
        else:
            predicted = float(extract_answer(str(prediction)))
        
        # Extract expected answer
        expected = float(extract_answer(example.answer))
        
        # Compare with tolerance
        return abs(predicted - expected) < tolerance
        
    except (ValueError, TypeError, AttributeError) as e:
        logger.debug(f"Error in math_accuracy: {e}")
        return False


def safe_execute_with_timeout(func, timeout_seconds: int = 30, *args, **kwargs):
    """
    Execute function with timeout protection.
    
    Args:
        func: Function to execute
        timeout_seconds: Maximum execution time
        *args, **kwargs: Arguments for the function
        
    Returns:
        Tuple of (result, execution_time, error)
    """
    
    import signal
    
    class TimeoutException(Exception):
        pass
    
    def timeout_handler(signum, frame):
        raise TimeoutException("Function execution timed out")
    
    start_time = time.time()
    result = None
    error = None
    
    try:
        # Set timeout (only works on Unix systems)
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
        
        result = func(*args, **kwargs)
        
    except TimeoutException:
        error = f"Timeout after {timeout_seconds} seconds"
        logger.warning(f"Function timed out: {func.__name__}")
        
    except Exception as e:
        error = str(e)
        logger.error(f"Error in {func.__name__}: {e}")
        
    finally:
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)  # Cancel alarm
    
    execution_time = time.time() - start_time
    return result, execution_time, error


def validate_prediction(prediction: Any) -> bool:
    """
    Validate that a prediction is properly formatted.
    
    Args:
        prediction: Model prediction to validate
        
    Returns:
        True if valid, False otherwise
    """
    
    if prediction is None:
        return False
    
    # Check if prediction has required attributes
    if hasattr(prediction, 'answer'):
        answer = prediction.answer
    else:
        answer = str(prediction)
    
    # Check if answer can be parsed as number
    try:
        float(extract_answer(answer))
        return True
    except (ValueError, TypeError):
        return False


def calculate_confidence_interval(successes: int, total: int, 
                                confidence_level: float = 0.95) -> tuple[float, float]:
    """
    Calculate binomial confidence interval for accuracy.
    
    Args:
        successes: Number of correct predictions
        total: Total number of predictions
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    
    if total == 0:
        return 0.0, 0.0
    
    from scipy.stats import binom
    
    alpha = 1 - confidence_level
    lower = binom.ppf(alpha/2, total, successes/total) / total
    upper = binom.ppf(1 - alpha/2, total, successes/total) / total
    
    # Handle edge cases
    lower = max(0.0, lower)
    upper = min(1.0, upper)
    
    return lower, upper


def format_time(seconds: float) -> str:
    """
    Format time duration in human-readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes:.0f}m {secs:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"


def get_error_analysis(predictions: List[Dict]) -> Dict[str, Any]:
    """
    Analyze prediction errors to identify patterns.
    
    Args:
        predictions: List of prediction dictionaries
        
    Returns:
        Dictionary with error analysis
    """
    
    errors = [p for p in predictions if not p.get('correct', False)]
    
    if not errors:
        return {"total_errors": 0, "error_patterns": {}}
    
    error_patterns = {
        'calculation_errors': [],
        'reading_comprehension': [],
        'format_issues': [],
        'large_deviations': []
    }
    
    for error in errors:
        try:
            expected = float(extract_answer(error['expected']))
            predicted = float(extract_answer(error['predicted']))
            
            relative_error = abs(expected - predicted) / max(abs(expected), 1)
            
            if relative_error < 0.1:  # Small error, likely calculation
                error_patterns['calculation_errors'].append(error)
            elif relative_error > 2.0:  # Large error, likely comprehension
                error_patterns['reading_comprehension'].append(error)
            else:
                error_patterns['large_deviations'].append(error)
                
        except (ValueError, TypeError):
            error_patterns['format_issues'].append(error)
    
    analysis = {
        "total_errors": len(errors),
        "error_rate": len(errors) / len(predictions) if predictions else 0,
        "error_patterns": {k: len(v) for k, v in error_patterns.items()},
        "common_issues": []
    }
    
    # Identify most common error type
    if error_patterns['calculation_errors']:
        analysis['common_issues'].append("Calculation errors")
    if error_patterns['reading_comprehension']:
        analysis['common_issues'].append("Reading comprehension issues")
    if error_patterns['format_issues']:
        analysis['common_issues'].append("Answer format problems")
    
    return analysis


def setup_dspy_with_config(model_config: Dict[str, Any]) -> bool:
    """
    Setup DSPy with the given model configuration.
    
    Args:
        model_config: Dictionary with model configuration
        
    Returns:
        True if setup successful, False otherwise
    """
    
    try:
        model_name = model_config.get('name', 'gpt-3.5-turbo')
        max_tokens = model_config.get('max_tokens', 1000)
        temperature = model_config.get('temperature', 0.0)
        
        # Try different DSPy configuration methods
        try:
            # Method 1: New DSPy API
            lm = dspy.LM(
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperature
            )
            dspy.configure(lm=lm)
            logger.info(" DSPy configured with LM class")
            return True
            
        except Exception:
            # Method 2: OpenAI class
            try:
                lm = dspy.OpenAI(
                    model=model_name,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                dspy.configure(lm=lm)
                logger.info(" DSPy configured with OpenAI class")
                return True
                
            except Exception:
                # Method 3: Legacy settings
                try:
                    lm = dspy.OpenAI(model=model_name)
                    dspy.settings.configure(lm=lm)
                    logger.info(" DSPy configured with legacy settings")
                    return True
                    
                except Exception as e:
                    logger.error(f" All DSPy configuration methods failed: {e}")
                    return False
    
    except Exception as e:
        logger.error(f" Error in setup_dspy_with_config: {e}")
        return False


def create_prediction_record(question: str, expected: str, predicted: str, 
                           is_correct: bool, reasoning: str = "", 
                           confidence: float = None) -> Dict[str, Any]:
    """
    Create a standardized prediction record.
    
    Args:
        question: The input question
        expected: Expected answer
        predicted: Predicted answer
        is_correct: Whether prediction is correct
        reasoning: Reasoning/explanation text
        confidence: Confidence score if available
        
    Returns:
        Dictionary with prediction record
    """
    
    record = {
        'question': question,
        'expected': expected,
        'predicted': predicted,
        'correct': is_correct,
        'reasoning': reasoning,
        'question_length': len(question.split()),
        'timestamp': time.time()
    }
    
    if confidence is not None:
        record['confidence'] = confidence
    
    return record


class ProgressTracker:
    """Simple progress tracker for benchmark runs"""
    
    def __init__(self, total: int, name: str = "Progress"):
        self.total = total
        self.current = 0
        self.name = name
        self.start_time = time.time()
    
    def update(self, increment: int = 1):
        """Update progress"""
        self.current += increment
        
        if self.current % max(1, self.total // 10) == 0 or self.current == self.total:
            self._print_progress()
    
    def _print_progress(self):
        """Print current progress"""
        elapsed = time.time() - self.start_time
        progress_pct = (self.current / self.total) * 100
        
        if self.current > 0:
            eta = (elapsed / self.current) * (self.total - self.current)
            eta_str = format_time(eta)
        else:
            eta_str = "Unknown"
        
        print(f"  {self.name}: {self.current}/{self.total} "
              f"({progress_pct:.1f}%) - ETA: {eta_str}")


if __name__ == "__main__":
    # Test utilities
    
    # Test answer extraction
    test_texts = [
        "The answer is 42",
        "ANSWER: 123",
        "After calculating: 56.7",
        "= 89",
        "Solution shows 234"
    ]
    
    print("Testing answer extraction:")
    for text in test_texts:
        answer = extract_answer(text)
        print(f"  '{text}' -> '{answer}'")
    
    # Test accuracy calculation
    class MockExample:
        def __init__(self, answer):
            self.answer = answer
    
    class MockPrediction:
        def __init__(self, answer):
            self.answer = answer
    
    example = MockExample("42")
    correct_pred = MockPrediction("42.0")
    wrong_pred = MockPrediction("24")
    
    print(f"\nTesting accuracy:")
    print(f"  Correct: {math_accuracy(example, correct_pred)}")
    print(f"  Wrong: {math_accuracy(example, wrong_pred)}")
