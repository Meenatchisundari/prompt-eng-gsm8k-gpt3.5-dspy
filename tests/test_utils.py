"""
Unit tests for utility functions.
"""

import pytest
from unittest.mock import Mock

from src.gsm8k_bench.utils import (
    extract_answer, math_accuracy, BenchmarkResult,
    validate_prediction, format_time, get_error_analysis
)


class TestAnswerExtraction:
    """Test answer extraction functions"""
    
    def test_extract_answer_simple_number(self):
        """Test extracting simple numbers"""
        assert extract_answer("42") == "42"
        assert extract_answer("42.5") == "42.5"
        assert extract_answer("-15") == "-15"
    
    def test_extract_answer_with_text(self):
        """Test extracting answers from text"""
        assert extract_answer("The answer is 42") == "42"
        assert extract_answer("Answer: 123") == "123"
        assert extract_answer("ANSWER: 456") == "456"
        assert extract_answer("= 789") == "789"
    
    def test_extract_answer_multiple_numbers(self):
        """Test extracting from text with multiple numbers"""
        text = "First we have 10, then we add 5, so the answer is 15"
        assert extract_answer(text) == "15"
    
    def test_extract_answer_no_numbers(self):
        """Test with no numbers present"""
        assert extract_answer("No numbers here") == "0"
        assert extract_answer("") == "0"
        assert extract_answer(None) == "0"
    
    def test_extract_answer_with_commas_dollars(self):
        """Test with formatted numbers"""
        assert extract_answer("$1,234") == "1234"
        assert extract_answer("The cost is $5,678.90") == "5678.90"


class TestMathAccuracy:
    """Test mathematical accuracy checking"""
    
    def test_math_accuracy_exact_match(self):
        """Test exact number matches"""
        example = Mock()
        example.answer = "42"
        
        prediction = Mock()
        prediction.answer = "42"
        
        assert math_accuracy(example, prediction) == True
    
    def test_math_accuracy_tolerance(self):
        """Test matches within tolerance"""
        example = Mock()
        example.answer = "42.0"
        
        prediction = Mock()
        prediction.answer = "42.005"  # Within 0.01 tolerance
        
        assert math_accuracy(example, prediction) == True
        
        prediction.answer = "42.02"  # Outside tolerance
        assert math_accuracy(example, prediction) == False
    
    def test_math_accuracy_string_prediction(self):
        """Test with string prediction (no .answer attribute)"""
        example = Mock()
        example.answer = "42"
        
        # Test with string prediction
        assert math_accuracy(example, "42") == True
        assert math_accuracy(example, "41") == False
    
    def test_math_accuracy_invalid_inputs(self):
        """Test with invalid inputs"""
        example = Mock()
        example.answer = "42"
        
        prediction = Mock()
        prediction.answer = "not a number"
        
        assert math_accuracy(example, prediction) == False


class TestBenchmarkResult:
    """Test BenchmarkResult dataclass"""
    
    def test_benchmark_result_creation(self):
        """Test creating BenchmarkResult"""
        result = BenchmarkResult(
            technique="Test",
            accuracy=0.75,
            correct=75,
            total=100,
            avg_time=2.5,
            error_rate=0.05
        )
        
        assert result.technique == "Test"
        assert result.accuracy == 0.75
        assert result.correct == 75
        assert result.total == 100
        assert result.avg_time == 2.5
        assert result.error_rate == 0.05


class TestValidation:
    """Test validation functions"""
    
    def test_validate_prediction_valid(self):
        """Test validating valid predictions"""
        prediction = Mock()
        prediction.answer = "42"
        assert validate_prediction(prediction) == True
        
        # Test string prediction
        assert validate_prediction("42") == True
    
    def test_validate_prediction_invalid(self):
        """Test validating invalid predictions"""
        prediction = Mock()
        prediction.answer = "not a number"
        assert validate_prediction(prediction) == False
        
        # Test None
        assert validate_prediction(None) == False


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_format_time(self):
        """Test time formatting"""
        assert format_time(0.5) == "500ms"
        assert format_time(1.5) == "1.5s"
        assert format_time(65) == "1m 5s"
        assert format_time(3665) == "1h 1m"
    
    def test_get_error_analysis_empty(self):
        """Test error analysis with no errors"""
        predictions = [
            {'correct': True, 'expected': '42', 'predicted': '42'},
            {'correct': True, 'expected': '24', 'predicted': '24'},
        ]
        
        analysis = get_error_analysis(predictions)
        assert analysis['total_errors'] == 0
        assert analysis['error_rate'] == 0.0
    
    def test_get_error_analysis_with_errors(self):
        """Test error analysis with errors"""
        predictions = [
            {'correct': True, 'expected': '42', 'predicted': '42'},
            {'correct': False, 'expected': '24', 'predicted': '25'},  # Small error
            {'correct': False, 'expected': '10', 'predicted': 'error'},  # Format error
        ]
        
        analysis = get_error_analysis(predictions)
        assert analysis['total_errors'] == 2
        assert analysis['error_rate'] == 2/3
        assert 'error_patterns' in analysis


if __name__ == "__main__":
    pytest.main([__file__])
