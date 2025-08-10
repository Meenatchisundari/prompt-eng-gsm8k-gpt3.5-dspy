"""
Unit tests for benchmark functionality.
"""

import pytest
from unittest.mock import Mock, patch
import dspy

from src.gsm8k_bench.benchmark import GSM8KBenchmark, run_ab_test
from src.gsm8k_bench.utils import BenchmarkResult


class TestGSM8KBenchmark:
    """Test GSM8K benchmark functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        # Create mock dataset
        self.mock_dataset = [
            dspy.Example(question="What is 2+2?", answer="4").with_inputs("question"),
            dspy.Example(question="What is 3*4?", answer="12").with_inputs("question"),
        ]
    
    @patch('src.gsm8k_bench.benchmark.setup_dspy_with_config')
    def test_benchmark_init(self, mock_setup):
        """Test benchmark initialization"""
        mock_setup.return_value = True
        
        benchmark = GSM8KBenchmark(
            self.mock_dataset, 
            selected_techniques=["zero_shot", "cot"]
        )
        
        assert len(benchmark.test_set) == 2
        assert len(benchmark.modules) == 2  # Only selected techniques
        mock_setup.assert_called_once()
    
    @patch('src.gsm8k_bench.benchmark.setup_dspy_with_config')
    def test_benchmark_init_all_techniques(self, mock_setup):
        """Test benchmark with all techniques"""
        mock_setup.return_value = True
        
        benchmark = GSM8KBenchmark(self.mock_dataset)
        
        assert len(benchmark.modules) == 5  # All techniques
    
    @patch('src.gsm8k_bench.benchmark.setup_dspy_with_config')
    def test_benchmark_init_dspy_failure(self, mock_setup):
        """Test benchmark initialization with DSPy setup failure"""
        mock_setup.return_value = False
        
        with pytest.raises(RuntimeError, match="Failed to configure DSPy"):
            GSM8KBenchmark(self.mock_dataset)
    
    @patch('src.gsm8k_bench.benchmark.setup_dspy_with_config')
    @patch('src.gsm8k_bench.benchmark.math_accuracy')
    def test_evaluate_module(self, mock_accuracy, mock_setup):
        """Test evaluating a single module"""
        mock_setup.return_value = True
        mock_accuracy.side_effect = [True, False]  # First correct, second wrong
        
        # Create mock module
        mock_module = Mock()
        mock_prediction = Mock()
        mock_prediction.answer = "4"
        mock_module.return_value = mock_prediction
        
        benchmark = GSM8KBenchmark(self.mock_dataset, selected_techniques=["zero_shot"])
        
        result = benchmark.evaluate_module(mock_module, "Test Module")
        
        assert isinstance(result, BenchmarkResult)
        assert result.technique == "Test Module"
        assert result.accuracy == 0.5  # 1 out of 2 correct
        assert result.correct == 1
        assert result.total == 2
        assert len(result.predictions) == 2
    
    @patch('src.gsm8k_bench.benchmark.setup_dspy_with_config')
    def test_evaluate_module_with_timeout(self, mock_setup):
        """Test module evaluation with timeout"""
        mock_setup.return_value = True
        
        # Create mock module that raises exception
        mock_module = Mock()
        mock_module.side_effect = Exception("Timeout")
        
        benchmark = GSM8KBenchmark(self.mock_dataset, selected_techniques=["zero_shot"])
        
        result = benchmark.evaluate_module(mock_module, "Failing Module", timeout_seconds=1)
        
        assert result.error_rate > 0
        assert result.accuracy == 0  # All failed
    
    @patch('src.gsm8k_bench.benchmark.setup_dspy_with_config')
    @patch('src.gsm8k_bench.benchmark.ZeroShotModule')
    @patch('src.gsm8k_bench.benchmark.math_accuracy')
    def test_run_benchmark(self, mock_accuracy, mock_zero_shot_class, mock_setup):
        """Test running full benchmark"""
        mock_setup.return_value = True
        mock_accuracy.return_value = True
        
        # Mock the module class and instance
        mock_module = Mock()
        mock_prediction = Mock()
        mock_prediction.answer = "4"
        mock_module.return_value = mock_prediction
        mock_zero_shot_class.return_value = mock_module
        
        benchmark = GSM8KBenchmark(
            self.mock_dataset, 
            selected_techniques=["zero_shot"]
        )
        
        results = benchmark.run_benchmark()
        
        assert len(results) == 1
        assert "1. Zero-Shot" in results
        assert isinstance(results["1. Zero-Shot"], BenchmarkResult)


class TestABTest:
    """Test A/B testing functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        # Create larger mock dataset for A/B testing
        self.mock_dataset = []
        for i in range(20):
            self.mock_dataset.append(
                dspy.Example(
                    question=f"What is {i}+1?", 
                    answer=str(i+1)
                ).with_inputs("question")
            )
    
    @patch('src.gsm8k_bench.benchmark.math_accuracy')
    @patch('src.gsm8k_bench.benchmark.ZeroShotModule')
    @patch('src.gsm8k_bench.benchmark.CoTModule')
    def test_run_ab_test_success(self, mock_cot_class, mock_zero_shot_class, mock_accuracy):
        """Test successful A/B test"""
        # Mock accuracy to make technique B better
        mock_accuracy.side_effect = [True] * 5 + [False] * 5 + [True] * 8 + [False] * 2
        
        # Mock module instances
        mock_zero_shot = Mock()
        mock_zero_shot.return_value = Mock(answer="1")
        mock_zero_shot_class.return_value = mock_zero_shot
        
        mock_cot = Mock()
        mock_cot.return_value = Mock(answer="1")
        mock_cot_class.return_value = mock_cot
        
        result = run_ab_test(
            "zero_shot", "cot", 
            self.mock_dataset, 
            test_size=10
        )
        
        assert result is not None
        assert 'accuracy_a' in result
        assert 'accuracy_b' in result
        assert 'p_value' in result
        assert result['technique_a'] == "zero_shot"
        assert result['technique_b'] == "cot"
    
    def test_run_ab_test_unknown_technique(self):
        """Test A/B test with unknown technique"""
        result = run_ab_test(
            "unknown_technique", "cot",
            self.mock_dataset,
            test_size=5
        )
        
        assert result is None
    
    def test_run_ab_test_insufficient_data(self):
        """Test A/B test with insufficient data"""
        small_dataset = self.mock_dataset[:8]  # Only 8 samples
        
        result = run_ab_test(
            "zero_shot", "cot",
            small_dataset,
            test_size=10  # Requesting more than available
        )
        
        assert result is None


class TestBenchmarkIntegration:
    """Integration tests for benchmark components"""
    
    @patch('src.gsm8k_bench.benchmark.setup_dspy_with_config')
    def test_benchmark_with_model_config(self, mock_setup):
        """Test benchmark with model configuration"""
        mock_setup.return_value = True
        
        model_config = {
            'name': 'gpt-3.5-turbo',
            'temperature': 0.1,
            'max_tokens': 500
        }
        
        mock_dataset = [
            dspy.Example(question="Test?", answer="1").with_inputs("question")
        ]
        
        benchmark = GSM8KBenchmark(
            mock_dataset,
            selected_techniques=["zero_shot"],
            model_config=model_config
        )
        
        assert benchmark.model_config == model_config
        mock_setup.assert_called_once_with(model_config)
    
    @patch('src.gsm8k_bench.benchmark.setup_dspy_with_config')
    def test_benchmark_error_handling(self, mock_setup):
        """Test benchmark error handling"""
        mock_setup.return_value = True
        
        mock_dataset = [
            dspy.Example(question="Test?", answer="1").with_inputs("question")
        ]
        
        # Test with invalid technique
        benchmark = GSM8KBenchmark(
            mock_dataset,
            selected_techniques=["invalid_technique"]
        )
        
        # Should have no modules for invalid technique
        assert len(benchmark.modules) == 0


if __name__ == "__main__":
    pytest.main([__file__])
