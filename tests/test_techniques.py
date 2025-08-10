"""
Unit tests for prompting techniques.
"""

import pytest
import dspy
from unittest.mock import Mock, patch

from src.gsm8k_bench.techniques import (
    ZeroShotModule, FewShotModule, CoTModule, 
    SelfConsistencyModule, PrologModule
)


class TestBasicTechniques:
    """Test basic prompting technique modules"""
    
    def setup_method(self):
        """Setup for each test"""
        # Mock DSPy configuration
        with patch('dspy.configure'):
            self.mock_lm = Mock()
            dspy.configure(lm=self.mock_lm)
    
    def test_zero_shot_module_init(self):
        """Test ZeroShot module initialization"""
        module = ZeroShotModule()
        assert module is not None
        assert hasattr(module, 'predict')
    
    def test_few_shot_module_init(self):
        """Test FewShot module initialization"""
        module = FewShotModule()
        assert module is not None
        assert hasattr(module, 'predict')
    
    def test_cot_module_init(self):
        """Test CoT module initialization"""
        module = CoTModule()
        assert module is not None
        assert hasattr(module, 'generate')
    
    def test_prolog_module_init(self):
        """Test Prolog module initialization"""
        module = PrologModule()
        assert module is not None
        assert hasattr(module, 'solver')
    
    def test_self_consistency_module_init(self):
        """Test SelfConsistency module initialization"""
        module = SelfConsistencyModule(n_samples=3)
        assert module is not None
        assert module.n_samples == 3
        assert hasattr(module, 'base_predictor')
    
    @patch('src.gsm8k_bench.techniques.zero_shot.dspy.Predict')
    def test_zero_shot_forward(self, mock_predict_class):
        """Test ZeroShot forward method"""
        # Setup mock
        mock_prediction = Mock()
        mock_prediction.answer = "42"
        mock_predict_instance = Mock()
        mock_predict_instance.return_value = mock_prediction
        mock_predict_class.return_value = mock_predict_instance
        
        # Test
        module = ZeroShotModule()
        result = module.forward("What is 21 * 2?")
        
        assert result == mock_prediction
        mock_predict_instance.assert_called_once_with(question="What is 21 * 2?")
    
    @patch('src.gsm8k_bench.techniques.cot.dspy.ChainOfThought')
    def test_cot_forward(self, mock_cot_class):
        """Test CoT forward method"""
        # Setup mock
        mock_prediction = Mock()
        mock_prediction.answer = "42"
        mock_prediction.reasoning = "21 * 2 = 42"
        mock_cot_instance = Mock()
        mock_cot_instance.return_value = mock_prediction
        mock_cot_class.return_value = mock_cot_instance
        
        # Test
        module = CoTModule()
        result = module.forward("What is 21 * 2?")
        
        assert result == mock_prediction
        mock_cot_instance.assert_called_once_with(question="What is 21 * 2?")


class TestSelfConsistency:
    """Test Self-Consistency specific functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        with patch('dspy.configure'):
            self.mock_lm = Mock()
            dspy.configure(lm=self.mock_lm)
    
    @patch('src.gsm8k_bench.techniques.self_consistency.dspy.ChainOfThought')
    @patch('src.gsm8k_bench.techniques.self_consistency.extract_answer')
    def test_self_consistency_majority_voting(self, mock_extract, mock_cot_class):
        """Test majority voting in self-consistency"""
        # Setup mocks
        mock_predictions = [
            Mock(answer="42", reasoning="reasoning 1"),
            Mock(answer="42", reasoning="reasoning 2"), 
            Mock(answer="24", reasoning="reasoning 3"),
        ]
        
        mock_cot_instance = Mock()
        mock_cot_instance.side_effect = mock_predictions
        mock_cot_class.return_value = mock_cot_instance
        
        mock_extract.side_effect = ["42", "42", "24"]
        
        # Test
        module = SelfConsistencyModule(n_samples=3)
        result = module.forward("What is 21 * 2?")
        
        # Check majority voting worked
        assert result.answer == "42"
        assert result.confidence == 2/3  # 2 out of 3 predictions
        assert len(result.all_predictions) == 3


class TestPrologStyle:
    """Test Prolog-style specific functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        with patch('dspy.configure'):
            self.mock_lm = Mock()
            dspy.configure(lm=self.mock_lm)
    
    @patch('src.gsm8k_bench.techniques.prolog_style.dspy.ChainOfThought')
    def test_prolog_forward(self, mock_cot_class):
        """Test Prolog forward method"""
        # Setup mock
        mock_prediction = Mock()
        mock_prediction.facts = "number1(21), number2(2)"
        mock_prediction.rules = "product = number1 × number2"
        mock_prediction.query = "product(?)"
        mock_prediction.derivation = "product = 21 × 2 = 42"
        mock_prediction.answer = "42"
        
        mock_cot_instance = Mock()
        mock_cot_instance.return_value = mock_prediction
        mock_cot_class.return_value = mock_cot_instance
        
        # Test
        module = PrologModule()
        result = module.forward("What is 21 * 2?")
        
        assert result == mock_prediction
        assert hasattr(result, 'facts')
        assert hasattr(result, 'rules')
        assert hasattr(result, 'derivation')
        mock_cot_instance.assert_called_once_with(question="What is 21 * 2?")


if __name__ == "__main__":
    pytest.main([__file__])
