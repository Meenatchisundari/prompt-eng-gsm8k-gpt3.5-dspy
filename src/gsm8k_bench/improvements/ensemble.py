"""
Weighted ensemble technique combining multiple approaches.
"""

import dspy
from collections import defaultdict
from ..techniques import CoTModule
from ..utils import extract_answer
from .enhanced_prolog import EnhancedPrologModule
from .calculator_augmented import CalculatorAugmentedModule
from .verification_chain import VerificationChainModule


class WeightedEnsembleModule(dspy.Module):
    """
    Weighted ensemble module combining multiple techniques.
    
    This technique runs multiple different prompting approaches and
    combines their results using weighted voting based on each
    technique's historical performance.
    """
    
    def __init__(self, weights: dict = None):
        """
        Initialize weighted ensemble.
        
        Args:
            weights: Dictionary mapping technique names to weights.
                    If None, uses default weights based on benchmark performance.
        """
        super().__init__()

        # Initialize component techniques
        self.modules = {
            'enhanced_prolog': EnhancedPrologModule(),
            'cot': CoTModule(),
            'calculator': CalculatorAugmentedModule(),
            'verification': VerificationChainModule()
        }

        # Default weights based on typical performance
        self.weights = weights or {
            'enhanced_prolog': 0.35,  # Highest weight for best performer
            'cot': 0.25,              # Good performance
            'calculator': 0.25,       # Should be good for calculations
            'verification': 0.15      # Lower weight, but adds verification
        }
    
    def forward(self, question: str):
        """
        Generate prediction using weighted ensemble.
        
        Args:
            question: The math word problem to solve
            
        Returns:
            DSPy prediction with ensemble results
        """
        predictions = {}
        weighted_scores = defaultdict(float)

        # Get predictions from each module
        for name, module in self.modules.items():
            try:
                pred = module(question=question)
                answer = extract_answer(pred.answer)
                confidence = getattr(pred, 'confidence', 1.0)
                
                predictions[name] = {
                    'answer': answer,
                    'reasoning': getattr(pred, 'reasoning', ''),
                    'confidence': confidence
                }
                
                # Weight the vote
                weight = self.weights.get(name, 0.2)
                weighted_scores[answer] += weight * confidence
                
            except Exception as e:
                print(f"Error in {name}: {e}")
                continue

        # Select answer with highest weighted score
        if weighted_scores:
            best_answer = max(weighted_scores.items(), key=lambda x: x[1])[0]
            total_weight = sum(weighted_scores.values())
            confidence = weighted_scores[best_answer] / total_weight if total_weight > 0 else 0
        else:
            best_answer = "0"
            confidence = 0

        # Find reasoning from the technique that gave the winning answer
        selected_reasoning = ""
        for name, pred_data in predictions.items():
            if pred_data['answer'] == best_answer:
                selected_reasoning = pred_data['reasoning']
                break

        return dspy.Prediction(
            answer=best_answer,
            confidence=confidence,
            reasoning=selected_reasoning,
            individual_predictions=predictions,
            weighted_scores=dict(weighted_scores)
        )
