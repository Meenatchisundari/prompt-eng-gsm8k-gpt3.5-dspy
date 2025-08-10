"""
Self-Consistency prompting technique implementation.
"""

import dspy
from collections import Counter
from .cot import CoTSolver
from ..utils import extract_answer


class SelfConsistencyModule(dspy.Module):
    """
    Self-Consistency prompting module - multiple reasoning paths with voting.
    
    This technique generates multiple chain-of-thought reasoning paths with
    some temperature/randomness, then uses majority voting to select the
    most consistent answer. This often improves robustness and accuracy.
    """
    
    def __init__(self, n_samples: int = 5, temperature: float = 0.7):
        """
        Initialize Self-Consistency module.
        
        Args:
            n_samples: Number of reasoning paths to generate
            temperature: Temperature for diverse generation
        """
        super().__init__()
        self.n_samples = n_samples
        self.temperature = temperature
        self.base_predictor = dspy.ChainOfThought(CoTSolver)
    
    def forward(self, question: str):
        """
        Generate prediction using self-consistency with multiple samples.
        
        Args:
            question: The math word problem to solve
            
        Returns:
            DSPy prediction with answer, confidence, and all predictions
        """
        predictions = []
        all_reasonings = []
        
        # Generate multiple predictions with temperature for diversity
        for i in range(self.n_samples):
            try:
                # Create temporary LM with temperature for diversity
                try:
                    temp_lm = dspy.LM(
                        model="gpt-3.5-turbo", 
                        max_tokens=1000, 
                        temperature=self.temperature
                    )
                    with dspy.context(lm=temp_lm):
                        pred = self.base_predictor(question=question)
                except:
                    # Fallback: use base predictor
                    pred = self.base_predictor(question=question)
                
                answer = extract_answer(pred.answer)
                predictions.append(answer)
                all_reasonings.append(
                    pred.reasoning if hasattr(pred, 'reasoning') else ""
                )
                
            except Exception as e:
                # Handle individual prediction failures
                predictions.append("0")
                all_reasonings.append(f"Error: {e}")
        
        # Majority voting
        answer_counts = Counter(predictions)
        most_common = answer_counts.most_common(1)[0]
        confidence = most_common[1] / len(predictions)
        
        # Find reasoning for the most common answer
        selected_reasoning = ""
        for i, pred_answer in enumerate(predictions):
            if pred_answer == most_common[0]:
                selected_reasoning = all_reasonings[i]
                break
        
        return dspy.Prediction(
            answer=most_common[0],
            reasoning=selected_reasoning,
            confidence=confidence,
            all_predictions=predictions
        )
