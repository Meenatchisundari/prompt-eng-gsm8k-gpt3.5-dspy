"""
Chain-of-Thought (CoT) prompting technique implementation.
"""

import dspy


class CoTSolver(dspy.Signature):
    """Solve this math problem step by step, showing your detailed reasoning and calculations."""
    question = dspy.InputField()
    reasoning = dspy.OutputField(desc="Step-by-step solution with calculations")
    answer = dspy.OutputField(desc="The final numerical answer")


class CoTModule(dspy.Module):
    """
    Chain-of-Thought prompting module - step-by-step reasoning.
    
    This technique encourages the model to break down problems into steps
    and show its reasoning process, which often leads to better accuracy
    on complex mathematical reasoning tasks.
    """
    
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(CoTSolver)
    
    def forward(self, question: str):
        """
        Generate prediction using chain-of-thought prompting.
        
        Args:
            question: The math word problem to solve
            
        Returns:
            DSPy prediction with reasoning and answer fields
        """
        return self.generate(question=question)
