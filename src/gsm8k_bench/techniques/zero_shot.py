"""
Zero-shot prompting technique implementation.
"""

import dspy


class ZeroShotSolver(dspy.Signature):
    """Solve this math word problem and provide only the numerical answer."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="The numerical answer")


class ZeroShotModule(dspy.Module):
    """
    Zero-shot prompting module - direct problem solving without examples.
    
    This is the baseline technique that asks the model to solve problems
    directly without any examples or special reasoning instructions.
    """
    
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(ZeroShotSolver)
    
    def forward(self, question: str):
        """
        Generate prediction using zero-shot prompting.
        
        Args:
            question: The math word problem to solve
            
        Returns:
            DSPy prediction with answer field
        """
        return self.predict(question=question)
