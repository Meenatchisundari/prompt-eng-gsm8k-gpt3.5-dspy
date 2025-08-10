"""
Few-shot prompting technique implementation.
"""

import dspy


class FewShotSolver(dspy.Signature):
    """Solve math word problems using these examples:

    Problem: Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes muffins with 4. She sells the remainder at $2 per egg. How much does she make daily?
    Answer: 18

    Problem: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts total?
    Answer: 3

    Problem: Josh buys a house for $80,000, puts in $50,000 repairs, increasing value by 150%. What profit did he make?
    Answer: 70000

    Problem: Sarah has 3 boxes of pencils. Each box contains 12 pencils. She gives away 8 pencils. How many pencils does she have left?
    Answer: 28

    Now solve this problem:"""
    
    question = dspy.InputField()
    answer = dspy.OutputField(desc="The numerical answer")


class FewShotModule(dspy.Module):
    """
    Few-shot prompting module - learning from examples.
    
    This technique provides several solved examples in the prompt to help
    the model learn the expected format and approach for solving problems.
    """
    
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(FewShotSolver)
    
    def forward(self, question: str):
        """
        Generate prediction using few-shot prompting.
        
        Args:
            question: The math word problem to solve
            
        Returns:
            DSPy prediction with answer field
        """
        return self.predict(question=question)
