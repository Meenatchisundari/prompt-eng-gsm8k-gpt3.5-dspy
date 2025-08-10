"""
Multi-step verification chain technique.
"""

import dspy


class VerificationChainSolver(dspy.Signature):
    """Solve math problem with multi-step verification process."""
    question = dspy.InputField()
    initial_solution = dspy.OutputField(desc="First attempt at solution")
    verification_check = dspy.OutputField(desc="Check if solution makes sense")
    final_answer = dspy.OutputField(desc="Verified final answer")


class VerifierSolver(dspy.Signature):
    """Verify a proposed solution to a math problem."""
    question = dspy.InputField()
    proposed_answer = dspy.InputField()
    verification = dspy.OutputField(desc="Analysis of whether the answer is correct")
    is_correct = dspy.OutputField(desc="True if correct, False if incorrect")


class VerificationChainModule(dspy.Module):
    """
    Multi-step verification chain module.
    
    This technique solves the problem initially, then runs it through
    a separate verification step that checks the solution for correctness,
    providing an additional layer of error detection.
    """
    
    def __init__(self):
        super().__init__()
        self.solver = dspy.ChainOfThought(VerificationChainSolver)
        self.verifier = dspy.ChainOfThought(VerifierSolver)
    
    def forward(self, question: str):
        """
        Generate prediction using verification chain.
        
        Args:
            question: The math word problem to solve
            
        Returns:
            DSPy prediction with verification information
        """
        # First solution attempt
        solution = self.solver(question=question)

        # Verify the solution
        verification = self.verifier(
            question=question,
            proposed_answer=solution.initial_solution
        )

        return dspy.Prediction(
            answer=solution.final_answer,
            reasoning=solution.initial_solution,
            verification=verification.verification,
            is_verified=verification.is_correct
        )
