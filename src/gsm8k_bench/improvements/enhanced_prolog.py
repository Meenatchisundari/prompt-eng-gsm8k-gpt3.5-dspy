"""
Enhanced Prolog-style reasoning with verification steps.
"""

import dspy


class EnhancedPrologSolver(dspy.Signature):
    """Solve this math problem using enhanced logical reasoning with verification.

    Use this exact structure:

    FACTS: [Extract all numerical values and relationships]
    RULES: [Define mathematical operations needed]
    QUERY: [State what we want to find]
    DERIVATION: [Step-by-step calculation with intermediate checks]
    VERIFICATION: [Double-check calculation using different approach]
    ANSWER: [Final numerical result]

    Example:
    FACTS: eggs_per_day(16), eats_breakfast(3), uses_for_muffins(4), price_per_egg(2)
    RULES: remaining = total - used, revenue = remaining × price
    QUERY: daily_revenue(?)
    DERIVATION: used = 3 + 4 = 7, remaining = 16 - 7 = 9, revenue = 9 × 2 = 18
    VERIFICATION: Check: 16 eggs - 3 - 4 = 9 eggs, 9 × $2 = $18 ✓
    ANSWER: 18"""

    question = dspy.InputField()
    facts = dspy.OutputField(desc="All known facts from the problem")
    rules = dspy.OutputField(desc="Mathematical rules and operations")
    query = dspy.OutputField(desc="What we want to find")
    derivation = dspy.OutputField(desc="Step-by-step logical calculation")
    verification = dspy.OutputField(desc="Double-check using alternative method")
    answer = dspy.OutputField(desc="Final numerical answer")


class EnhancedPrologModule(dspy.Module):
    """
    Enhanced Prolog-style reasoning with built-in verification.
    
    Extends the basic Prolog approach by adding an explicit verification
    step that double-checks the calculation using a different method or
    approach, improving accuracy and catching errors.
    """
    
    def __init__(self):
        super().__init__()
        self.solver = dspy.ChainOfThought(EnhancedPrologSolver)
    
    def forward(self, question: str):
        """
        Generate prediction using enhanced Prolog-style reasoning.
        
        Args:
            question: The math word problem to solve
            
        Returns:
            DSPy prediction with all reasoning components and verification
        """
        return self.solver(question=question)
