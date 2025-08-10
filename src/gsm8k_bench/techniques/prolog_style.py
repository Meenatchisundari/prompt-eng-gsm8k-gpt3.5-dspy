"""
Prolog-style logical reasoning technique implementation.
"""

import dspy


class PrologSolver(dspy.Signature):
    """Solve this math problem using logical reasoning with facts, rules, and derivation.

    Structure your response exactly as follows:

    FACTS: [List what we know from the problem]
    RULES: [State the mathematical relationships and operations needed]
    QUERY: [What we want to find]
    DERIVATION: [Step-by-step logical reasoning using facts and rules]
    ANSWER: [Final numerical result]

    Example format:
    FACTS: eggs_per_day(16), eats_breakfast(3), uses_for_muffins(4), price_per_egg(2)
    RULES: remaining_eggs = total_eggs - used_eggs, revenue = remaining_eggs × price
    QUERY: daily_revenue(?)
    DERIVATION: used_eggs = 3 + 4 = 7, remaining_eggs = 16 - 7 = 9, revenue = 9 × 2 = 18
    ANSWER: 18"""

    question = dspy.InputField()
    facts = dspy.OutputField(desc="Known facts from the problem")
    rules = dspy.OutputField(desc="Mathematical rules and relationships")
    query = dspy.OutputField(desc="What we want to find")
    derivation = dspy.OutputField(desc="Logical step-by-step reasoning")
    answer = dspy.OutputField(desc="Final numerical answer")


class PrologModule(dspy.Module):
    """
    Prolog-style logical reasoning module.
    
    This technique structures mathematical reasoning like logical programming,
    explicitly separating facts, rules, queries, and derivations. This approach
    often leads to more systematic and accurate problem solving.
    """
    
    def __init__(self):
        super().__init__()
        self.solver = dspy.ChainOfThought(PrologSolver)
    
    def forward(self, question: str):
        """
        Generate prediction using Prolog-style logical reasoning.
        
        Args:
            question: The math word problem to solve
            
        Returns:
            DSPy prediction with facts, rules, query, derivation, and answer
        """
        return self.solver(question=question)
