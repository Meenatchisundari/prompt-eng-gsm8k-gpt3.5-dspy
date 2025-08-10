"""
Calculator-augmented reasoning technique.
"""

import dspy
import re
import math


class CalculatorModule:
    """Module that can perform exact calculations"""

    def safe_eval(self, expression: str):
        """Safely evaluate mathematical expressions"""
        try:
            # Only allow safe mathematical operations
            allowed_names = {
                k: v for k, v in math.__dict__.items()
                if not k.startswith("__")
            }
            allowed_names.update({
                'abs': abs, 'round': round, 'min': min, 'max': max,
                'sum': sum, 'pow': pow
            })

            # Remove spaces and validate expression
            expr = expression.replace(' ', '')

            # Simple validation - only allow numbers, operators, parentheses
            if re.match(r'^[0-9+\-*/().]+$', expr):
                return eval(expr)
            else:
                return None
        except:
            return None

    def extract_and_calculate(self, text: str) -> str:
        """Extract calculations from text and compute them"""
        # Find expressions like "3 + 4", "16 - 7", etc.
        calc_patterns = [
            r'(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*×\s*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*÷\s*(\d+(?:\.\d+)?)'
        ]

        for pattern in calc_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) == 3:  # Binary operation
                    num1, op, num2 = match
                    if op == '×': op = '*'
                    elif op == '÷': op = '/'

                    expr = f"{num1}{op}{num2}"
                    result = self.safe_eval(expr)
                    if result is not None:
                        # Replace the calculation with the result
                        original = f"{num1} {op} {num2}"
                        text = text.replace(original, str(result))
                elif len(match) == 2:  # Special symbols
                    num1, num2 = match
                    # Handle × and ÷
                    if '×' in text:
                        result = self.safe_eval(f"{num1}*{num2}")
                        text = text.replace(f"{num1} × {num2}", str(result))
                    elif '÷' in text:
                        result = self.safe_eval(f"{num1}/{num2}")
                        text = text.replace(f"{num1} ÷ {num2}", str(result))

        return text


class CalculatorAugmentedSolver(dspy.Signature):
    """Solve math problems with explicit calculation steps that will be computed exactly."""
    question = dspy.InputField()
    reasoning = dspy.OutputField(desc="Step-by-step reasoning with explicit calculations like '3 + 4' or '16 × 2'")
    answer = dspy.OutputField(desc="Final numerical answer")


class CalculatorAugmentedModule(dspy.Module):
    """
    Calculator-augmented reasoning module.
    
    This technique encourages the model to write explicit mathematical
    expressions that are then computed exactly by a calculator module,
    reducing arithmetic errors.
    """
    
    def __init__(self):
        super().__init__()
        self.solver = dspy.ChainOfThought(CalculatorAugmentedSolver)
        self.calculator = CalculatorModule()
    
    def forward(self, question: str):
        """
        Generate prediction using calculator-augmented reasoning.
        
        Args:
            question: The math word problem to solve
            
        Returns:
            DSPy prediction with calculator-processed reasoning
        """
        solution = self.solver(question=question)

        # Process reasoning with calculator
        enhanced_reasoning = self.calculator.extract_and_calculate(solution.reasoning)

        return dspy.Prediction(
            answer=solution.answer,
            reasoning=enhanced_reasoning,
            original_reasoning=solution.reasoning
        )
