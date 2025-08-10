"""
GSM8K dataset loading and preprocessing utilities.
"""

import dspy
from datasets import load_dataset
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def load_gsm8k_dataset(n_samples: int = 50, split: str = "test") -> List[dspy.Example]:
    """
    Load GSM8K dataset from HuggingFace.
    
    Args:
        n_samples: Number of samples to load
        split: Dataset split to use ('test' or 'train')
        
    Returns:
        List of DSPy examples with question and answer fields
    """
    
    try:
        # Load from HuggingFace datasets
        logger.info(f"Loading GSM8K dataset from HuggingFace (split: {split})")
        dataset = load_dataset("gsm8k", "main")
        data = dataset[split]
        
        # Convert to list of examples for DSPy
        examples = []
        for i, item in enumerate(data):
            if i >= n_samples:
                break
                
            # Extract numerical answer from the full answer
            answer_text = item["answer"]
            numerical_answer = extract_numerical_answer(answer_text)
            
            example = dspy.Example(
                question=item["question"],
                answer=numerical_answer,
                full_answer=answer_text  # Keep original for reference
            ).with_inputs("question")
            
            examples.append(example)
        
        logger.info(f" Loaded {len(examples)} GSM8K examples")
        return examples
        
    except Exception as e:
        logger.error(f" Error loading dataset: {e}")
        logger.info("Falling back to sample data...")
        return create_sample_gsm8k(n_samples)


def extract_numerical_answer(answer_text: str) -> str:
    """
    Extract the numerical answer from GSM8K answer text.
    GSM8K answers are formatted like: "Step by step solution... #### 42"
    
    Args:
        answer_text: The full answer text from GSM8K
        
    Returns:
        The numerical answer as a string
    """
    
    if "####" in answer_text:
        # Standard GSM8K format
        return answer_text.split("####")[-1].strip()
    else:
        # Fallback: try to find last number
        import re
        numbers = re.findall(r'-?\d+\.?\d*', answer_text)
        return numbers[-1] if numbers else "0"


def create_sample_gsm8k(n_samples: int = 5) -> List[dspy.Example]:
    """
    Create sample GSM8K problems for testing when dataset loading fails.
    
    Args:
        n_samples: Number of sample problems to create
        
    Returns:
        List of sample DSPy examples
    """
    
    sample_problems = [
        {
            "question": "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes muffins for her friends every day with 4. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
            "answer": "18"
        },
        {
            "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
            "answer": "3"
        },
        {
            "question": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
            "answer": "70000"
        },
        {
            "question": "Sarah has 3 boxes of pencils. Each box contains 12 pencils. She gives away 8 pencils. How many pencils does she have left?",
            "answer": "28"
        },
        {
            "question": "A store sells pencils for $0.25 each. If Maria buys 8 pencils and pays with a $5 bill, how much change does she receive?",
            "answer": "3.00"
        },
        {
            "question": "Tom has 5 bags of marbles. Each bag contains 8 marbles. He gives 3 marbles to his sister. How many marbles does Tom have left?",
            "answer": "37"
        },
        {
            "question": "A pizza is cut into 8 equal slices. If 3 people each eat 2 slices, how many slices are left?",
            "answer": "2"
        },
        {
            "question": "Lisa earns $15 per hour. If she works 6 hours a day for 5 days, how much money does she earn in total?",
            "answer": "450"
        },
        {
            "question": "A book has 240 pages. If Mark reads 30 pages per day, how many days will it take him to finish the book?",
            "answer": "8"
        },
        {
            "question": "A car travels 60 miles in 2 hours. What is the car's speed in miles per hour?",
            "answer": "30"
        }
    ]
    
    examples = []
    for i, item in enumerate(sample_problems):
        if i >= n_samples:
            break
            
        example = dspy.Example(
            question=item["question"],
            answer=item["answer"]
        ).with_inputs("question")
        
        examples.append(example)
    
    logger.info(f" Created {len(examples)} sample problems")
    return examples


def create_training_split(examples: List[dspy.Example], 
                         train_ratio: float = 0.8) -> tuple[List[dspy.Example], List[dspy.Example]]:
    """
    Split examples into training and test sets.
    
    Args:
        examples: List of DSPy examples
        train_ratio: Ratio of examples to use for training
        
    Returns:
        Tuple of (train_examples, test_examples)
    """
    
    import random
    random.seed(42)  # For reproducibility
    
    shuffled = examples.copy()
    random.shuffle(shuffled)
    
    split_idx = int(len(shuffled) * train_ratio)
    train_examples = shuffled[:split_idx]
    test_examples = shuffled[split_idx:]
    
    logger.info(f"Split data: {len(train_examples)} train, {len(test_examples)} test")
    
    return train_examples, test_examples


def get_problem_categories() -> Dict[str, List[str]]:
    """
    Define categories of math problems for analysis.
    
    Returns:
        Dictionary mapping category names to keyword lists
    """
    
    return {
        "arithmetic": ["add", "subtract", "multiply", "divide", "total", "sum"],
        "money": ["dollar", "cost", "price", "pay", "earn", "profit", "change"],
        "time": ["hour", "day", "week", "month", "year", "minute"],
        "geometry": ["area", "perimeter", "length", "width", "height", "circle"],
        "fractions": ["half", "third", "quarter", "fraction", "part"],
        "percentages": ["percent", "%", "percentage", "rate"],
        "word_problems": ["people", "students", "children", "friends", "family"],
    }


def categorize_problem(question: str) -> List[str]:
    """
    Categorize a problem based on keywords in the question.
    
    Args:
        question: The problem question text
        
    Returns:
        List of categories that match the problem
    """
    
    categories = get_problem_categories()
    question_lower = question.lower()
    
    matched_categories = []
    for category, keywords in categories.items():
        if any(keyword in question_lower for keyword in keywords):
            matched_categories.append(category)
    
    return matched_categories if matched_categories else ["general"]


def analyze_dataset_distribution(examples: List[dspy.Example]) -> Dict[str, Any]:
    """
    Analyze the distribution of problem types in the dataset.
    
    Args:
        examples: List of DSPy examples
        
    Returns:
        Dictionary with analysis results
    """
    
    from collections import Counter
    
    # Categorize all problems
    all_categories = []
    for example in examples:
        categories = categorize_problem(example.question)
        all_categories.extend(categories)
    
    category_counts = Counter(all_categories)
    
    # Calculate statistics
    question_lengths = [len(example.question.split()) for example in examples]
    
    analysis = {
        "total_problems": len(examples),
        "category_distribution": dict(category_counts),
        "avg_question_length": sum(question_lengths) / len(question_lengths),
        "min_question_length": min(question_lengths),
        "max_question_length": max(question_lengths),
    }
    
    return analysis


if __name__ == "__main__":
    # Test the data loading
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Test loading dataset
    examples = load_gsm8k_dataset(n_samples=10)
    print(f"Loaded {len(examples)} examples")
    
    # Test analysis
    analysis = analyze_dataset_distribution(examples)
    print(f"Dataset analysis: {analysis}")
    
    # Show first example
    if examples:
        print(f"\nExample problem:")
        print(f"Question: {examples[0].question}")
        print(f"Answer: {examples[0].answer}")
        print(f"Categories: {categorize_problem(examples[0].question)}")
