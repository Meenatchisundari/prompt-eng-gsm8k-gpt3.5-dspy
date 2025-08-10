"""
Quick fine-tuning implementation for GSM8K.
"""

import torch
import time
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Optional imports for fine-tuning
try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
        DataCollatorForLanguageModeling, pipeline, set_seed
    )
    from datasets import Dataset
    FINETUNING_AVAILABLE = True
except ImportError:
    FINETUNING_AVAILABLE = False
    logger.warning("Fine-tuning dependencies not available")


class QuickFineTuner:
    """
    Quick fine-tuner for demonstration and comparison with prompting.
    Uses DistilGPT2 (82M params) - small enough for most setups.
    """
    
    def __init__(self, model_name: str = "distilgpt2", max_length: int = 512):
        """
        Initialize fine-tuner.
        
        Args:
            model_name: HuggingFace model name
            max_length: Maximum sequence length
        """
        if not FINETUNING_AVAILABLE:
            raise ImportError("Fine-tuning dependencies not installed. "
                            "Install with: pip install transformers torch accelerate")
        
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        
        # Set random seed for reproducibility
        set_seed(42)
    
    def setup(self):
        """Setup model and tokenizer"""
        logger.info(f"🔧 Loading {self.model_name}...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Add special tokens
        special_tokens = {
            "pad_token": "<|pad|>",
            "eos_token": "<|endoftext|>",
        }
        self.tokenizer.add_special_tokens(special_tokens)

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

        # Resize embeddings for new tokens
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Move to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        logger.info(f" Model loaded: {self.model.num_parameters():,} parameters")
    
    def prepare_training_data(self, examples: List, format_type: str = "prolog", 
                            max_samples: int = 100) -> List[str]:
        """
        Prepare training data in specified format.
        
        Args:
            examples: List of DSPy examples
            format_type: Format to use ("prolog", "cot", or "simple")
            max_samples: Maximum number of samples to prepare
            
        Returns:
            List of formatted training texts
        """
        
        def create_training_prompt(question: str, answer: str, format_type: str) -> str:
            if format_type == "prolog":
                return f"""<|startoftext|>Question: {question}

Solution using logical reasoning:
FACTS: Extract the key numerical information and relationships
RULES: Define the mathematical operations needed
QUERY: State what we want to find
DERIVATION: Show step-by-step logical reasoning
ANSWER: {answer}<|endoftext|>"""

            elif format_type == "cot":
                return f"""<|startoftext|>Question: {question}

Let me solve this step by step:
[Work through the problem systematically]
Answer: {answer}<|endoftext|>"""

            else:  # simple
                return f"<|startoftext|>Question: {question}\nAnswer: {answer}<|endoftext|>"

        # Prepare training examples
        training_texts = []
        data_to_use = examples[:max_samples] if len(examples) > max_samples else examples

        for example in data_to_use:
            question = example.question if hasattr(example, 'question') else example['question']
            answer = example.answer if hasattr(example, 'answer') else example['answer']

            # Extract just the number from answer
            answer_num = re.findall(r'-?\d+\.?\d*', str(answer))
            clean_answer = answer_num[-1] if answer_num else str(answer)

            prompt = create_training_prompt(question, clean_answer, format_type)
            training_texts.append(prompt)

        logger.info(f" Prepared {len(training_texts)} training examples")
        return training_texts
    
    def create_dataset(self, training_texts: List[str]) -> Dataset:
        """Create tokenized dataset"""

        def tokenize_function(examples):
            # Tokenize the texts
            tokenized = self.tokenizer(
                examples,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt"
            )

            # For causal LM, labels are the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].clone()

            return tokenized

        # Create dataset
        dataset = Dataset.from_dict({"text": training_texts})

        # Tokenize
        tokenized_dataset = dataset.map(
            lambda examples: tokenize_function(examples["text"]),
            batched=True,
            remove_columns=["text"]
        )

        return tokenized_dataset
    
    def train(self, train_dataset: Dataset, output_dir: str = "./finetuned_model",
              num_epochs: int = 2, batch_size: int = 2, 
              learning_rate: float = 5e-5) -> Any:
        """
        Train the model.
        
        Args:
            train_dataset: Tokenized training dataset
            output_dir: Directory to save model
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            
        Returns:
            Trained model trainer
        """

        # Training arguments - optimized for resource constraints
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,  # Effective batch size = 2*4 = 8
            warmup_steps=50,
            weight_decay=0.01,
            logging_steps=25,
            save_steps=500,
            save_total_limit=1,
            learning_rate=learning_rate,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=0,
            report_to=None,  # Disable wandb
            remove_unused_columns=False,
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM
        )

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )

        logger.info(" Starting training...")
        trainer.train()

        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)

        logger.info(f" Training completed! Model saved to {output_dir}")
        return trainer
    
    def evaluate(self, test_data: List, model_path: Optional[str] = None) -> Tuple[float, List[Dict]]:
        """
        Evaluate the fine-tuned model.
        
        Args:
            test_data: List of test examples
            model_path: Path to fine-tuned model (None to use current model)
            
        Returns:
            Tuple of (accuracy, detailed_results)
        """

        if model_path:
            # Load fine-tuned model
            model = AutoModelForCausalLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            model = self.model
            tokenizer = self.tokenizer

        # Create generation pipeline
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=150,
            temperature=0.1,
            do_sample=False,
            device=0 if torch.cuda.is_available() else -1,
            pad_token_id=tokenizer.eos_token_id
        )

        correct = 0
        total = len(test_data)
        results = []

        logger.info(f"🧪 Evaluating on {total} examples...")

        for i, example in enumerate(tqdm(test_data)):
            question = example.question if hasattr(example, 'question') else example['question']
            expected = example.answer if hasattr(example, 'answer') else example['answer']

            # Create prompt for generation
            prompt = f"Question: {question}\n\nSolution using logical reasoning:\nFACTS:"

            try:
                # Generate response
                outputs = generator(prompt, max_new_tokens=150, num_return_sequences=1)
                generated_text = outputs[0]['generated_text']

                # Extract answer
                predicted_answer = self.extract_answer(generated_text)

                # Check accuracy
                is_correct = self.check_accuracy(expected, predicted_answer)
                if is_correct:
                    correct += 1

                results.append({
                    'question': question,
                    'expected': expected,
                    'predicted': predicted_answer,
                    'generated': generated_text[len(prompt):],  # Only new text
                    'correct': is_correct
                })

            except Exception as e:
                logger.error(f"Error on example {i}: {e}")
                results.append({
                    'question': question,
                    'expected': expected,
                    'predicted': "ERROR",
                    'generated': f"ERROR: {e}",
                    'correct': False
                })

        accuracy = correct / total
        logger.info(f" Fine-tuned model accuracy: {accuracy*100:.2f}% ({correct}/{total})")

        return accuracy, results
    
    def extract_answer(self, text: str) -> str:
        """Extract numerical answer from generated text"""
        # Look for ANSWER: pattern first
        answer_match = re.search(r'ANSWER:\s*([+-]?\d+\.?\d*)', text)
        if answer_match:
            return answer_match.group(1)

        # Fallback: find last number in text
        numbers = re.findall(r'[+-]?\d+\.?\d*', text)
        return numbers[-1] if numbers else "0"
    
    def check_accuracy(self, expected: str, predicted: str) -> bool:
        """Check if answers match"""
        try:
            expected_num = float(str(expected).replace(',', ''))
            predicted_num = float(str(predicted).replace(',', ''))
            return abs(expected_num - predicted_num) < 0.01
        except:
            return False


if __name__ == "__main__":
    # Test the fine-tuner
    if FINETUNING_AVAILABLE:
        tuner = QuickFineTuner("distilgpt2")
        print(" Fine-tuner initialized successfully")
    else:
        print(" Fine-tuning dependencies not available")
