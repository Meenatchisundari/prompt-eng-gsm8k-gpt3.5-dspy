# API Reference

Complete reference for the GSM8K Benchmark API.

## Core Classes

### GSM8KBenchmark

The main benchmark class for evaluating prompting techniques.

```python
class GSM8KBenchmark:
    """Main benchmark runner for GSM8K prompting techniques evaluation."""
    
    def __init__(self, 
                 test_dataset: List[dspy.Example],
                 selected_techniques: Optional[List[str]] = None,
                 model_config: Optional[Dict[str, Any]] = None,
                 parallel_workers: int = 1,
                 timeout_seconds: int = 30):
        """
        Initialize the benchmark.
        
        Args:
            test_dataset: List of DSPy examples to evaluate
            selected_techniques: List of technique names to run (None for all)
            model_config: Model configuration dictionary
            parallel_workers: Number of parallel workers for evaluation
            timeout_seconds: Timeout for each prediction
        """
```

#### Methods

##### `run_benchmark()`
```python
def run_benchmark(self, save_predictions: bool = True) -> Dict[str, BenchmarkResult]:
    """
    Run the complete benchmark evaluation.
    
    Args:
        save_predictions: Whether to save detailed predictions
        
    Returns:
        Dictionary mapping technique names to BenchmarkResult objects
        
    Example:
        >>> benchmark = GSM8KBenchmark(dataset)
        >>> results = benchmark.run_benchmark()
        >>> print(f"Best accuracy: {max(r.accuracy for r in results.values())}")
    """
```

##### `evaluate_module()`
```python
def evaluate_module(self, 
                   module: dspy.Module, 
                   name: str,
                   timeout_seconds: int = 30) -> BenchmarkResult:
    """
    Evaluate a single technique module.
    
    Args:
        module: DSPy module implementing the technique
        name: Display name for the technique
        timeout_seconds: Timeout for each prediction
        
    Returns:
        BenchmarkResult with evaluation metrics
    """
```

##### `add_technique()`
```python
def add_technique(self, name: str, module: dspy.Module) -> None:
    """
    Add a custom technique to the benchmark.
    
    Args:
        name: Name for the technique
        module: DSPy module implementing the technique
        
    Example:
        >>> benchmark.add_technique("my_custom", MyCustomTechnique())
    """
```

### BenchmarkResult

Container for benchmark evaluation results.

```python
@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    
    technique: str
    accuracy: float
    correct: int
    total: int
    avg_time: float
    error_rate: float
    predictions: Optional[List[Dict]] = None
    confidence: Optional[float] = None
```

## Data Loading Functions

### `load_gsm8k_dataset()`
```python
def load_gsm8k_dataset(n_samples: int = 50, 
                      split: str = "test") -> List[dspy.Example]:
    """
    Load GSM8K dataset from HuggingFace.
    
    Args:
        n_samples: Number of samples to load
        split: Dataset split ('test' or 'train')
        
    Returns:
        List of DSPy examples
        
    Example:
        >>> dataset = load_gsm8k_dataset(n_samples=100)
        >>> print(f"Loaded {len(dataset)} problems")
    """
```

### `create_training_split()`
```python
def create_training_split(examples: List[dspy.Example], 
                         train_ratio: float = 0.8) -> Tuple[List[dspy.Example], List[dspy.Example]]:
    """
    Split examples into training and test sets.
    
    Args:
        examples: List of DSPy examples
        train_ratio: Ratio for training split
        
    Returns:
        Tuple of (train_examples, test_examples)
    """
```

## Utility Functions

### Answer Extraction

#### `extract_answer()`
```python
def extract_answer(text: str) -> str:
    """
    Extract numerical answer from text.
    
    Args:
        text: Text containing the answer
        
    Returns:
        Extracted numerical answer as string
        
    Example:
        >>> extract_answer("The answer is 42")
        "42"
        >>> extract_answer("Result: $1,234.56")
        "1234.56"
    """
```

#### `math_accuracy()`
```python
def math_accuracy(example: dspy.Example, 
                 prediction: Any, 
                 tolerance: float = 0.01) -> bool:
    """
    Check if prediction matches expected answer.
    
    Args:
        example: DSPy example with expected answer
        prediction: Model prediction
        tolerance: Numerical tolerance for comparison
        
    Returns:
        True if prediction is correct
        
    Example:
        >>> example = dspy.Example(answer="42")
        >>> pred = dspy.Prediction(answer="42.005")
        >>> math_accuracy(example, pred)
        True
    """
```

### Statistical Functions

#### `calculate_confidence_interval()`
```python
def calculate_confidence_interval(successes: int, 
                                total: int, 
                                confidence_level: float = 0.95) -> Tuple[float, float]:
    """
    Calculate binomial confidence interval.
    
    Args:
        successes: Number of correct predictions
        total: Total predictions
        confidence_level: Confidence level (e.g., 0.95)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
```

## Visualization Functions

### `create_results_table()`
```python
def create_results_table(results: Dict[str, BenchmarkResult]) -> pd.DataFrame:
    """
    Create formatted results table.
    
    Args:
        results: Dictionary of benchmark results
        
    Returns:
        Pandas DataFrame with formatted results
        
    Example:
        >>> table = create_results_table(results)
        >>> print(table.to_string(index=False))
    """
```

### `plot_results()`
```python
def plot_results(results: Dict[str, BenchmarkResult], 
                save_path: Optional[str] = None) -> None:
    """
    Create comprehensive result visualizations.
    
    Args:
        results: Dictionary of benchmark results
        save_path: Optional path to save plots
        
    Example:
        >>> plot_results(results, "benchmark_plots.png")
    """
```

### `create_performance_heatmap()`
```python
def create_performance_heatmap(results: Dict[str, BenchmarkResult], 
                              save_path: Optional[str] = None) -> None:
    """
    Create performance heatmap across metrics.
    
    Args:
        results: Dictionary of benchmark results
        save_path: Optional path to save heatmap
    """
```

## Technique Classes

### Base Classes

#### `BaseTechnique`
```python
class BaseTechnique(dspy.Module):
    """Base class for all prompting techniques."""
    
    def __init__(self, **kwargs):
        super().__init__()
        
    def forward(self, question: str) -> dspy.Prediction:
        """
        Generate prediction for given question.
        
        Args:
            question: Math problem to solve
            
        Returns:
            DSPy prediction with answer
        """
        raise NotImplementedError
```

### Core Techniques

#### `ZeroShotModule`
```python
class ZeroShotModule(dspy.Module):
    """Zero-shot prompting technique."""
    
    def forward(self, question: str) -> dspy.Prediction:
        """Generate zero-shot prediction."""
```

#### `FewShotModule`
```python
class FewShotModule(dspy.Module):
    """Few-shot prompting with examples."""
    
    def forward(self, question: str) -> dspy.Prediction:
        """Generate few-shot prediction with examples."""
```

#### `CoTModule`
```python
class CoTModule(dspy.Module):
    """Chain-of-Thought prompting."""
    
    def forward(self, question: str) -> dspy.Prediction:
        """Generate step-by-step reasoning prediction."""
```

#### `SelfConsistencyModule`
```python
class SelfConsistencyModule(dspy.Module):
    """Self-consistency with multiple samples."""
    
    def __init__(self, n_samples: int = 5, temperature: float = 0.7):
        """
        Initialize self-consistency module.
        
        Args:
            n_samples: Number of reasoning paths
            temperature: Temperature for diverse generation
        """
        
    def forward(self, question: str) -> dspy.Prediction:
        """Generate prediction using multiple samples and voting."""
```

#### `PrologModule`
```python
class PrologModule(dspy.Module):
    """Prolog-style logical reasoning."""
    
    def forward(self, question: str) -> dspy.Prediction:
        """Generate structured logical reasoning prediction."""
```

## Advanced Techniques

### `EnhancedPrologModule`
```python
class EnhancedPrologModule(dspy.Module):
    """Enhanced Prolog with verification steps."""
    
    def forward(self, question: str) -> dspy.Prediction:
        """Generate Prolog-style prediction with verification."""
```

### `CalculatorAugmentedModule`
```python
class CalculatorAugmentedModule(dspy.Module):
    """Calculator-augmented reasoning."""
    
    def forward(self, question: str) -> dspy.Prediction:
        """Generate prediction with exact arithmetic computation."""
```

### `WeightedEnsembleModule`
```python
class WeightedEnsembleModule(dspy.Module):
    """Weighted ensemble of multiple techniques."""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize ensemble module.
        
        Args:
            weights: Dictionary mapping technique names to weights
        """
        
    def forward(self, question: str) -> dspy.Prediction:
        """Generate ensemble prediction using weighted voting."""
```

## CLI Commands

### `gsm8k-bench run`
```bash
python -m gsm8k_bench.cli run [OPTIONS]

Options:
  --samples INTEGER     Number of problems to test [default: 20]
  --techniques TEXT     Comma-separated list of techniques
  --config PATH         Configuration file path
  --output TEXT         Output directory [default: results]
  --model TEXT          OpenAI model name [default: gpt-3.5-turbo]
  --temperature FLOAT   Model temperature [default: 0.0]
  --visualize/--no-visualize  Create visualizations [default: True]
```

### `gsm8k-bench compare`
```bash
python -m gsm8k_bench.cli compare [OPTIONS]

Options:
  --technique1 TEXT     First technique to compare [required]
  --technique2 TEXT     Second technique to compare [required]
  --samples INTEGER     Number of problems per group [default: 50]
  --alpha FLOAT         Significance level [default: 0.05]
```

### `gsm8k-bench list-techniques`
```bash
python -m gsm8k_bench.cli list-techniques

Lists all available prompting techniques with descriptions.
```

## Configuration Schema

### Model Configuration
```yaml
model:
  name: str              # Model name (e.g., "gpt-3.5-turbo")
  temperature: float     # Temperature for generation [0.0-2.0]
  max_tokens: int        # Maximum tokens to generate
  timeout_seconds: int   # Timeout for API calls
```

### Benchmark Configuration
```yaml
benchmark:
  n_samples: int                    # Number of problems to evaluate
  techniques: List[str]             # List of technique names
  parallel_workers: int             # Number of parallel workers
  timeout_seconds: int              # Timeout per prediction
  save_detailed_predictions: bool   # Save prediction details
```

### Evaluation Configuration
```yaml
evaluation:
  accuracy_threshold: float     # Tolerance for numerical comparison
  confidence_level: float       # For confidence intervals
  statistical_tests: List[str]  # Statistical tests to run
```

## Error Handling

### Exception Classes

#### `BenchmarkError`
```python
class BenchmarkError(Exception):
    """Base exception for benchmark errors."""
    pass
```

#### `TechniqueError`
```python
class TechniqueError(BenchmarkError):
    """Error in technique implementation."""
    pass
```

#### `DataLoadError`
```python
class DataLoadError(BenchmarkError):
    """Error loading dataset."""
    pass
```

#### `APIError`
```python
class APIError(BenchmarkError):
    """Error calling OpenAI API."""
    pass
```

### Error Handling Patterns

```python
try:
    results = benchmark.run_benchmark()
except APIError as e:
    print(f"API error: {e}")
    # Implement retry logic
except DataLoadError as e:
    print(f"Data loading failed: {e}")
    # Try alternative data source
except BenchmarkError as e:
    print(f"Benchmark error: {e}")
    # Handle gracefully
```

## Environment Variables

```bash
# Required
OPENAI_API_KEY="your-openai-api-key"

# Optional
GSM8K_CACHE_DIR="/path/to/cache"          # Cache directory
GSM8K_RESULTS_DIR="/path/to/results"      # Results output directory  
GSM8K_LOG_LEVEL="INFO"                    # Logging level
GSM8K_MAX_WORKERS="4"                     # Maximum parallel workers
GSM8K_DEFAULT_MODEL="gpt-3.5-turbo"      # Default model name
GSM8K_API_TIMEOUT="30"                    # API timeout in seconds
```

## Type Definitions

```python
from typing import Dict, List, Optional, Tuple, Any, Union

# Type aliases
TechniqueResult = Dict[str, BenchmarkResult]
PredictionRecord = Dict[str, Any]
ModelConfig = Dict[str, Any]
BenchmarkConfig = Dict[str, Any]

# Protocol definitions
class TechniqueProtocol(Protocol):
    def forward(self, question: str) -> dspy.Prediction:
        ...

class EvaluatorProtocol(Protocol):
    def evaluate(self, predictions: List[PredictionRecord]) -> float:
        ...
```

## Version Compatibility

### DSPy Versions
- **Supported**: 2.4.0 - 2.5.x
- **Recommended**: 2.4.9
- **Breaking Changes**: Version 2.6.0+ requires code updates

### Python Versions
- **Minimum**: Python 3.8
- **Recommended**: Python 3.10+
- **Tested**: 3.8, 3.9, 3.10, 3.11

### API Compatibility
- **OpenAI API**: v1.0.0+
- **Models**: gpt-3.5-turbo, gpt-4, gpt-4-turbo
- **Rate Limits**: Standard tier recommended (3,500 RPM)
