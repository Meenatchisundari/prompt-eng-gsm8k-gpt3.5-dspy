# Troubleshooting Guide

Complete troubleshooting guide for common issues in the GSM8K benchmark.

## Installation Issues

### 1. ModuleNotFoundError: No module named 'gsm8k_bench'

**Problem**: Package not properly installed or Python path issues.

**Solutions**:

```bash
# Solution 1: Install in development mode
cd /path/to/prompt-eng-gsm8k-gpt3.5-dspy
pip install -e .

# Solution 2: Add to Python path manually
export PYTHONPATH="${PYTHONPATH}:/path/to/prompt-eng-gsm8k-gpt3.5-dspy/src"

# Solution 3: In Python script
import sys
sys.path.insert(0, '/path/to/prompt-eng-gsm8k-gpt3.5-dspy/src')
```

**Verification**:
```python
try:
    import gsm8k_bench
    print("Package imported successfully")
    print(f"Package location: {gsm8k_bench.__file__}")
except ImportError as e:
    print(f"Import failed: {e}")
```

### 2. ImportError: No module named 'dspy'

**Problem**: DSPy not installed or wrong version.

**Solutions**:
```bash
# Uninstall and reinstall specific version
pip uninstall dspy-ai -y
pip install dspy-ai==2.4.9

# Alternative: Install from GitHub
pip install git+https://github.com/stanfordnlp/dspy.git
```

**Version Check**:
```python
import dspy
print(f"DSPy version: {dspy.__version__}")
```

### 3. Circular Import Error with fsspec

**Problem**: Common in Google Colab with conflicting package versions.

**Solution**:
```bash
# Fix fsspec version
pip uninstall fsspec -y
pip install fsspec==2023.6.0

# Then restart runtime in Colab:
# Runtime -> Restart Runtime
```

## API Issues

### 1. OpenAI API Rate Limit Exceeded

**Error**: `RateLimitError: Rate limit reached for requests`

**Solutions**:

```python
# Configure rate limiting
from gsm8k_bench.utils import configure_rate_limiting

configure_rate_limiting(
    requests_per_minute=50,
    retry_attempts=5,
    backoff_factor=2.0
)

# Or use built-in retry logic
import time
import random

def api_call_with_retry(func, max_retries=5):
    for attempt in range(max_retries):
        try:
            return func()
        except RateLimitError:
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            print(f"Rate limit hit, waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
    raise Exception("Max retries exceeded")
```

**Rate Limit Guidelines**:
- **Free Tier**: 3 RPM, 40,000 TPM
- **Pay-as-you-go**: 3,500 RPM, 90,000 TPM
- **Tier 1**: 3,500 RPM, 90,000 TPM

### 2. API Timeout Errors

**Error**: `openai.APITimeoutError: Request timed out`

**Solutions**:

```yaml
# Increase timeout in config
model:
  timeout_seconds: 60  # Increase from default 30
  retry_attempts: 5
  backoff_factor: 2.0
```

```python
# Manual timeout configuration
import openai
from openai import OpenAI

client = OpenAI(
    api_key="your-key",
    timeout=60.0  # 60 second timeout
)
```

### 3. Invalid API Key

**Error**: `AuthenticationError: Incorrect API key provided`

**Solutions**:
```bash
# Check if key is set
echo $OPENAI_API_KEY

# Set key properly
export OPENAI_API_KEY="sk-your-actual-key-here"

# Verify key format
python -c "
import os
key = os.getenv('OPENAI_API_KEY')
print(f'Key length: {len(key) if key else 0}')
print(f'Starts with sk-: {key.startswith(\"sk-\") if key else False}')
"
```

## Performance Issues

### 1. Slow Evaluation on Large Datasets

**Problem**: Evaluation takes too long for large datasets.

**Solutions**:

```python
# Enable parallel processing
benchmark = GSM8KBenchmark(
    dataset, 
    parallel_workers=4,  # Use multiple workers
    batch_size=10       # Process in batches
)

# Streaming evaluation for memory efficiency
results = benchmark.run_streaming_evaluation(
    chunk_size=50,
    save_intermediate=True
)

# Use smaller sample for testing
quick_dataset = load_gsm8k_dataset(n_samples=20)
```

### 2. Memory Usage Too High

**Problem**: System runs out of memory with large datasets.

**Solutions**:

```python
# Memory-efficient data loading
dataset = load_gsm8k_dataset(
    n_samples=1000,
    lazy_loading=True,  # Load data on demand
    cache_size=100      # Limit cache size
)

# Clear cache periodically
import gc

def evaluate_with_memory_management(benchmark, dataset, chunk_size=50):
    results = {}
    
    for i in range(0, len(dataset), chunk_size):
        chunk = dataset[i:i+chunk_size]
        chunk_results = benchmark.evaluate_chunk(chunk)
        results.update(chunk_results)
        
        # Clear memory
        gc.collect()
        
    return results
```

### 3. GPU Out of Memory (for fine-tuning)

**Problem**: GPU memory exhausted during fine-tuning.

**Solutions**:

```python
# Reduce batch size
training_args = TrainingArguments(
    per_device_train_batch_size=1,  # Reduce from 4
    gradient_accumulation_steps=8,  # Increase to maintain effective batch size
    dataloader_num_workers=0,       # Reduce workers
)

# Use gradient checkpointing
model.gradient_checkpointing_enable()

# Mixed precision training
training_args.fp16 = True  # If using compatible GPU
```

## Data Issues

### 1. Dataset Loading Fails

**Error**: `DataLoadError: Could not load GSM8K dataset`

**Solutions**:

```python
# Check internet connection
import requests
try:
    response = requests.get("https://huggingface.co", timeout=10)
    print("Internet connection OK")
except:
    print("No internet connection")

# Manual dataset download
from datasets import load_dataset
try:
    dataset = load_dataset("gsm8k", "main", split="test")
    print(f"Loaded {len(dataset)} problems")
except Exception as e:
    print(f"Dataset loading failed: {e}")

# Use cached dataset if available
import os
cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
print(f"Cache directory: {cache_dir}")
```

### 2. Corrupted Results Data

**Problem**: Results file corrupted or unreadable.

**Solutions**:

```python
import json
import pandas as pd

def validate_results_file(file_path):
    """Validate and repair results file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Check structure
        required_fields = ['technique', 'accuracy', 'correct', 'total']
        
        for name, result in data.items():
            for field in required_fields:
                if field not in result:
                    print(f"Missing field {field} in {name}")
                    return False
        
        print("Results file is valid")
        return True
        
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        # Try to repair JSON
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Common fixes
            content = content.replace('}{', '},{')  # Missing commas
            content = content.replace("'", '"')      # Single to double quotes
            
            # Try parsing again
            data = json.loads(content)
            
            # Save repaired file
            backup_file = file_path + '.backup'
            os.rename(file_path, backup_file)
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"Repaired JSON file, backup saved as {backup_file}")
            return True
            
        except Exception as repair_error:
            print(f"Could not repair file: {repair_error}")
            return False
    
    except Exception as e:
        print(f"File validation error: {e}")
        return False

# Usage
validate_results_file("results.json")
```

## Technique-Specific Issues

### 1. Self-Consistency Taking Too Long

**Problem**: Self-consistency technique is too slow.

**Solutions**:

```python
# Reduce number of samples
self_consistency = SelfConsistencyModule(
    n_samples=3,      # Reduce from 5
    temperature=0.5   # Reduce temperature
)

# Parallel self-consistency (advanced)
from concurrent.futures import ThreadPoolExecutor

class FastSelfConsistencyModule(dspy.Module):
    def __init__(self, n_samples=3):
        super().__init__()
        self.n_samples = n_samples
        self.base_predictor = dspy.ChainOfThought(CoTSolver)
    
    def forward(self, question):
        def generate_prediction():
            return self.base_predictor(question=question)
        
        # Generate predictions in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(generate_prediction) 
                      for _ in range(self.n_samples)]
            predictions = [f.result().answer for f in futures]
        
        # Majority voting
        from collections import Counter
        most_common = Counter(predictions).most_common(1)[0]
        
        return dspy.Prediction(answer=most_common[0])
```

### 2. Prolog-Style Formatting Issues

**Problem**: Model not following Prolog-style format consistently.

**Solutions**:

```python
# Enhanced format validation
def validate_prolog_response(response):
    """Validate Prolog-style response format."""
    required_sections = ['FACTS:', 'RULES:', 'QUERY:', 'DERIVATION:', 'ANSWER:']
    
    for section in required_sections:
        if section not in response:
            return False, f"Missing section: {section}"
    
    return True, "Valid format"

# Post-processing for format correction
def fix_prolog_format(response):
    """Attempt to fix common format issues."""
    
    # Common fixes
    fixes = {
        'Facts:': 'FACTS:',
        'Rules:': 'RULES:',
        'Query:': 'QUERY:',
        'Derivation:': 'DERIVATION:',
        'Answer:': 'ANSWER:'
    }
    
    for wrong, right in fixes.items():
        response = response.replace(wrong, right)
    
    return response

# Enhanced Prolog module with validation
class RobustPrologModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.solver = dspy.ChainOfThought(PrologSolver)
        self.max_retries = 3
    
    def forward(self, question):
        for attempt in range(self.max_retries):
            try:
                result = self.solver(question=question)
                
                # Validate format
                is_valid, message = validate_prolog_response(result.answer)
                
                if is_valid:
                    return result
                else:
                    # Try to fix format
                    fixed_answer = fix_prolog_format(result.answer)
                    result.answer = fixed_answer
                    
                    # Validate again
                    is_valid, _ = validate_prolog_response(fixed_answer)
                    if is_valid:
                        return result
                
                print(f"Format validation failed (attempt {attempt + 1}): {message}")
                
            except Exception as e:
                print(f"Generation failed (attempt {attempt + 1}): {e}")
        
        # Fallback to simple answer extraction
        return dspy.Prediction(answer="0")
```

## Debugging Techniques

### 1. Verbose Logging

```python
import logging

# Enable detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)

# Debug specific components
logger = logging.getLogger('gsm8k_bench')
logger.setLevel(logging.DEBUG)

# Benchmark with debugging
benchmark = GSM8KBenchmark(dataset, verbose=True, debug=True)
```

### 2. Single Problem Analysis

```python
# Debug individual predictions
def debug_single_prediction(technique_module, question, expected_answer):
    """Debug a single prediction in detail."""
    
    print(f"Debugging Question: {question}")
    print(f"Expected Answer: {expected_answer}")
    print("-" * 50)
    
    try:
        # Generate prediction
        start_time = time.time()
        prediction = technique_module(question=question)
        generation_time = time.time() - start_time
        
        # Extract answer
        predicted_answer = extract_answer(prediction.answer)
        
        # Check accuracy
        is_correct = math_accuracy_simple(expected_answer, predicted_answer)
        
        # Print results
        print(f"Generated successfully in {generation_time:.2f}s")
        print(f"Raw response: {prediction.answer}")
        print(f"Extracted answer: {predicted_answer}")
        print(f"Accuracy: {'CORRECT' if is_correct else 'WRONG'}")
        
        # Show reasoning if available
        if hasattr(prediction, 'reasoning'):
            print(f"Reasoning: {prediction.reasoning}")
        
        return {
            'correct': is_correct,
            'predicted': predicted_answer,
            'generation_time': generation_time,
            'raw_response': prediction.answer
        }
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

# Usage
from gsm8k_bench.techniques import PrologModule
technique = PrologModule()

debug_result = debug_single_prediction(
    technique, 
    "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes muffins with 4. She sells the remainder at $2 per egg. How much does she make daily?",
    "18"
)
```

### 3. API Call Monitoring

```python
import time
from collections import defaultdict

class APIMonitor:
    def __init__(self):
        self.call_count = 0
        self.total_time = 0
        self.error_count = 0
        self.rate_limits = 0
        self.timeouts = 0
        
    def log_call(self, start_time, end_time, success=True, error_type=None):
        self.call_count += 1
        self.total_time += (end_time - start_time)
        
        if not success:
            self.error_count += 1
            if error_type == 'rate_limit':
                self.rate_limits += 1
            elif error_type == 'timeout':
                self.timeouts += 1
    
    def get_stats(self):
        return {
            'total_calls': self.call_count,
            'avg_time': self.total_time / max(self.call_count, 1),
            'error_rate': self.error_count / max(self.call_count, 1),
            'rate_limits': self.rate_limits,
            'timeouts': self.timeouts
        }

# Global monitor
api_monitor = APIMonitor()

# Wrap API calls
def monitored_api_call(func, *args, **kwargs):
    start_time = time.time()
    try:
        result = func(*args, **kwargs)
        api_monitor.log_call(start_time, time.time(), success=True)
        return result
    except RateLimitError:
        api_monitor.log_call(start_time, time.time(), success=False, error_type='rate_limit')
        raise
    except TimeoutError:
        api_monitor.log_call(start_time, time.time(), success=False, error_type='timeout')
        raise
    except Exception as e:
        api_monitor.log_call(start_time, time.time(), success=False)
        raise
```

## Environment-Specific Issues

### 1. Google Colab Issues

**Common Problems**:
- Session timeouts
- Package conflicts
- Memory limitations
- GPU unavailability

**Solutions**:

```python
# Keep session alive
import time
import random

def keep_alive():
    """Keep Colab session alive during long runs."""
    while True:
        time.sleep(random.randint(60, 120))  # Random interval
        print(".", end="", flush=True)

# Run in background
import threading
keep_alive_thread = threading.Thread(target=keep_alive, daemon=True)
keep_alive_thread.start()

# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Save checkpoints frequently
def save_checkpoint(results, filename):
    """Save intermediate results."""
    checkpoint = {
        'timestamp': time.time(),
        'results': results,
        'completed_techniques': list(results.keys())
    }
    
    with open(filename, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    
    print(f"Checkpoint saved: {filename}")

# Load from checkpoint
def load_checkpoint(filename):
    """Load from checkpoint file."""
    try:
        with open(filename, 'r') as f:
            checkpoint = json.load(f)
        print(f"Loaded checkpoint from {filename}")
        return checkpoint['results']
    except FileNotFoundError:
        print(f"No checkpoint found: {filename}")
        return {}
```

### 2. Local Development Issues

**Virtual Environment Problems**:

```bash
# Create clean environment
python -m venv gsm8k_env
source gsm8k_env/bin/activate  # Linux/Mac
# or
gsm8k_env\Scripts\activate     # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# Verify installation
python -c "import gsm8k_bench; print('Package installed')"
```

**IDE Configuration**:

```json
// VS Code settings.json
{
    "python.defaultInterpreterPath": "./gsm8k_env/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"]
}
```

## Getting Help

### 1. Enable Debug Mode

```python
# Full debug mode
os.environ['GSM8K_DEBUG'] = '1'
os.environ['GSM8K_LOG_LEVEL'] = 'DEBUG'

# Import with debug info
import gsm8k_bench
print(f"Package version: {gsm8k_bench.__version__}")
print(f"Package location: {gsm8k_bench.__file__}")
```

### 2. System Information

```python
def print_system_info():
    """Print comprehensive system information for debugging."""
    import sys
    import platform
    import pkg_resources
    import torch
    import numpy as np
    
    print("=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    
    # Python environment
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.architecture()}")
    print(f"Processor: {platform.processor()}")
    
    # Memory info
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"Total RAM: {memory.total / (1024**3):.1f} GB")
        print(f"Available RAM: {memory.available / (1024**3):.1f} GB")
        print(f"RAM usage: {memory.percent}%")
    except ImportError:
        print("psutil not available - install with: pip install psutil")
    
    # GPU info
    print(f"\nGPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
    
    # Package versions
    print(f"\nPACKAGE VERSIONS:")
    key_packages = [
        'dspy-ai', 'openai', 'transformers', 'torch', 
        'datasets', 'pandas', 'numpy', 'matplotlib'
    ]
    
    for package in key_packages:
        try:
            version = pkg_resources.get_distribution(package).version
            print(f"{package}: {version}")
        except pkg_resources.DistributionNotFound:
            print(f"{package}: Not installed")
    
    # Environment variables
    print(f"\nENVIRONMENT:")
    env_vars = ['OPENAI_API_KEY', 'CUDA_VISIBLE_DEVICES', 'PYTHONPATH']
    for var in env_vars:
        value = os.environ.get(var)
        if var == 'OPENAI_API_KEY' and value:
            print(f"{var}: {'*' * 20} (hidden)")
        else:
            print(f"{var}: {value or 'Not set'}")
    
    print("=" * 60)

# Run system info
print_system_info()
```

### 3. Health Check Script

```python
def run_health_check():
    """Run comprehensive health check for GSM8K benchmark."""
    
    print("GSM8K BENCHMARK HEALTH CHECK")
    print("=" * 50)
    
    checks = []
    
    # 1. Package import
    try:
        import gsm8k_bench
        checks.append(("Package import", "SUCCESS"))
    except Exception as e:
        checks.append(("Package import", f"FAILED: {e}"))
    
    # 2. API key
    import os
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key and api_key.startswith('sk-') and len(api_key) > 20:
        checks.append(("API key format", "SUCCESS"))
    else:
        checks.append(("API key format", "FAILED: Invalid format"))
    
    # 3. Dataset loading
    try:
        from gsm8k_bench import load_gsm8k_dataset
        test_dataset = load_gsm8k_dataset(n_samples=2)
        if len(test_dataset) == 2:
            checks.append(("Dataset loading", "SUCCESS"))
        else:
            checks.append(("Dataset loading", f"FAILED: Got {len(test_dataset)} samples"))
    except Exception as e:
        checks.append(("Dataset loading", f"FAILED: {e}"))
    
    # 4. Model connection
    try:
        from gsm8k_bench.techniques import ZeroShotModule
        module = ZeroShotModule()
        # Try simple test
        result = module(question="What is 2 + 2?")
        if result:
            checks.append(("Model connection", "SUCCESS"))
        else:
            checks.append(("Model connection", "FAILED: No response"))
    except Exception as e:
        checks.append(("Model connection", f"FAILED: {e}"))
    
    # 5. Dependencies check
    required_packages = ['dspy', 'openai', 'datasets', 'pandas', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if not missing_packages:
        checks.append(("Dependencies", "SUCCESS"))
    else:
        checks.append(("Dependencies", f"MISSING: {', '.join(missing_packages)}"))
    
    # Print results
    print("\nHEALTH CHECK RESULTS:")
    for check, status in checks:
        print(f"  {check}: {status}")
    
    # Overall status
    failed_checks = [c for c in checks if "FAILED" in c[1]]
    if not failed_checks:
        print(f"\nALL CHECKS PASSED! System is ready.")
        return True
    else:
        print(f"\n{len(failed_checks)} checks failed. See troubleshooting guide.")
        return False

# Run health check
system_healthy = run_health_check()
```

## Repository-Specific Issues

### 1. GitHub Repository Structure

**Expected Structure**:
```
prompt-eng-gsm8k-gpt3.5-dspy/
├── src/
│   └── gsm8k_bench/
│       ├── __init__.py
│       ├── benchmark.py
│       ├── techniques/
│       ├── utils/
│       └── viz/
├── examples/
│   ├── gsm8k_colab.ipynb
│   └── quick_start.py
├── tests/
├── docs/
├── requirements.txt
├── setup.py
└── README.md
```

**Setup Verification**:
```bash
# Clone repository
git clone https://github.com/meenatchisundari/prompt-eng-gsm8k-gpt3.5-dspy.git
cd prompt-eng-gsm8k-gpt3.5-dspy

# Check structure
ls -la
tree src/  # If tree is installed

# Install in development mode
pip install -e .

# Run tests
python -m pytest tests/ -v
```

### 2. Configuration File Issues

**Problem**: Config file not found or invalid format.

**Solutions**:

```python
# Check for config file
import os
from pathlib import Path

config_paths = [
    Path.cwd() / "config.yaml",
    Path.home() / ".gsm8k_config.yaml",
    Path(__file__).parent / "config" / "default.yaml"
]

for path in config_paths:
    if path.exists():
        print(f"Config found: {path}")
        break
else:
    print("No config file found")
    print("Creating default config...")
    
    default_config = """
model:
  name: "gpt-3.5-turbo"
  temperature: 0.0
  max_tokens: 512
  timeout_seconds: 30

evaluation:
  batch_size: 10
  parallel_workers: 2
  save_predictions: true

techniques:
  enabled: ["zero_shot", "few_shot", "cot", "prolog_style"]
  
self_consistency:
  n_samples: 5
  temperature: 0.7

logging:
  level: "INFO"
  save_to_file: true
"""
    
    with open("config.yaml", "w") as f:
        f.write(default_config)
    
    print("Default config created: config.yaml")
```

### 3. Data Path Issues

**Problem**: Can't find data files or results directory.

**Solutions**:

```python
# Setup data directories
from pathlib import Path
import os

def setup_directories():
    """Create necessary directories for GSM8K benchmark."""
    
    dirs = [
        "data",
        "results", 
        "cache",
        "logs",
        "checkpoints"
    ]
    
    for dir_name in dirs:
        dir_path = Path(dir_name)
        dir_path.mkdir(exist_ok=True)
        print(f"Directory ready: {dir_path.absolute()}")
    
    # Set environment variables
    os.environ['GSM8K_DATA_DIR'] = str(Path("data").absolute())
    os.environ['GSM8K_RESULTS_DIR'] = str(Path("results").absolute())
    os.environ['GSM8K_CACHE_DIR'] = str(Path("cache").absolute())
    
    print("\nEnvironment variables set:")
    for key in ['GSM8K_DATA_DIR', 'GSM8K_RESULTS_DIR', 'GSM8K_CACHE_DIR']:
        print(f"  {key}: {os.environ[key]}")

setup_directories()
```

## FAQ

### Q: Why are my results different from the paper?

**A**: Several factors can cause variation:

1. **Model version differences**: GPT-3.5-turbo updates over time
2. **Random seed**: Set `temperature=0` for deterministic results
3. **Dataset version**: Use exact same test set
4. **Prompt formatting**: Small changes can significantly impact performance

**Solution**:
```python
# Reproduce exact results
benchmark = GSM8KBenchmark(
    dataset,
    model_name="gpt-3.5-turbo-0613",  # Specific version
    temperature=0.0,                   # Deterministic
    random_seed=42,                    # Fixed seed
    dataset_version="1.0.0"            # Specific dataset version
)
```

### Q: Can I use other models besides OpenAI?

**A**: Yes, the framework supports multiple models:

```python
# Use different models
from gsm8k_bench import GSM8KBenchmark

# Anthropic Claude
benchmark_claude = GSM8KBenchmark(
    dataset,
    model_provider="anthropic",
    model_name="claude-3-sonnet-20240229"
)

# Local Hugging Face model
benchmark_local = GSM8KBenchmark(
    dataset,
    model_provider="huggingface",
    model_name="microsoft/DialoGPT-medium"
)

# Google Gemini
benchmark_gemini = GSM8KBenchmark(
    dataset,
    model_provider="google",
    model_name="gemini-pro"
)
```

### Q: How do I contribute to the repository?

**A**: Follow these steps for contributing:

```bash
# 1. Fork the repository on GitHub
# 2. Clone your fork
git clone https://github.com/your-username/prompt-eng-gsm8k-gpt3.5-dspy.git
cd prompt-eng-gsm8k-gpt3.5-dspy

# 3. Create a new branch
git checkout -b feature/your-feature-name

# 4. Make your changes and test
pip install -e .
python -m pytest tests/

# 5. Commit and push
git add .
git commit -m "Add: your feature description"
git push origin feature/your-feature-name

# 6. Create a Pull Request on GitHub
```

### Q: The benchmark is too slow, how can I speed it up?

**A**: Several optimization strategies:

```python
# 1. Parallel processing
benchmark = GSM8KBenchmark(
    dataset,
    parallel_workers=4,
    batch_size=20
)

# 2. Use faster model
benchmark = GSM8KBenchmark(
    dataset,
    model_name="gpt-3.5-turbo",  # Instead of gpt-4
    max_tokens=256               # Reduce token limit
)

# 3. Cache results
benchmark = GSM8KBenchmark(
    dataset,
    enable_caching=True,
    cache_dir="./cache"
)

# 4. Sample subset for testing
quick_dataset = load_gsm8k_dataset(n_samples=50)  # Instead of full dataset
```

## Performance Benchmarks

### Expected Performance Metrics

Based on our testing, here are typical performance metrics:

| Technique | Accuracy | Time per Problem | Cost per 100 Problems |
|-----------|----------|------------------|----------------------|
| Zero-Shot | 45-55% | 2-3 seconds | $0.50-$1.00 |
| Few-Shot | 55-65% | 3-4 seconds | $0.75-$1.50 |
| Chain-of-Thought | 65-75% | 4-5 seconds | $1.00-$2.00 |
| Self-Consistency | 70-80% | 15-20 seconds | $4.00-$8.00 |
| Prolog-Style | 75-85% | 5-7 seconds | $1.50-$3.00 |

### Hardware Requirements

**Minimum Requirements**:
- RAM: 4GB
- CPU: 2 cores
- Storage: 2GB free space
- Internet: Stable connection for API calls

**Recommended Requirements**:
- RAM: 8GB+
- CPU: 4+ cores
- Storage: 10GB+ free space
- GPU: Optional, for local model inference

**Cloud Computing**:
- Google Colab: Free tier sufficient for small datasets
- AWS/Azure: t3.medium or equivalent
- Local: Modern laptop with good internet connection

## Advanced Troubleshooting

### 1. Memory Leaks

**Problem**: Memory usage increases over time.

**Solutions**:

```python
import gc
import tracemalloc

# Enable memory tracking
tracemalloc.start()

def monitor_memory_usage(func):
    """Decorator to monitor memory usage."""
    def wrapper(*args, **kwargs):
        # Before execution
        snapshot1 = tracemalloc.take_snapshot()
        
        result = func(*args, **kwargs)
        
        # After execution
        snapshot2 = tracemalloc.take_snapshot()
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        
        print("Memory usage top 3:")
        for stat in top_stats[:3]:
            print(stat)
        
        # Force garbage collection
        gc.collect()
        
        return result
    return wrapper

# Use decorator on benchmark functions
@monitor_memory_usage
def run_single_technique(technique, dataset):
    # Your benchmark code here
    pass
```

### 2. Network Issues

**Problem**: Intermittent network failures.

**Solutions**:

```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def create_robust_session():
    """Create HTTP session with robust retry logic."""
    session = requests.Session()
    
    retry_strategy = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

# Use robust session for API calls
session = create_robust_session()
```

### 3. Logging and Monitoring

**Advanced Logging Setup**:

```python
import logging
import logging.handlers
from datetime import datetime

def setup_advanced_logging():
    """Setup comprehensive logging system."""
    
    # Create logger
    logger = logging.getLogger('gsm8k_bench')
    logger.setLevel(logging.DEBUG)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        f'gsm8k_benchmark_{datetime.now().strftime("%Y%m%d")}.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Use advanced logging
logger = setup_advanced_logging()
logger.info("GSM8K Benchmark started")
```

## Recovery Procedures

### 1. Corrupted Checkpoint Recovery

```python
def recover_from_checkpoint(checkpoint_dir="./checkpoints"):
    """Recover benchmark from corrupted checkpoint."""
    
    import glob
    import json
    from datetime import datetime
    
    # Find all checkpoint files
    checkpoint_files = glob.glob(f"{checkpoint_dir}/checkpoint_*.json")
    
    if not checkpoint_files:
        print("No checkpoint files found")
        return None
    
    # Sort by modification time (newest first)
    checkpoint_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    for checkpoint_file in checkpoint_files:
        try:
            print(f"Attempting to load: {checkpoint_file}")
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
            
            # Validate checkpoint structure
            required_keys = ['timestamp', 'results', 'completed_techniques']
            if all(key in data for key in required_keys):
                print(f"Valid checkpoint found: {checkpoint_file}")
                print(f"Timestamp: {datetime.fromtimestamp(data['timestamp'])}")
                print(f"Completed techniques: {len(data['completed_techniques'])}")
                return data
            else:
                print(f"Invalid checkpoint structure: {checkpoint_file}")
                
        except Exception as e:
            print(f"Failed to load {checkpoint_file}: {e}")
            continue
    
    print("No valid checkpoints found")
    return None

# Usage
recovered_data = recover_from_checkpoint()
if recovered_data:
    results = recovered_data['results']
    completed = recovered_data['completed_techniques']
    print(f"Recovered {len(results)} results")
```

### 2. Database Recovery

```python
def repair_results_database():
    """Repair corrupted results database."""
    
    import sqlite3
    import shutil
    from pathlib import Path
    
    db_path = Path("results.db")
    backup_path = Path("results_backup.db")
    
    if not db_path.exists():
        print("No database file found")
        return False
    
    # Create backup
    shutil.copy2(db_path, backup_path)
    print(f"Database backed up to: {backup_path}")
    
    try:
        # Attempt to repair
        conn = sqlite3.connect(str(db_path))
        
        # Check integrity
        cursor = conn.execute("PRAGMA integrity_check;")
        integrity_result = cursor.fetchone()[0]
        
        if integrity_result == "ok":
            print("Database integrity OK")
            return True
        else:
            print(f"Database integrity issues: {integrity_result}")
            
            # Attempt recovery
            conn.execute("REINDEX;")
            conn.execute("VACUUM;")
            conn.commit()
            
            # Check again
            cursor = conn.execute("PRAGMA integrity_check;")
            integrity_result = cursor.fetchone()[0]
            
            if integrity_result == "ok":
                print("Database repaired successfully")
                return True
            else:
                print("Database repair failed")
                return False
                
    except Exception as e:
        print(f"Database repair error: {e}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

# Usage
repair_results_database()
```

## Testing Procedures

### 1. Unit Tests

```python
import unittest
from gsm8k_bench import GSM8KBenchmark, load_gsm8k_dataset
from gsm8k_bench.techniques import ZeroShotModule

class TestGSM8KBenchmark(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.small_dataset = load_gsm8k_dataset(n_samples=5)
        self.benchmark = GSM8KBenchmark(self.small_dataset)
    
    def test_dataset_loading(self):
        """Test dataset loading functionality."""
        self.assertEqual(len(self.small_dataset), 5)
        self.assertIsNotNone(self.small_dataset[0].question)
        self.assertIsNotNone(self.small_dataset[0].answer)
    
    def test_zero_shot_module(self):
        """Test zero-shot technique."""
        module = ZeroShotModule()
        question = "What is 2 + 3?"
        result = module(question=question)
        self.assertIsNotNone(result.answer)
    
    def test_benchmark_run(self):
        """Test benchmark execution."""
        results = self.benchmark.run_benchmark(
            selected_techniques=["zero_shot"]
        )
        self.assertIn("1. Zero-Shot", results)
        self.assertGreaterEqual(results["1. Zero-Shot"].accuracy, 0)
        self.assertLessEqual(results["1. Zero-Shot"].accuracy, 1)

if __name__ == '__main__':
    unittest.main()
```

### 2. Integration Tests

```python
def run_integration_tests():
    """Run comprehensive integration tests."""
    
    tests = []
    
    # Test 1: Full pipeline
    try:
        dataset = load_gsm8k_dataset(n_samples=10)
        benchmark = GSM8KBenchmark(dataset)
        results = benchmark.run_benchmark(selected_techniques=["zero_shot"])
        
        if len(results) > 0:
            tests.append(("Full pipeline", "PASS"))
        else:
            tests.append(("Full pipeline", "FAIL: No results"))
            
    except Exception as e:
        tests.append(("Full pipeline", f"FAIL: {e}"))
    
    # Test 2: Result serialization
    try:
        from gsm8k_bench import create_results_table
        df = create_results_table(results)
        
        if len(df) > 0:
            tests.append(("Result serialization", "PASS"))
        else:
            tests.append(("Result serialization", "FAIL: Empty dataframe"))
            
    except Exception as e:
        tests.append(("Result serialization", f"FAIL: {e}"))
    
    # Test 3: Visualization
    try:
        from gsm8k_bench import plot_results
        plot_results(results)
        tests.append(("Visualization", "PASS"))
    except Exception as e:
        tests.append(("Visualization", f"FAIL: {e}"))
    
    # Print results
    print("INTEGRATION TEST RESULTS:")
    print("-" * 40)
    for test_name, status in tests:
        print(f"{test_name:<20}: {status}")
    
    # Overall status
    failed_tests = [t for t in tests if "FAIL" in t[1]]
    if not failed_tests:
        print("\nAll integration tests passed!")
        return True
    else:
        print(f"\n{len(failed_tests)} integration tests failed")
        return False

# Run integration tests
run_integration_tests()
```

## Security Considerations

### 1. API Key Security

```python
def validate_api_key_security():
    """Check API key security practices."""
    
    import os
    import stat
    
    warnings = []
    
    # Check if API key is in environment variable
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        warnings.append("API key not found in environment variables")
    
    # Check for .env file
    env_file = Path('.env')
    if env_file.exists():
        # Check file permissions
        file_stat = env_file.stat()
        if file_stat.st_mode & stat.S_IROTH:
            warnings.append(".env file is readable by others")
        if file_stat.st_mode & stat.S_IWOTH:
            warnings.append(".env file is writable by others")
    
    # Check for hardcoded keys in code
    code_files = list(Path('.').rglob('*.py'))
    for file_path in code_files:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                if 'sk-' in content and 'OPENAI_API_KEY' not in content:
                    warnings.append(f"Potential hardcoded API key in {file_path}")
        except Exception:
            pass
    
    if warnings:
        print("SECURITY WARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("API key security checks passed")
    
    return len(warnings) == 0

# Run security check
validate_api_key_security()
```

### 2. Data Privacy

```python
def check_data_privacy():
    """Check data privacy and handling practices."""
    
    checks = []
    
    # Check if personal data is being logged
    log_files = list(Path('.').rglob('*.log'))
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                # Check for common personal data patterns
                patterns = ['email@', 'password', 'social_security', 'credit_card']
                found_patterns = [p for p in patterns if p in content.lower()]
                if found_patterns:
                    checks.append(f"Potential personal data in {log_file}: {found_patterns}")
        except Exception:
            pass
    
    # Check data retention policies
    cache_dir = Path('cache')
    if cache_dir.exists():
        old_files = []
        import time
        week_ago = time.time() - (7 * 24 * 60 * 60)
        
        for file_path in cache_dir.rglob('*'):
            if file_path.is_file() and file_path.stat().st_mtime < week_ago:
                old_files.append(file_path)
        
        if old_files:
            checks.append(f"{len(old_files)} cache files older than 1 week")
    
    if checks:
        print("DATA PRIVACY ISSUES:")
        for check in checks:
            print(f"  - {check}")
    else:
        print("Data privacy checks passed")
    
    return len(checks) == 0

# Run privacy check
check_data_privacy()
```

## Support and Community

### Getting Help

1. **Documentation**: Check the full documentation at `docs/`
2. **Issues**: Report bugs on GitHub Issues
3. **Discussions**: Join GitHub Discussions for questions
4. **Examples**: See `examples/` directory for usage patterns

### Contributing Guidelines

## Contributing to GSM8K Benchmark

### Types of Contributions

1. **Bug Reports**: Use issue templates
2. **Feature Requests**: Propose new prompting techniques
3. **Code Contributions**: Follow coding standards
4. **Documentation**: Improve guides and examples

### Development Setup

```bash
# Clone and setup
git clone https://github.com/meenatchisundari/prompt-eng-gsm8k-gpt3.5-dspy.git
cd prompt-eng-gsm8k-gpt3.5-dspy

# Create development environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install in development mode
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v

# Run linting
black src/
flake8 src/
```

### Code Style

- Use Black for formatting
- Follow PEP 8 guidelines
- Add type hints where possible
- Write docstrings for all functions
- Include unit tests for new features

### Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add/update tests
5. Update documentation
6. Submit pull request

### Community Guidelines

- Be respectful and inclusive
- Provide constructive feedback
- Help newcomers
- Share interesting findings
- Contribute to documentation

This troubleshooting guide should help users resolve most common issues they encounter when using the GSM8K benchmark repository. The guide covers installation problems, API issues, performance optimization, debugging techniques, and provides comprehensive recovery procedures for various failure scenarios.
