"""
GSM8K Prompting Techniques Benchmark with DSPy

A comprehensive benchmark for evaluating different prompting techniques
on the GSM8K math word problems dataset.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .benchmark import GSM8KBenchmark, run_ab_test
from .data import load_gsm8k_dataset
from .utils import BenchmarkResult, math_accuracy, extract_answer
from .viz import create_results_table, plot_results

# Import core techniques
from .techniques import (
    ZeroShotModule,
    FewShotModule, 
    CoTModule,
    SelfConsistencyModule,
    PrologModule
)

# Import improvements (optional)
try:
    from .improvements import (
        EnhancedPrologModule,
        CalculatorAugmentedModule,
        VerificationChainModule,
        WeightedEnsembleModule
    )
    IMPROVEMENTS_AVAILABLE = True
except ImportError:
    IMPROVEMENTS_AVAILABLE = False

# Import fine-tuning (optional)
try:
    from .finetune import QuickFineTuner, run_finetuning_vs_prompting_comparison
    FINETUNING_AVAILABLE = True
except ImportError:
    FINETUNING_AVAILABLE = False

__all__ = [
    # Core functionality
    'GSM8KBenchmark',
    'run_ab_test',
    'load_gsm8k_dataset',
    'BenchmarkResult',
    'math_accuracy',
    'extract_answer',
    'create_results_table',
    'plot_results',
    
    # Core techniques
    'ZeroShotModule',
    'FewShotModule',
    'CoTModule', 
    'SelfConsistencyModule',
    'PrologModule',
]

# Add improvements if available
if IMPROVEMENTS_AVAILABLE:
    __all__.extend([
        'EnhancedPrologModule',
        'CalculatorAugmentedModule', 
        'VerificationChainModule',
        'WeightedEnsembleModule'
    ])

# Add fine-tuning if available
if FINETUNING_AVAILABLE:
    __all__.extend([
        'QuickFineTuner',
        'run_finetuning_vs_prompting_comparison'
    ])


def get_info():
    """Get package information"""
    info = {
        'version': __version__,
        'author': __author__,
        'email': __email__,
        'improvements_available': IMPROVEMENTS_AVAILABLE,
        'finetuning_available': FINETUNING_AVAILABLE,
    }
    return info
