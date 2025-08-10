"""
Visualization and results formatting utilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging

from .utils import BenchmarkResult

logger = logging.getLogger(__name__)

# Optional matplotlib import
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
except ImportError:
    PLOTTING_AVAILABLE = False
    logger.warning("Matplotlib/Seaborn not available - plotting disabled")


def create_results_table(results: Dict[str, BenchmarkResult]) -> pd.DataFrame:
    """
    Create a formatted results table from benchmark results.
    
    Args:
        results: Dictionary mapping technique names to BenchmarkResult objects
        
    Returns:
        Pandas DataFrame with formatted results
    """
    
    data = []
    for name, result in results.items():
        # Calculate efficiency (accuracy per second)
        efficiency = result.accuracy / result.avg_time if result.avg_time > 0 else 0
        
        data.append({
            'Technique': name,
            'Accuracy (%)': f"{result.accuracy*100:.2f}%",
            'Correct/Total': f"{result.correct}/{result.total}",
            'Avg Time (s)': f"{result.avg_time:.2f}",
            'Error Rate (%)': f"{result.error_rate*100:.1f}%",
            'Efficiency': f"{efficiency:.2f}",
            'Raw Accuracy': result.accuracy,  # For sorting
