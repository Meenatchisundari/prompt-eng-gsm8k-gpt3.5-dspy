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
        })
    
    df = pd.DataFrame(data)
    
    # Sort by accuracy (descending)
    df = df.sort_values('Raw Accuracy', ascending=False)
    df = df.drop('Raw Accuracy', axis=1)  # Remove helper column
    
    return df


def plot_results(results: Dict[str, BenchmarkResult], 
                save_path: Optional[str] = None, 
                figsize: tuple = (20, 16)) -> None:
    """
    Create comprehensive visualizations of benchmark results.
    
    Args:
        results: Dictionary mapping technique names to BenchmarkResult objects
        save_path: Optional path to save the plot
        figsize: Figure size as (width, height)
    """
    
    if not PLOTTING_AVAILABLE:
        logger.warning("Plotting not available - install matplotlib and seaborn")
        return
    
    # Prepare data
    techniques = list(results.keys())
    accuracies = [r.accuracy * 100 for r in results.values()]
    avg_times = [r.avg_time for r in results.values()]
    correct_counts = [r.correct for r in results.values()]
    total_count = list(results.values())[0].total
    
    # Create comprehensive subplot layout
    fig = plt.figure(figsize=figsize)
    
    # 1. Main Accuracy Comparison (Large)
    ax1 = plt.subplot(3, 3, (1, 2))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    bars1 = ax1.bar(range(len(techniques)), accuracies,
                    color=colors[:len(techniques)], alpha=0.8, 
                    edgecolor='black', linewidth=1)
    ax1.set_xlabel('Prompting Technique', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title(' Accuracy Comparison Across Prompting Techniques', 
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(techniques)))
    ax1.set_xticklabels([_clean_technique_name(t) for t in techniques],
                       rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. Time Performance
    ax2 = plt.subplot(3, 3, 3)
    bars2 = ax2.bar(range(len(techniques)), avg_times,
                    color='lightcoral', alpha=0.8, edgecolor='black')
    ax2.set_xlabel('Technique', fontsize=10)
    ax2.set_ylabel('Avg Time (s)', fontsize=10)
    ax2.set_title(' Response Time', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(len(techniques)))
    ax2.set_xticklabels([_clean_technique_name(t)[:8] for t in techniques],
                       rotation=45, ha='right', fontsize=8)
    
    # 3. Accuracy vs Time Scatter
    ax3 = plt.subplot(3, 3, 4)
    scatter = ax3.scatter(avg_times, accuracies, s=200, alpha=0.7, 
                         c=colors[:len(techniques)], edgecolors='black')
    for i, technique in enumerate(techniques):
        ax3.annotate(_clean_technique_name(technique)[:10],
                    (avg_times[i], accuracies[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax3.set_xlabel('Average Time (seconds)', fontsize=10)
    ax3.set_ylabel('Accuracy (%)', fontsize=10)
    ax3.set_title(' Accuracy vs Time Trade-off', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Efficiency (Accuracy/Time)
    efficiency = [acc/time if time > 0 else 0 for acc, time in zip(accuracies, avg_times)]
    ax4 = plt.subplot(3, 3, 5)
    bars4 = ax4.bar(range(len(techniques)), efficiency, color='gold', 
                    alpha=0.8, edgecolor='black')
    ax4.set_xlabel('Technique', fontsize=10)
    ax4.set_ylabel('Efficiency', fontsize=10)
    ax4.set_title(' Efficiency (Acc%/Sec)', fontsize=12, fontweight='bold')
    ax4.set_xticks(range(len(techniques)))
    ax4.set_xticklabels([_clean_technique_name(t)[:8] for t in techniques],
                       rotation=45, ha='right', fontsize=8)
    
    # 5. Correct vs Incorrect Distribution
    ax5 = plt.subplot(3, 3, 6)
    incorrect_counts = [total_count - correct for correct in correct_counts]
    x = np.arange(len(techniques))
    width = 0.6
    
    bars_correct = ax5.bar(x, correct_counts, width, label='Correct',
                          color='lightgreen', alpha=0.8, edgecolor='black')
    bars_incorrect = ax5.bar(x, incorrect_counts, width, bottom=correct_counts,
                            label='Incorrect', color='lightcoral', alpha=0.8, 
                            edgecolor='black')
    
    ax5.set_xlabel('Technique', fontsize=10)
    ax5.set_ylabel('Count', fontsize=10)
    ax5.set_title(' Correct vs Incorrect', fontsize=12, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels([_clean_technique_name(t)[:8] for t in techniques],
                       rotation=45, ha='right', fontsize=8)
    ax5.legend()
    
    # 6. Improvement Over Baseline
    ax6 = plt.subplot(3, 3, 7)
    baseline_acc = min(accuracies)  # Use lowest as baseline
    improvements = [acc - baseline_acc for acc in accuracies]
    colors_improvement = ['gray' if imp <= 0 else 'green' for imp in improvements]
    bars6 = ax6.bar(range(len(techniques)), improvements,
                    color=colors_improvement, alpha=0.8, edgecolor='black')
    ax6.set_xlabel('Technique', fontsize=10)
    ax6.set_ylabel('Improvement (%)', fontsize=10)
    ax6.set_title(' Improvement Over Baseline', fontsize=12, fontweight='bold')
    ax6.set_xticks(range(len(techniques)))
    ax6.set_xticklabels([_clean_technique_name(t)[:8] for t in techniques],
                       rotation=45, ha='right', fontsize=8)
    ax6.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add improvement values on bars
    for bar, imp in zip(bars6, improvements):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., 
                height + (0.5 if height > 0 else -1),
                f'{imp:+.1f}%', ha='center', 
                va='bottom' if height > 0 else 'top', fontsize=9)
    
    # 7. Cost Effectiveness (bottom row)
    ax7 = plt.subplot(3, 3, (8, 9))
    # Estimate API costs (Self-Consistency uses ~5x calls)
    costs = [5 if 'Self-Consistency' in t else 1 for t in techniques]
    cost_effectiveness = [acc/cost for acc, cost in zip(accuracies, costs)]
    
    bars7 = ax7.bar(range(len(techniques)), cost_effectiveness,
                    color='purple', alpha=0.7, edgecolor='black')
    ax7.set_xlabel('Prompting Technique', fontsize=12)
    ax7.set_ylabel('Cost Effectiveness (Accuracy % per API Call)', fontsize=12)
    ax7.set_title(' Cost Effectiveness Analysis', fontsize=14, fontweight='bold')
    ax7.set_xticks(range(len(techniques)))
    ax7.set_xticklabels([_clean_technique_name(t) for t in techniques],
                       rotation=45, ha='right')
    
    # Add cost effectiveness values
    for bar, ce in zip(bars7, cost_effectiveness):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{ce:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    else:
        plt.show()


def create_performance_heatmap(results: Dict[str, BenchmarkResult], 
                              save_path: Optional[str] = None) -> None:
    """
    Create a heatmap showing performance across different metrics.
    
    Args:
        results: Dictionary mapping technique names to BenchmarkResult objects
        save_path: Optional path to save the plot
    """
    
    if not PLOTTING_AVAILABLE:
        logger.warning("Plotting not available - install matplotlib and seaborn")
        return
    
    techniques = list(results.keys())
    metrics = ['Accuracy', 'Speed', 'Efficiency', 'Cost-Effectiveness']
    
    # Prepare and normalize metrics to 0-1 scale
    accuracies = [r.accuracy for r in results.values()]
    speeds = [1/r.avg_time if r.avg_time > 0 else 0 for r in results.values()]
    efficiencies = [r.accuracy/r.avg_time if r.avg_time > 0 else 0 for r in results.values()]
    
    # Estimate costs (Self-Consistency uses more API calls)
    costs = [5 if 'Self-Consistency' in t else 1 for t in techniques]
    cost_effectiveness = [acc/cost for acc, cost in zip(accuracies, costs)]
    
    # Normalize to 0-1 scale
    def normalize(values):
        max_val = max(values) if max(values) > 0 else 1
        return [v/max_val for v in values]
    
    data = np.array([
        normalize(accuracies),
        normalize(speeds),
        normalize(efficiencies),
        normalize(cost_effectiveness)
    ])
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(data,
                xticklabels=[_clean_technique_name(t) for t in techniques],
                yticklabels=metrics,
                annot=True,
                fmt='.2f',
                cmap='RdYlGn',
                center=0.5,
                cbar_kws={'label': 'Normalized Performance (0-1)'})
    
    plt.title('🌡️ Performance Heatmap Across All Metrics', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Prompting Techniques', fontsize=12)
    plt.ylabel('Performance Metrics', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Heatmap saved to {save_path}")
    else:
        plt.show()


def create_detailed_analysis_report(results: Dict[str, BenchmarkResult]) -> str:
    """
    Create a detailed text analysis report.
    
    Args:
        results: Dictionary mapping technique names to BenchmarkResult objects
        
    Returns:
        Formatted analysis report as string
    """
    
    report = []
    report.append("=" * 80)
    report.append(" DETAILED ANALYSIS REPORT")
    report.append("=" * 80)
    
    # Sort results by accuracy
    sorted_results = sorted(results.items(), key=lambda x: x[1].accuracy, reverse=True)
    
    # Overall summary
    best_name, best_result = sorted_results[0]
    worst_name, worst_result = sorted_results[-1]
    
    report.append(f"\n OVERALL SUMMARY")
    report.append("-" * 40)
    report.append(f"Best technique: {best_name} ({best_result.accuracy*100:.2f}%)")
    report.append(f"Worst technique: {worst_name} ({worst_result.accuracy*100:.2f}%)")
    report.append(f"Performance spread: {(best_result.accuracy - worst_result.accuracy)*100:.2f} percentage points")
    
    # Individual technique analysis
    report.append(f"\n TECHNIQUE ANALYSIS")
    report.append("-" * 40)
    
    for name, result in sorted_results:
        report.append(f"\n{name}:")
        report.append(f"  • Accuracy: {result.accuracy*100:.2f}% ({result.correct}/{result.total})")
        report.append(f"  • Average time: {result.avg_time:.2f} seconds")
        report.append(f"  • Error rate: {result.error_rate*100:.1f}%")
        
        efficiency = result.accuracy / result.avg_time if result.avg_time > 0 else 0
        report.append(f"  • Efficiency: {efficiency:.2f} accuracy points per second")
        
        # Performance characterization
        if result.accuracy > 0.7:
            perf_label = "Excellent"
        elif result.accuracy > 0.6:
            perf_label = "Good"
        elif result.accuracy > 0.5:
            perf_label = "Average"
        else:
            perf_label = "Poor"
        report.append(f"  • Performance: {perf_label}")
    
    # Comparative analysis
    report.append(f"\n COMPARATIVE INSIGHTS")
    report.append("-" * 40)
    
    # Find zero-shot baseline for comparisons
    zero_shot_result = None
    for name, result in results.items():
        if 'Zero-Shot' in name:
            zero_shot_result = result
            break
    
    if zero_shot_result:
        report.append(f"Improvements over Zero-Shot baseline:")
        for name, result in sorted_results:
            if 'Zero-Shot' not in name:
                improvement = (result.accuracy - zero_shot_result.accuracy) * 100
                report.append(f"  • {name}: {improvement:+.1f} percentage points")
    
    # Speed vs accuracy trade-offs
    report.append(f"\nSpeed vs Accuracy trade-offs:")
    for name, result in sorted_results:
        if result.avg_time > 5:
            speed_label = "Slow"
        elif result.avg_time > 2:
            speed_label = "Medium"
        else:
            speed_label = "Fast"
        report.append(f"  • {name}: {speed_label} ({result.avg_time:.1f}s) "
                     f"→ {result.accuracy*100:.1f}% accuracy")
    
    # Recommendations
    report.append(f"\n RECOMMENDATIONS")
    report.append("-" * 40)
    
    # Best overall
    report.append(f"1. For highest accuracy: Use {best_name}")
    
    # Best efficiency
    efficiencies = {name: r.accuracy/r.avg_time if r.avg_time > 0 else 0 
                   for name, r in results.items()}
    best_efficiency = max(efficiencies.items(), key=lambda x: x[1])
    report.append(f"2. For best efficiency: Use {best_efficiency[0]}")
    
    # Budget-conscious
    fast_techniques = [(name, r) for name, r in results.items() if r.avg_time < 3]
    if fast_techniques:
        best_fast = max(fast_techniques, key=lambda x: x[1].accuracy)
        report.append(f"3. For budget-conscious applications: Use {best_fast[0]}")
    
    return "\n".join(report)


def _clean_technique_name(name: str) -> str:
    """Clean technique name for display"""
    # Remove numbers and periods from the beginning
    cleaned = name
    if '. ' in cleaned:
        cleaned = cleaned.split('. ', 1)[1]
    return cleaned


def export_results_to_csv(results: Dict[str, BenchmarkResult], 
                         filename: str) -> None:
    """
    Export results to CSV file.
    
    Args:
        results: Dictionary mapping technique names to BenchmarkResult objects
        filename: Output CSV filename
    """
    
    df = create_results_table(results)
    df.to_csv(filename, index=False)
    logger.info(f"Results exported to {filename}")


def export_detailed_predictions(results: Dict[str, BenchmarkResult], 
                               filename: str) -> None:
    """
    Export detailed predictions to CSV file.
    
    Args:
        results: Dictionary mapping technique names to BenchmarkResult objects
        filename: Output CSV filename
    """
    
    detailed_data = []
    for technique_name, result in results.items():
        if hasattr(result, 'predictions') and result.predictions:
            for pred in result.predictions:
                row = {
                    'technique': technique_name,
                    'question': pred.get('question', ''),
                    'expected': pred.get('expected', ''),
                    'predicted': pred.get('predicted', ''),
                    'correct': pred.get('correct', False),
                    'reasoning': pred.get('reasoning', ''),
                    'question_length': pred.get('question_length', 0),
                }
                if 'confidence' in pred:
                    row['confidence'] = pred['confidence']
                detailed_data.append(row)
    
    if detailed_data:
        df = pd.DataFrame(detailed_data)
        df.to_csv(filename, index=False)
        logger.info(f"Detailed predictions exported to {filename}")
    else:
        logger.warning("No detailed predictions available for export")


if __name__ == "__main__":
    # Test visualization functions
    from .utils import BenchmarkResult
    
    # Create mock results for testing
    mock_results = {
        "Zero-Shot": BenchmarkResult("Zero-Shot", 0.45, 45, 100, 1.5, 0.05),
        "Few-Shot": BenchmarkResult("Few-Shot", 0.58, 58, 100, 1.8, 0.03),
        "Chain-of-Thought": BenchmarkResult("CoT", 0.65, 65, 100, 2.3, 0.02),
        "Self-Consistency": BenchmarkResult("Self-Consistency", 0.68, 68, 100, 8.5, 0.01),
        "Prolog-Style": BenchmarkResult("Prolog-Style", 0.71, 71, 100, 2.1, 0.02),
    }
    
    # Test table creation
    table = create_results_table(mock_results)
    print("Results table:")
    print(table.to_string(index=False))
    
    # Test analysis report
    report = create_detailed_analysis_report(mock_results)
    print(f"\n{report}")
    
    if PLOTTING_AVAILABLE:
        print("\nCreating test plots...")
        plot_results(mock_results)
        create_performance_heatmap(mock_results)
    else:
        print("\nPlotting not available - install matplotlib and seaborn")
