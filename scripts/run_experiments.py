#!/usr/bin/env python3
"""
Batch experiment runner for GSM8K benchmark evaluation.
"""

import os
import sys
import argparse
import logging
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gsm8k_bench import GSM8KBenchmark, load_gsm8k_dataset
from gsm8k_bench.viz import create_results_table, plot_results
from gsm8k_bench.utils import setup_dspy_with_config


def setup_logging(log_level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """Setup logging configuration."""
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


class ExperimentRunner:
    """Manages and executes batch experiments."""
    
    def __init__(self, output_dir: Path, log_level: str = "INFO"):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup experiment logging
        log_file = self.output_dir / "experiments.log"
        setup_logging(log_level, log_file)
        self.logger = logging.getLogger(__name__)
        
        # Track experiments
        self.experiment_history = []
        self.results_summary = {}
    
    def load_experiment_config(self, config_file: Path) -> Dict[str, Any]:
        """Load experiment configuration from YAML file."""
        self.logger.info(f"Loading experiment configuration from {config_file}")
        
        with open(config_file, 'r') as f:
            if config_file.suffix.lower() == '.yaml':
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
        
        return config
    
    def generate_experiment_grid(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate all experiment combinations from configuration."""
        grid_params = config.get("grid_search", {})
        base_config = config.get("base_config", {})
        
        # Extract parameter combinations
        param_names = list(grid_params.keys())
        param_values = list(grid_params.values())
        
        # Generate all combinations
        experiments = []
        for combination in itertools.product(*param_values):
            experiment = base_config.copy()
            
            # Update with current parameter combination
            for param_name, param_value in zip(param_names, combination):
                # Handle nested parameters (e.g., "model.temperature")
                keys = param_name.split('.')
                current = experiment
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[keys[-1]] = param_value
            
            experiments.append(experiment)
        
        self.logger.info(f"Generated {len(experiments)} experiment combinations")
        return experiments
    
    def run_single_experiment(self, experiment_config: Dict[str, Any], 
                             experiment_id: str) -> Dict[str, Any]:
        """Run a single experiment with given configuration."""
        self.logger.info(f"Starting experiment {experiment_id}")
        start_time = time.time()
        
        try:
            # Setup experiment directory
            exp_dir = self.output_dir / experiment_id
            exp_dir.mkdir(exist_ok=True)
            
            # Save experiment configuration
            config_file = exp_dir / "config.json"
            with open(config_file, 'w') as f:
                json.dump(experiment_config, f, indent=2)
            
            # Load dataset
            dataset_config = experiment_config.get("dataset", {})
            n_samples = dataset_config.get("n_samples", 50)
            dataset = load_gsm8k_dataset(n_samples=n_samples)
            
            # Configure model
            model_config = experiment_config.get("model", {})
            if not setup_dspy_with_config(model_config):
                raise RuntimeError("Failed to configure DSPy")
            
            # Setup benchmark
            benchmark_config = experiment_config.get("benchmark", {})
            selected_techniques = benchmark_config.get("techniques", None)
            
            benchmark = GSM8KBenchmark(
                dataset, 
                selected_techniques=selected_techniques,
                model_config=model_config
            )
            
            # Run evaluation
            results = benchmark.run_benchmark()
            
            # Save results
            results_data = {}
            for name, result in results.items():
                results_data[name] = {
                    'accuracy': result.accuracy,
                    'correct': result.correct,
                    'total': result.total,
                    'avg_time': result.avg_time,
                    'error_rate': result.error_rate
                }
            
            results_file = exp_dir / "results.json"
            with open(results_file, 'w') as f:
                json.dump(results_data, f, indent=2)
            
            # Create visualizations
            try:
                plot_results(results, save_path=str(exp_dir / "plots.png"))
                
                # Save results table
                results_table = create_results_table(results)
                results_table.to_csv(exp_dir / "results_table.csv", index=False)
                
            except Exception as viz_error:
                self.logger.warning(f"Visualization failed for {experiment_id}: {viz_error}")
            
            # Calculate experiment summary
            best_technique = max(results.items(), key=lambda x: x[1].accuracy)
            
            experiment_summary = {
                'experiment_id': experiment_id,
                'status': 'completed',
                'duration_seconds': time.time() - start_time,
                'best_technique': best_technique[0],
                'best_accuracy': best_technique[1].accuracy,
                'total_problems': best_technique[1].total,
                'config': experiment_config,
                'results_file': str(results_file)
            }
            
            self.logger.info(f"Completed experiment {experiment_id} - "
                           f"Best: {best_technique[0]} ({best_technique[1].accuracy*100:.1f}%)")
            
            return experiment_summary
            
        except Exception as e:
            error_summary = {
                'experiment_id': experiment_id,
                'status': 'failed',
                'duration_seconds': time.time() - start_time,
                'error': str(e),
                'config': experiment_config
            }
            
            self.logger.error(f"Experiment {experiment_id} failed: {e}")
            return error_summary
    
    def run_batch_experiments(self, experiments: List[Dict[str, Any]], 
                             max_workers: int = 1, 
                             experiment_prefix: str = "exp") -> List[Dict[str, Any]]:
        """Run batch of experiments with optional parallelization."""
        self.logger.info(f"Starting batch of {len(experiments)} experiments with {max_workers} workers")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = []
        
        if max_workers == 1:
            # Sequential execution
            for i, experiment_config in enumerate(experiments):
                experiment_id = f"{experiment_prefix}_{timestamp}_{i:03d}"
                result = self.run_single_experiment(experiment_config, experiment_id)
                results.append(result)
                self.experiment_history.append(result)
        else:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all experiments
                future_to_exp = {}
                for i, experiment_config in enumerate(experiments):
                    experiment_id = f"{experiment_prefix}_{timestamp}_{i:03d}"
                    future = executor.submit(self.run_single_experiment, experiment_config, experiment_id)
                    future_to_exp[future] = experiment_id
                
                # Collect results as they complete
                for future in as_completed(future_to_exp):
                    experiment_id = future_to_exp[future]
                    try:
                        result = future.result()
                        results.append(result)
                        self.experiment_history.append(result)
                    except Exception as e:
                        self.logger.error(f"Exception in experiment {experiment_id}: {e}")
        
        # Save batch summary
        batch_summary = {
            'timestamp': timestamp,
            'total_experiments': len(experiments),
            'completed': len([r for r in results if r['status'] == 'completed']),
            'failed': len([r for r in results if r['status'] == 'failed']),
            'results': results
        }
        
        summary_file = self.output_dir / f"batch_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(batch_summary, f, indent=2)
        
        self.logger.info(f"Batch experiments completed. Summary saved to {summary_file}")
        return results
    
    def generate_comparison_report(self, results: List[Dict[str, Any]], 
                                  output_file: Optional[Path] = None) -> str:
        """Generate comprehensive comparison report from batch results."""
        self.logger.info("Generating comparison report...")
        
        completed_results = [r for r in results if r['status'] == 'completed']
        
        if not completed_results:
            return "No completed experiments to analyze."
        
        report_lines = []
        report_lines.append("# Batch Experiment Comparison Report")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total Experiments: {len(results)}")
        report_lines.append(f"Completed: {len(completed_results)}")
        report_lines.append(f"Failed: {len(results) - len(completed_results)}")
        report_lines.append("")
        
        # Overall performance ranking
        report_lines.append("## Performance Ranking")
        sorted_results = sorted(completed_results, 
                              key=lambda x: x.get('best_accuracy', 0), 
                              reverse=True)
        
        for i, result in enumerate(sorted_results[:10], 1):
            accuracy = result.get('best_accuracy', 0) * 100
            technique = result.get('best_technique', 'Unknown')
            exp_id = result.get('experiment_id', 'Unknown')
            
            report_lines.append(f"{i}. **{exp_id}**: {accuracy:.2f}% ({technique})")
        
        report_lines.append("")
        
        # Parameter analysis
        report_lines.append("## Parameter Impact Analysis")
        
        # Group by varying parameters
        param_analysis = {}
        for result in completed_results:
            config = result.get('config', {})
            
            # Extract key parameters for analysis
            model_name = config.get('model', {}).get('name', 'unknown')
            sample_size = config.get('dataset', {}).get('n_samples', 0)
            techniques = config.get('benchmark', {}).get('techniques', [])
            
            key = f"model={model_name}, samples={sample_size}, techniques={len(techniques)}"
            
            if key not in param_analysis:
                param_analysis[key] = []
            
            param_analysis[key].append({
                'accuracy': result.get('best_accuracy', 0),
                'experiment_id': result.get('experiment_id', '')
            })
        
        for param_combo, exp_results in param_analysis.items():
            if len(exp_results) > 1:  # Only show if multiple experiments
                avg_accuracy = sum(r['accuracy'] for r in exp_results) / len(exp_results)
                report_lines.append(f"**{param_combo}**: {avg_accuracy*100:.2f}% avg ({len(exp_results)} experiments)")
        
        report_lines.append("")
        
        # Timing analysis
        report_lines.append("## Timing Analysis")
        durations = [r.get('duration_seconds', 0) for r in completed_results]
        
        if durations:
            import statistics
            report_lines.append(f"- Average duration: {statistics.mean(durations):.1f} seconds")
            report_lines.append(f"- Median duration: {statistics.median(durations):.1f} seconds")
            report_lines.append(f"- Min duration: {min(durations):.1f} seconds")
            report_lines.append(f"- Max duration: {max(durations):.1f} seconds")
        
        report_lines.append("")
        
        # Failed experiments analysis
        failed_results = [r for r in results if r['status'] == 'failed']
        if failed_results:
            report_lines.append("## Failed Experiments")
            
            error_types = {}
            for result in failed_results:
                error = result.get('error', 'Unknown error')
                error_type = error.split(':')[0] if ':' in error else error
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            for error_type, count in error_types.items():
                report_lines.append(f"- **{error_type}**: {count} experiments")
            
            report_lines.append("")
        
        # Recommendations
        report_lines.append("## Recommendations")
        
        if completed_results:
            best_result = max(completed_results, key=lambda x: x.get('best_accuracy', 0))
            report_lines.append(f"1. **Best Configuration**: {best_result['experiment_id']}")
            report_lines.append(f"   - Accuracy: {best_result.get('best_accuracy', 0)*100:.2f}%")
            report_lines.append(f"   - Technique: {best_result.get('best_technique', 'Unknown')}")
            
            # Find fastest experiment with good accuracy
            good_results = [r for r in completed_results if r.get('best_accuracy', 0) > 0.6]
            if good_results:
                fastest_good = min(good_results, key=lambda x: x.get('duration_seconds', float('inf')))
                report_lines.append(f"2. **Fastest Good Result**: {fastest_good['experiment_id']}")
                report_lines.append(f"   - Accuracy: {fastest_good.get('best_accuracy', 0)*100:.2f}%")
                report_lines.append(f"   - Duration: {fastest_good.get('duration_seconds
