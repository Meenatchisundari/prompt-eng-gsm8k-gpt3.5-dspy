import click
import os
import sys
import yaml
from pathlib import Path
from typing import List, Optional

from .benchmark import GSM8KBenchmark
from .data import load_gsm8k_dataset
from .viz import create_results_table, plot_results


@click.group()
@click.version_option()
def cli():
    """GSM8K Prompting Techniques Benchmark with DSPy"""
    pass


@cli.command()
@click.option('--samples', '-n', default=20, help='Number of problems to test')
@click.option('--techniques', '-t', default='all', 
              help='Comma-separated list of techniques to run')
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Path to configuration file')
@click.option('--output', '-o', default='results',
              help='Output directory for results')
@click.option('--model', default='gpt-3.5-turbo',
              help='OpenAI model to use')
@click.option('--temperature', default=0.0, type=float,
              help='Temperature for model generation')
@click.option('--max-tokens', default=1000, type=int,
              help='Maximum tokens for generation')
@click.option('--visualize/--no-visualize', default=True,
              help='Create visualizations')
@click.option('--save-detailed/--no-save-detailed', default=False,
              help='Save detailed prediction results')
def run(samples: int, techniques: str, config: Optional[str], output: str,
        model: str, temperature: float, max_tokens: int, 
        visualize: bool, save_detailed: bool):
    """Run the GSM8K benchmark"""
    
    click.echo(" Starting GSM8K Benchmark...")
    
    # Check API key
    if not os.getenv('OPENAI_API_KEY'):
        click.echo(" Error: OPENAI_API_KEY environment variable not set")
        click.echo("Set it with: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)
    
    # Load configuration
    if config:
        with open(config, 'r') as f:
            config_data = yaml.safe_load(f)
        click.echo(f" Loaded config from {config}")
    else:
        config_data = {}
    
    # Override with CLI arguments
    config_data.setdefault('model', {})
    config_data['model']['name'] = model
    config_data['model']['temperature'] = temperature
    config_data['model']['max_tokens'] = max_tokens
    
    config_data.setdefault('benchmark', {})
    config_data['benchmark']['n_samples'] = samples
    
    # Parse techniques
    if techniques == 'all':
        selected_techniques = None  # Use all available
    else:
        selected_techniques = [t.strip() for t in techniques.split(',')]
    
    # Load dataset
    click.echo(f" Loading {samples} GSM8K problems...")
    try:
        test_dataset = load_gsm8k_dataset(n_samples=samples)
        click.echo(f" Loaded {len(test_dataset)} problems")
    except Exception as e:
        click.echo(f" Error loading dataset: {e}")
        sys.exit(1)
    
    # Create benchmark
    try:
        benchmark = GSM8KBenchmark(
            test_dataset, 
            selected_techniques=selected_techniques,
            model_config=config_data.get('model', {})
        )
        click.echo(f" Initialized benchmark with {len(benchmark.modules)} techniques")
    except Exception as e:
        click.echo(f" Error creating benchmark: {e}")
        sys.exit(1)
    
    # Run benchmark
    try:
        results = benchmark.run_benchmark()
        click.echo(" Benchmark completed!")
    except Exception as e:
        click.echo(f" Error running benchmark: {e}")
        sys.exit(1)
    
    # Create output directory
    output_path = Path(output)
    output_path.mkdir(exist_ok=True)
    
    # Save results
    results_table = create_results_table(results)
    results_file = output_path / 'benchmark_results.csv'
    results_table.to_csv(results_file, index=False)
    click.echo(f" Results saved to {results_file}")
    
    # Display summary
    click.echo("\n" + "=" * 60)
    click.echo(" BENCHMARK RESULTS SUMMARY")
    click.echo("=" * 60)
    click.echo(results_table.to_string(index=False))
    
    # Find best technique
    best_technique = max(results.items(), key=lambda x: x[1].accuracy)
    click.echo(f"\n Best technique: {best_technique[0]} "
               f"({best_technique[1].accuracy*100:.2f}%)")
    
    # Create visualizations
    if visualize:
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend for CLI
            
            plot_file = output_path / 'benchmark_plots.png'
            plot_results(results, save_path=str(plot_file))
            click.echo(f" Visualizations saved to {plot_file}")
        except ImportError:
            click.echo(" Matplotlib not available, skipping visualizations")
        except Exception as e:
            click.echo(f" Error creating visualizations: {e}")
    
    # Save detailed results
    if save_detailed:
        detailed_file = output_path / 'detailed_results.json'
        detailed_data = {}
        for name, result in results.items():
            detailed_data[name] = {
                'accuracy': result.accuracy,
                'correct': result.correct,
                'total': result.total,
                'avg_time': result.avg_time,
                'error_rate': result.error_rate,
                'predictions': getattr(result, 'predictions', [])
            }
        
        import json
        with open(detailed_file, 'w') as f:
            json.dump(detailed_data, f, indent=2)
        click.echo(f" Detailed results saved to {detailed_file}")


@cli.command()
@click.option('--input', '-i', required=True, type=click.Path(exists=True),
              help='Input results file (CSV or JSON)')
@click.option('--output', '-o', default='analysis_report.html',
              help='Output analysis report file')
def analyze(input: str, output: str):
    """Analyze existing benchmark results"""
    
    click.echo(f" Analyzing results from {input}")
    
    # Load results
    input_path = Path(input)
    if input_path.suffix == '.csv':
        import pandas as pd
        df = pd.read_csv(input_path)
        click.echo(f" Loaded {len(df)} technique results")
    elif input_path.suffix == '.json':
        import json
        with open(input_path, 'r') as f:
            data = json.load(f)
        click.echo(f" Loaded {len(data)} technique results")
    else:
        click.echo(" Error: Input file must be CSV or JSON")
        sys.exit(1)
    
    # TODO: Implement detailed analysis
    click.echo("🔍 Detailed analysis coming soon...")


@cli.command()
@click.option('--samples', '-n', default=100, help='Training samples')
@click.option('--model', default='distilgpt2', help='Model to fine-tune')
@click.option('--epochs', default=2, help='Training epochs')
@click.option('--output', '-o', default='finetuned_model',
              help='Output directory')
def finetune(samples: int, model: str, epochs: int, output: str):
    """Fine-tune a model on GSM8K with comparison to prompting"""
    
    click.echo(" Starting fine-tuning comparison...")
    
    try:
        from .finetune.compare import run_finetuning_vs_prompting_comparison
        
        results = run_finetuning_vs_prompting_comparison(
            train_samples=samples,
            model_name=model,
            num_epochs=epochs,
            output_dir=output
        )
        
        click.echo(" Fine-tuning comparison completed!")
        click.echo(f" Results: {results}")
        
    except ImportError:
        click.echo(" Fine-tuning dependencies not installed")
        click.echo("Install with: pip install transformers torch accelerate peft")
        sys.exit(1)
    except Exception as e:
        click.echo(f" Error during fine-tuning: {e}")
        sys.exit(1)


@cli.command()
@click.option('--technique1', required=True, help='First technique to compare')
@click.option('--technique2', required=True, help='Second technique to compare')
@click.option('--samples', '-n', default=50, help='Number of problems per group')
@click.option('--alpha', default=0.05, help='Significance level for statistical test')
def compare(technique1: str, technique2: str, samples: int, alpha: float):
    """Run A/B test between two techniques"""
    
    click.echo(f"🧪 A/B Testing: {technique1} vs {technique2}")
    
    try:
        from .benchmark import run_ab_test
        
        # Load dataset
        dataset = load_gsm8k_dataset(n_samples=samples * 2)
        
        # Run A/B test
        result = run_ab_test(
            technique1, technique2, dataset, 
            test_size=samples, alpha=alpha
        )
        
        if result:
            click.echo(f"\n📊 Results:")
            click.echo(f"  {technique1}: {result['accuracy_a']*100:.1f}%")
            click.echo(f"  {technique2}: {result['accuracy_b']*100:.1f}%")
            click.echo(f"  P-value: {result['p_value']:.4f}")
            
            if result['is_significant']:
                winner = technique2 if result['accuracy_b'] > result['accuracy_a'] else technique1
                click.echo(f"  Winner: {winner} ")
            else:
                click.echo(f"  Result: No significant difference ")
        
    except Exception as e:
        click.echo(f" Error during comparison: {e}")
        sys.exit(1)


@cli.command()
def list_techniques():
    """List all available prompting techniques"""
    
    click.echo("🔧 Available Prompting Techniques:")
    click.echo("")
    
    techniques = [
        ("zero_shot", "Zero-Shot prompting - direct problem solving"),
        ("few_shot", "Few-Shot prompting - learning from examples"),
        ("cot", "Chain-of-Thought - step-by-step reasoning"),
        ("self_consistency", "Self-Consistency - multiple paths with voting"),
        ("prolog_style", "Prolog-Style - logical reasoning with facts/rules"),
    ]
    
    for name, description in techniques:
        click.echo(f"  • {name:<15} - {description}")
    
    click.echo("")
    click.echo("Advanced techniques (in improvements/):")
    
    advanced = [
        ("enhanced_prolog", "Enhanced Prolog with verification"),
        ("calculator_augmented", "Calculator-augmented reasoning"),
        ("verification_chain", "Multi-step verification chain"),
        ("ensemble", "Weighted ensemble of techniques"),
    ]
    
    for name, description in advanced:
        click.echo(f"  • {name:<15} - {description}")


if __name__ == '__main__':
    cli()
