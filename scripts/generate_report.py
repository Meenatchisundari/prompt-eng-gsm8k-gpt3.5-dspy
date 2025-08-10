"""
Comprehensive report generation script for GSM8K benchmark results.
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gsm8k_bench.viz import create_results_table, create_detailed_analysis_report
from gsm8k_bench.utils import BenchmarkResult, get_error_analysis


class ReportGenerator:
    """Comprehensive report generator for benchmark results."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_results(self, results_file: Path) -> Dict[str, BenchmarkResult]:
        """Load benchmark results from JSON file."""
        self.logger.info(f"Loading results from {results_file}")
        
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        # Convert to BenchmarkResult objects
        results = {}
        for name, result_data in data.items():
            results[name] = BenchmarkResult(
                technique=result_data.get('technique', name),
                accuracy=result_data.get('accuracy', 0.0),
                correct=result_data.get('correct', 0),
                total=result_data.get('total', 0),
                avg_time=result_data.get('avg_time', 0.0),
                error_rate=result_data.get('error_rate', 0.0),
                predictions=result_data.get('predictions', [])
            )
        
        return results
    
    def generate_executive_summary(self, results: Dict[str, BenchmarkResult]) -> str:
        """Generate executive summary section."""
        lines = []
        lines.append("# Executive Summary")
        lines.append("")
        
        # Key findings
        sorted_results = sorted(results.items(), key=lambda x: x[1].accuracy, reverse=True)
        best_name, best_result = sorted_results[0]
        worst_name, worst_result = sorted_results[-1]
        
        improvement = (best_result.accuracy - worst_result.accuracy) * 100
        
        lines.append("## Key Findings")
        lines.append("")
        lines.append(f"- **Best Performance**: {best_name} achieved {best_result.accuracy*100:.1f}% accuracy")
        lines.append(f"- **Performance Range**: {improvement:.1f} percentage point difference between best and worst")
        lines.append(f"- **Total Problems Evaluated**: {best_result.total:,}")
        lines.append(f"- **Evaluation Date**: {datetime.now().strftime('%B %d, %Y')}")
        lines.append("")
        
        # Recommendations
        lines.append("## Recommendations")
        lines.append("")
        
        # Find most efficient technique
        efficiencies = {name: r.accuracy / r.avg_time if r.avg_time > 0 else 0 
                       for name, r in results.items()}
        most_efficient = max(efficiencies.items(), key=lambda x: x[1])[0]
        
        lines.append(f"1. **For Production Deployment**: Use {best_name} for highest accuracy")
        lines.append(f"2. **For Cost Efficiency**: Use {most_efficient} for best accuracy/time ratio")
        lines.append(f"3. **For Baseline Comparison**: {worst_name} provides minimum viable performance")
        lines.append("")
        
        return "\n".join(lines)
    
    def generate_methodology_section(self, results: Dict[str, BenchmarkResult]) -> str:
        """Generate methodology section."""
        lines = []
        lines.append("# Methodology")
        lines.append("")
        
        # Get sample size from first result
        sample_size = list(results.values())[0].total
        
        lines.append("## Experimental Setup")
        lines.append("")
        lines.append(f"- **Dataset**: GSM8K grade school math problems")
        lines.append(f"- **Sample Size**: {sample_size:,} problems")
        lines.append(f"- **Model**: GPT-3.5-turbo")
        lines.append(f"- **Temperature**: 0.0 (deterministic)")
        lines.append(f"- **Evaluation Metric**: Exact match accuracy")
        lines.append(f"- **Numerical Tolerance**: ±0.01")
        lines.append("")
        
        lines.append("## Techniques Evaluated")
        lines.append("")
        for i, (name, result) in enumerate(results.items(), 1):
            clean_name = name.replace('1. ', '').replace('2. ', '').replace('3. ', '').replace('4. ', '').replace('5. ', '')
            lines.append(f"{i}. **{clean_name}**: {self._get_technique_description(clean_name)}")
        lines.append("")
        
        return "\n".join(lines)
    
    def _get_technique_description(self, technique_name: str) -> str:
        """Get description for a technique."""
        descriptions = {
            "Zero-Shot": "Direct problem solving without examples",
            "Few-Shot": "Learning from 4 provided examples",
            "Chain-of-Thought": "Step-by-step reasoning process",
            "Self-Consistency": "Multiple reasoning paths with majority voting",
            "Prolog-Style": "Structured logical reasoning with facts, rules, and derivation"
        }
        return descriptions.get(technique_name, "Advanced prompting technique")
    
    def generate_results_section(self, results: Dict[str, BenchmarkResult]) -> str:
        """Generate detailed results section."""
        lines = []
        lines.append("# Results")
        lines.append("")
        
        # Performance table
        lines.append("## Performance Summary")
        lines.append("")
        
        # Create formatted table
        table_data = []
        for name, result in results.items():
            clean_name = name.replace('1. ', '').replace('2. ', '').replace('3. ', '').replace('4. ', '').replace('5. ', '')
            efficiency = result.accuracy / result.avg_time if result.avg_time > 0 else 0
            
            table_data.append({
                'Technique': clean_name,
                'Accuracy (%)': f"{result.accuracy*100:.1f}%",
                'Correct/Total': f"{result.correct}/{result.total}",
                'Avg Time (s)': f"{result.avg_time:.2f}",
                'Efficiency': f"{efficiency:.1f}",
                'Error Rate (%)': f"{result.error_rate*100:.1f}%"
            })
        
        # Sort by accuracy
        table_data.sort(key=lambda x: float(x['Accuracy (%)'].rstrip('%')), reverse=True)
        
        # Create markdown table
        headers = list(table_data[0].keys())
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        
        for row in table_data:
            lines.append("| " + " | ".join(row.values()) + " |")
        lines.append("")
        
        # Statistical significance
        lines.append("## Statistical Analysis")
        lines.append("")
        lines.append("### Confidence Intervals (95%)")
        lines.append("")
        
        for name, result in results.items():
            clean_name = name.replace('1. ', '').replace('2. ', '').replace('3. ', '').replace('4. ', '').replace('5. ', '')
            
            # Calculate binomial confidence interval
            if result.total > 0:
                from scipy.stats import binom
                ci_lower = binom.ppf(0.025, result.total, result.accuracy) / result.total
                ci_upper = binom.ppf(0.975, result.total, result.accuracy) / result.total
                
                ci_lower = max(0, ci_lower)
                ci_upper = min(1, ci_upper)
                
                lines.append(f"- **{clean_name}**: {result.accuracy*100:.1f}% "
                           f"[{ci_lower*100:.1f}%, {ci_upper*100:.1f}%]")
        
        lines.append("")
        
        return "\n".join(lines)
    
    def generate_analysis_section(self, results: Dict[str, BenchmarkResult]) -> str:
        """Generate detailed analysis section."""
        lines = []
        lines.append("# Detailed Analysis")
        lines.append("")
        
        # Performance insights
        lines.append("## Performance Insights")
        lines.append("")
        
        sorted_results = sorted(results.items(), key=lambda x: x[1].accuracy, reverse=True)
        
        # Accuracy analysis
        accuracies = [r.accuracy for _, r in sorted_results]
        accuracy_std = np.std(accuracies)
        accuracy_range = max(accuracies) - min(accuracies)
        
        lines.append(f"- **Performance Spread**: {accuracy_range*100:.1f} percentage points between best and worst")
        lines.append(f"- **Standard Deviation**: {accuracy_std*100:.1f} percentage points")
        lines.append("")
        
        # Top performers
        lines.append("### Top Performing Techniques")
        lines.append("")
        
        for i, (name, result) in enumerate(sorted_results[:3], 1):
            clean_name = name.replace('1. ', '').replace('2. ', '').replace('3. ', '').replace('4. ', '').replace('5. ', '')
            lines.append(f"{i}. **{clean_name}**: {result.accuracy*100:.1f}% accuracy "
                        f"({result.correct}/{result.total} correct)")
        lines.append("")
        
        # Cost-effectiveness analysis
        lines.append("## Cost-Effectiveness Analysis")
        lines.append("")
        
        # Estimate relative costs
        cost_factors = {
            "Zero-Shot": 1.0,
            "Few-Shot": 1.2,
            "Chain-of-Thought": 1.5,
            "Self-Consistency": 5.0,
            "Prolog-Style": 1.8,
        }
        
        cost_effectiveness = []
        for name, result in results.items():
            clean_name = name.replace('1. ', '').replace('2. ', '').replace('3. ', '').replace('4. ', '').replace('5. ', '')
            cost_factor = cost_factors.get(clean_name, 1.0)
            effectiveness = (result.accuracy * 100) / cost_factor
            cost_effectiveness.append((clean_name, effectiveness, cost_factor))
        
        # Sort by cost-effectiveness
        cost_effectiveness.sort(key=lambda x: x[1], reverse=True)
        
        lines.append("### Cost-Effectiveness Ranking")
        lines.append("")
        for i, (name, effectiveness, cost) in enumerate(cost_effectiveness, 1):
            lines.append(f"{i}. **{name}**: {effectiveness:.1f} accuracy points per cost unit "
                        f"(relative cost: {cost:.1f}x)")
        lines.append("")
        
        return "\n".join(lines)
    
    def generate_error_analysis(self, results: Dict[str, BenchmarkResult]) -> str:
        """Generate error analysis section."""
        lines = []
        lines.append("# Error Analysis")
        lines.append("")
        
        # Find best performing technique for detailed analysis
        best_technique = max(results.items(), key=lambda x: x[1].accuracy)
        best_name, best_result = best_technique
        
        lines.append(f"## Detailed Analysis: {best_name}")
        lines.append("")
        
        if hasattr(best_result, 'predictions') and best_result.predictions:
            try:
                error_analysis = get_error_analysis(best_result.predictions)
                
                lines.append(f"### Error Statistics")
                lines.append(f"- **Total Errors**: {error_analysis['total_errors']}")
                lines.append(f"- **Error Rate**: {error_analysis['error_rate']*100:.1f}%")
                lines.append("")
                
                lines.append("### Error Categories")
                lines.append("")
                for pattern, count in error_analysis['error_patterns'].items():
                    if count > 0:
                        percentage = (count / error_analysis['total_errors']) * 100
                        pattern_name = pattern.replace('_', ' ').title()
                        lines.append(f"- **{pattern_name}**: {count} ({percentage:.1f}%)")
                
                lines.append("")
                
                if error_analysis['common_issues']:
                    lines.append("### Common Issues")
                    lines.append("")
                    for issue in error_analysis['common_issues']:
                        lines.append(f"- {issue}")
                    lines.append("")
                
            except Exception as e:
                lines.append(f"Error analysis could not be completed: {e}")
                lines.append("")
        else:
            lines.append("Detailed prediction data not available for error analysis.")
            lines.append("")
        
        return "\n".join(lines)
    
    def generate_conclusions_section(self, results: Dict[str, BenchmarkResult]) -> str:
        """Generate conclusions and future work section."""
        lines = []
        lines.append("# Conclusions and Future Work")
        lines.append("")
        
        # Key conclusions
        lines.append("## Key Conclusions")
        lines.append("")
        
        sorted_results = sorted(results.items(), key=lambda x: x[1].accuracy, reverse=True)
        best_name, best_result = sorted_results[0]
        
        lines.append(f"1. **{best_name} achieves the highest accuracy** at {best_result.accuracy*100:.1f}%, "
                    "demonstrating the effectiveness of structured reasoning approaches.")
        lines.append("")
        
        # Find zero-shot for comparison
        zero_shot_result = None
        for name, result in results.items():
            if 'Zero-Shot' in name:
                zero_shot_result = result
                break
        
        if zero_shot_result:
            improvement = (best_result.accuracy - zero_shot_result.accuracy) * 100
            lines.append(f"2. **Prompting techniques provide substantial improvements** over baseline, "
                        f"with up to {improvement:.1f} percentage point gains.")
            lines.append("")
        
        lines.append("3. **Cost-effectiveness varies significantly** across techniques, with some "
                    "providing better accuracy per computational cost than others.")
        lines.append("")
        
        # Future work
        lines.append("## Future Work")
        lines.append("")
        lines.append("### Immediate Improvements")
        lines.append("- Evaluate on larger datasets (1000+ problems)")
        lines.append("- Test with different language models (GPT-4, Claude, etc.)")
        lines.append("- Implement hybrid techniques combining multiple approaches")
        lines.append("- Add confidence estimation for predictions")
        lines.append("")
        
        lines.append("### Research Directions")
        lines.append("- Study technique performance across problem difficulty levels")
        lines.append("- Investigate adaptive technique selection based on problem type")
        lines.append("- Explore few-shot example selection strategies")
        lines.append("- Develop meta-learning approaches for prompt optimization")
        lines.append("")
        
        lines.append("### Technical Enhancements")
        lines.append("- Implement tool-augmented reasoning (calculator, web search)")
        lines.append("- Add multi-modal capabilities for diagram-based problems")
        lines.append("- Develop real-time evaluation and monitoring systems")
        lines.append("- Create educational applications for math reasoning")
        lines.append("")
        
        return "\n".join(lines)
    
    def generate_appendix(self, results: Dict[str, BenchmarkResult]) -> str:
        """Generate appendix with technical details."""
        lines = []
        lines.append("# Appendix")
        lines.append("")
        
        # Technical specifications
        lines.append("## Technical Specifications")
        lines.append("")
        lines.append("### Model Configuration")
        lines.append("```yaml")
        lines.append("model: gpt-3.5-turbo")
        lines.append("temperature: 0.0")
        lines.append("max_tokens: 1000")
        lines.append("top_p: 1.0")
        lines.append("frequency_penalty: 0.0")
        lines.append("presence_penalty: 0.0")
        lines.append("```")
        lines.append("")
        
        lines.append("### Evaluation Parameters")
        lines.append("```yaml")
        lines.append("accuracy_threshold: 0.01")
        lines.append("timeout_seconds: 30")
        lines.append("retry_attempts: 3")
        lines.append("confidence_level: 0.95")
        lines.append("```")
        lines.append("")
        
        # Raw data summary
        lines.append("## Raw Data Summary")
        lines.append("")
        
        total_predictions = sum(r.total for r in results.values())
        total_time = sum(r.avg_time * r.total for r in results.values())
        
        lines.append(f"- **Total Predictions**: {total_predictions:,}")
        lines.append(f"- **Total Evaluation Time**: {total_time/60:.1f} minutes")
        lines.append(f"- **Average Time per Problem**: {total_time/total_predictions:.2f} seconds")
        lines.append("")
        
        # Environment information
        lines.append("## Environment Information")
        lines.append("")
        lines.append(f"- **Evaluation Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"- **Python Version**: {sys.version.split()[0]}")
        lines.append(f"- **Platform**: {sys.platform}")
        lines.append("")
        
        return "\n".join(lines)
    
    def generate_full_report(self, results: Dict[str, BenchmarkResult], 
                           title: str = "GSM8K Benchmark Evaluation Report") -> str:
        """Generate complete comprehensive report."""
        self.logger.info("Generating comprehensive report...")
        
        sections = []
        
        # Title and table of contents
        sections.append(f"# {title}")
        sections.append("")
        sections.append("---")
        sections.append("")
        
        # Executive summary
        sections.append(self.generate_executive_summary(results))
        sections.append("")
        
        # Methodology
        sections.append(self.generate_methodology_section(results))
        sections.append("")
        
        # Results
        sections.append(self.generate_results_section(results))
        sections.append("")
        
        # Analysis
        sections.append(self.generate_analysis_section(results))
        sections.append("")
        
        # Error analysis
        sections.append(self.generate_error_analysis(results))
        sections.append("")
        
        # Conclusions
        sections.append(self.generate_conclusions_section(results))
        sections.append("")
        
        # Appendix
        sections.append(self.generate_appendix(results))
        
        return "\n".join(sections)
    
    def export_to_formats(self, report_content: str, base_filename: str) -> List[Path]:
        """Export report to multiple formats."""
        exported_files = []
        
        # Markdown
        md_file = self.output_dir / f"{base_filename}.md"
        with open(md_file, 'w') as f:
            f.write(report_content)
        exported_files.append(md_file)
        self.logger.info(f"Markdown report saved to {md_file}")
        
        # Try to convert to other formats if dependencies available
        try:
            import markdown
            import pdfkit
            
            # HTML conversion
            html_content = markdown.markdown(report_content, extensions=['tables'])
            html_file = self.output_dir / f"{base_filename}.html"
            
            html_template = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>GSM8K Benchmark Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    code {{ background-color: #f4f4f4; padding: 2px 4px; border-radius: 3px; }}
                    pre {{ background-color: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }}
                </style>
            </head>
            <body>
                {html_content}
            </body>
            </html>
            """
            
            with open(html_file, 'w') as f:
                f.write(html_template)
            exported_files.append(html_file)
            self.logger.info(f"HTML report saved to {html_file}")
            
            # PDF conversion (requires wkhtmltopdf)
            try:
                pdf_file = self.output_dir / f"{base_filename}.pdf"
                pdfkit.from_file(str(html_file), str(pdf_file))
                exported_files.append(pdf_file)
                self.logger.info(f"PDF report saved to {pdf_file}")
            except Exception as e:
                self.logger.warning(f"PDF conversion failed: {e}")
                
        except ImportError:
            self.logger.info("Optional dependencies not available for HTML/PDF export")
        
        return exported_files


def main():
    """Main function for report generation script."""
    parser = argparse.ArgumentParser(description="Generate comprehensive GSM8K benchmark report")
    parser.add_argument("--results", type=Path, required=True,
                       help="Path to benchmark results JSON file")
    parser.add_argument("--output-dir", type=Path, default=Path("reports"),
                       help="Output directory for generated reports")
    parser.add_argument("--title", default="GSM8K Benchmark Evaluation Report",
                       help="Report title")
    parser.add_argument("--format", choices=["markdown", "html", "pdf", "all"],
                       default="all", help="Output format(s)")
    parser.add_argument("--filename", default="gsm8k_report",
                       help="Base filename for reports")
    
    args = parser.parse_args()
    
    try:
        # Initialize report generator
        generator = ReportGenerator(args.output_dir)
        
        # Load results
        results = generator.load_results(args.results)
        
        if not results:
            print("No results found in the specified file.")
            return
        
        print(f"Loaded results for {len(results)} techniques")
        
        # Generate comprehensive report
        report_content = generator.generate_full_report(results, args.title)
        
        # Export to requested formats
        if args.format == "all":
            exported_files = generator.export_to_formats(report_content, args.filename)
        else:
            # Export only requested format
            if args.format == "markdown":
                md_file = args.output_dir / f"{args.filename}.md"
                with open(md_file, 'w') as f:
                    f.write(report_content)
                exported_files = [md_file]
            else:
                # For HTML/PDF, need to export markdown first then convert
                exported_files = generator.export_to_formats(report_content, args.filename)
                if args.format == "html":
                    exported_files = [f for f in exported_files if f.suffix == '.html']
                elif args.format == "pdf":
                    exported_files = [f for f in exported_files if f.suffix == '.pdf']
        
        # Print summary
        print("\n" + "="*60)
        print("REPORT GENERATION COMPLETED")
        print("="*60)
        print(f"Report title: {args.title}")
        print(f"Source data: {args.results}")
        print(f"Techniques analyzed: {len(results)}")
        
        best_technique = max(results.items(), key=lambda x: x[1].accuracy)
        print(f"Best performing technique: {best_technique[0]} ({best_technique[1].accuracy*100:.1f}%)")
        
        print(f"\nGenerated files:")
        for file_path in exported_files:
            print(f"  - {file_path}")
        
        print("="*60)
        
    except Exception as e:
        logging.error(f"Error generating report: {e}")
        raise


if __name__ == "__main__":
    main()
