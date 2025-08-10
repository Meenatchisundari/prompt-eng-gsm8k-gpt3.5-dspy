#!/usr/bin/env python3
"""
Data download and preparation script for GSM8K benchmark.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import json
import hashlib

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datasets import load_dataset
import pandas as pd
from gsm8k_bench.data import extract_numerical_answer, categorize_problem


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('data_download.log'),
            logging.StreamHandler()
        ]
    )


def download_gsm8k_dataset(output_dir: Path, splits: List[str] = ["train", "test"]) -> Dict[str, Any]:
    """
    Download and process GSM8K dataset from HuggingFace.
    
    Args:
        output_dir: Directory to save processed data
        splits: Dataset splits to download
        
    Returns:
        Dictionary with dataset statistics
    """
    logging.info("Downloading GSM8K dataset from HuggingFace...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    
    for split in splits:
        logging.info(f"Processing {split} split...")
        
        # Load dataset
        dataset = load_dataset("gsm8k", "main", split=split)
        
        # Process and save
        processed_data = []
        categories_count = {}
        
        for i, item in enumerate(dataset):
            # Extract numerical answer
            numerical_answer = extract_numerical_answer(item["answer"])
            
            # Categorize problem
            categories = categorize_problem(item["question"])
            
            # Count categories
            for category in categories:
                categories_count[category] = categories_count.get(category, 0) + 1
            
            # Create processed item
            processed_item = {
                "id": f"{split}_{i:05d}",
                "question": item["question"],
                "full_answer": item["answer"],
                "numerical_answer": numerical_answer,
                "categories": categories,
                "word_count": len(item["question"].split()),
                "number_count": len([w for w in item["question"].split() if any(c.isdigit() for c in w)])
            }
            
            processed_data.append(processed_item)
            
            if (i + 1) % 100 == 0:
                logging.info(f"Processed {i + 1} problems...")
        
        # Save processed data
        output_file = output_dir / f"gsm8k_{split}.json"
        with open(output_file, 'w') as f:
            json.dump(processed_data, f, indent=2)
        
        # Save CSV version for easy viewing
        df = pd.DataFrame(processed_data)
        df.to_csv(output_dir / f"gsm8k_{split}.csv", index=False)
        
        # Calculate statistics
        stats[split] = {
            "total_problems": len(processed_data),
            "avg_word_count": df["word_count"].mean(),
            "avg_number_count": df["number_count"].mean(),
            "categories": categories_count,
            "file_path": str(output_file)
        }
        
        logging.info(f"Saved {len(processed_data)} {split} problems to {output_file}")
    
    return stats


def create_sample_datasets(input_dir: Path, output_dir: Path, 
                          sample_sizes: List[int] = [10, 50, 100, 200]) -> None:
    """
    Create sample datasets for quick testing.
    
    Args:
        input_dir: Directory with full datasets
        output_dir: Directory to save sample datasets
        sample_sizes: List of sample sizes to create
    """
    logging.info("Creating sample datasets...")
    
    # Load test set
    test_file = input_dir / "gsm8k_test.json"
    with open(test_file, 'r') as f:
        test_data = json.load(f)
    
    # Create stratified samples
    import random
    random.seed(42)  # For reproducibility
    
    for size in sample_sizes:
        if size > len(test_data):
            logging.warning(f"Sample size {size} larger than dataset, skipping...")
            continue
        
        # Stratified sampling by categories
        category_samples = {}
        for item in test_data:
            for category in item["categories"]:
                if category not in category_samples:
                    category_samples[category] = []
                category_samples[category].append(item)
        
        # Sample proportionally from each category
        sample_data = []
        remaining_size = size
        
        for category, items in category_samples.items():
            if remaining_size <= 0:
                break
                
            # Calculate proportion
            proportion = len(items) / len(test_data)
            category_size = min(int(size * proportion), remaining_size, len(items))
            
            # Sample from category
            sampled_items = random.sample(items, category_size)
            sample_data.extend(sampled_items)
            remaining_size -= category_size
        
        # Fill remaining with random samples if needed
        if remaining_size > 0:
            remaining_items = [item for item in test_data if item not in sample_data]
            additional_samples = random.sample(remaining_items, min(remaining_size, len(remaining_items)))
            sample_data.extend(additional_samples)
        
        # Remove duplicates and trim to exact size
        unique_sample = list({item["id"]: item for item in sample_data}.values())[:size]
        
        # Save sample
        sample_file = output_dir / f"gsm8k_sample_{size}.json"
        with open(sample_file, 'w') as f:
            json.dump(unique_sample, f, indent=2)
        
        logging.info(f"Created sample dataset with {len(unique_sample)} problems: {sample_file}")


def validate_dataset(data_dir: Path) -> Dict[str, Any]:
    """
    Validate downloaded datasets for completeness and correctness.
    
    Args:
        data_dir: Directory containing datasets
        
    Returns:
        Validation results
    """
    logging.info("Validating datasets...")
    
    validation_results = {}
    
    for split in ["train", "test"]:
        file_path = data_dir / f"gsm8k_{split}.json"
        
        if not file_path.exists():
            validation_results[split] = {"status": "missing", "errors": [f"File not found: {file_path}"]}
            continue
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        errors = []
        warnings = []
        
        # Check basic structure
        if not isinstance(data, list):
            errors.append("Data is not a list")
        
        # Validate individual items
        required_fields = ["id", "question", "full_answer", "numerical_answer", "categories"]
        
        for i, item in enumerate(data[:100]):  # Check first 100 items
            if not isinstance(item, dict):
                errors.append(f"Item {i} is not a dictionary")
                continue
            
            # Check required fields
            for field in required_fields:
                if field not in item:
                    errors.append(f"Item {i} missing field: {field}")
            
            # Validate data types
            if "question" in item and not isinstance(item["question"], str):
                errors.append(f"Item {i} question is not string")
            
            if "categories" in item and not isinstance(item["categories"], list):
                errors.append(f"Item {i} categories is not list")
            
            # Check for empty questions
            if "question" in item and not item["question"].strip():
                warnings.append(f"Item {i} has empty question")
            
            # Validate numerical answer
            if "numerical_answer" in item:
                try:
                    float(item["numerical_answer"])
                except (ValueError, TypeError):
                    warnings.append(f"Item {i} has non-numeric answer: {item['numerical_answer']}")
        
        # Calculate file hash for integrity
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        
        validation_results[split] = {
            "status": "valid" if not errors else "invalid",
            "total_items": len(data),
            "errors": errors,
            "warnings": warnings,
            "file_hash": file_hash
        }
    
    return validation_results


def generate_dataset_report(data_dir: Path, output_file: Optional[Path] = None) -> None:
    """
    Generate comprehensive dataset analysis report.
    
    Args:
        data_dir: Directory containing datasets
        output_file: Optional output file for report
    """
    logging.info("Generating dataset analysis report...")
    
    report_lines = []
    report_lines.append("# GSM8K Dataset Analysis Report")
    report_lines.append(f"Generated on: {pd.Timestamp.now()}")
    report_lines.append("")
    
    for split in ["train", "test"]:
        file_path = data_dir / f"gsm8k_{split}.json"
        
        if not file_path.exists():
            continue
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        
        report_lines.append(f"## {split.title()} Set Analysis")
        report_lines.append("")
        report_lines.append(f"- **Total Problems**: {len(data):,}")
        report_lines.append(f"- **Average Word Count**: {df['word_count'].mean():.1f}")
        report_lines.append(f"- **Average Numbers per Problem**: {df['number_count'].mean():.1f}")
        report_lines.append("")
        
        # Category distribution
        all_categories = []
        for categories in df["categories"]:
            all_categories.extend(categories)
        
        category_counts = pd.Series(all_categories).value_counts()
        
        report_lines.append("### Problem Categories")
        for category, count in category_counts.head(10).items():
            percentage = (count / len(data)) * 100
            report_lines.append(f"- **{category.title()}**: {count:,} ({percentage:.1f}%)")
        report_lines.append("")
        
        # Word count distribution
        report_lines.append("### Word Count Distribution")
        report_lines.append(f"- **Min**: {df['word_count'].min()}")
        report_lines.append(f"- **25th percentile**: {df['word_count'].quantile(0.25):.0f}")
        report_lines.append(f"- **Median**: {df['word_count'].median():.0f}")
        report_lines.append(f"- **75th percentile**: {df['word_count'].quantile(0.75):.0f}")
        report_lines.append(f"- **Max**: {df['word_count'].max()}")
        report_lines.append("")
        
        # Sample problems
        report_lines.append("### Sample Problems")
        sample_problems = df.sample(3, random_state=42)
        for _, problem in sample_problems.iterrows():
            report_lines.append(f"**Problem {problem['id']}**")
            report_lines.append(f"- Question: {problem['question'][:100]}...")
            report_lines.append(f"- Answer: {problem['numerical_answer']}")
            report_lines.append(f"- Categories: {', '.join(problem['categories'])}")
            report_lines.append("")
        
        report_lines.append("---")
        report_lines.append("")
    
    # Save report
    report_content = "\n".join(report_lines)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_content)
        logging.info(f"Report saved to {output_file}")
    else:
        print(report_content)


def main():
    """Main function for data download script."""
    parser = argparse.ArgumentParser(description="Download and prepare GSM8K dataset")
    parser.add_argument("--output-dir", type=Path, default=Path("data"),
                       help="Output directory for datasets")
    parser.add_argument("--splits", nargs="+", default=["train", "test"],
                       help="Dataset splits to download")
    parser.add_argument("--sample-sizes", type=int, nargs="+", 
                       default=[10, 50, 100, 200],
                       help="Sample sizes to create")
    parser.add_argument("--validate", action="store_true",
                       help="Validate downloaded datasets")
    parser.add_argument("--report", action="store_true",
                       help="Generate analysis report")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    try:
        # Create output directory
        args.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Download and process datasets
        logging.info("Starting data download and processing...")
        stats = download_gsm8k_dataset(args.output_dir, args.splits)
        
        # Create sample datasets
        if "test" in args.splits:
            sample_dir = args.output_dir / "samples"
            sample_dir.mkdir(exist_ok=True)
            create_sample_datasets(args.output_dir, sample_dir, args.sample_sizes)
        
        # Validate datasets
        if args.validate:
            validation_results = validate_dataset(args.output_dir)
            
            # Save validation results
            validation_file = args.output_dir / "validation_results.json"
            with open(validation_file, 'w') as f:
                json.dump(validation_results, f, indent=2)
            
            # Print validation summary
            for split, result in validation_results.items():
                status = result["status"]
                errors = len(result.get("errors", []))
                warnings = len(result.get("warnings", []))
                
                print(f"{split} validation: {status} ({errors} errors, {warnings} warnings)")
        
        # Generate report
        if args.report:
            report_file = args.output_dir / "dataset_report.md"
            generate_dataset_report(args.output_dir, report_file)
        
        # Save download statistics
        stats_file = args.output_dir / "download_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logging.info("Data download and processing completed successfully!")
        
        # Print summary
        print("\n" + "="*50)
        print("DATA DOWNLOAD SUMMARY")
        print("="*50)
        for split, split_stats in stats.items():
            print(f"{split.title()} set: {split_stats['total_problems']:,} problems")
        print(f"Output directory: {args.output_dir}")
        print("="*50)
        
    except Exception as e:
        logging.error(f"Error during data download: {e}")
        raise


if __name__ == "__main__":
    main()
