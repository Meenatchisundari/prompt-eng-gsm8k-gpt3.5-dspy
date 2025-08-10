"""
Setup script for GSM8K Benchmark package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="gsm8k-bench",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive benchmark for prompting techniques on GSM8K math problems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/prompt-eng-gsm8k-gpt3.5-dspy",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "finetuning": [
            "transformers>=4.30.0",
            "torch>=2.0.0",
            "accelerate>=0.20.0",
            "peft>=0.4.0",
            "bitsandbytes>=0.39.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "gsm8k-bench=gsm8k_bench.cli:cli",
        ],
    },
    include_package_data=True,
    package_data={
        "gsm8k_bench": ["configs/*.yaml"],
    },
)
