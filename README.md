# 📊 Prompt Engineering on GSM8K with GPT-3.5 & DSPy

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/prompt-eng-gsm8k-gpt3.5-dspy/blob/main/examples/gsm8k_colab.ipynb)

**Author:** Meenatchi Sundari  
**Repo:** `prompt-eng-gsm8k-gpt3.5-dspy`

---

## 📌 Overview

This repository implements and benchmarks **multiple prompt engineering techniques** on the [GSM8K](https://huggingface.co/datasets/openai/gsm8k) arithmetic reasoning dataset using:

- **Model:** GPT-3.5 (`gpt-3.5-turbo`)
- **Framework:** [DSPy](https://github.com/stanfordnlp/dspy) for structured prompting & evaluation
- **Evaluation Metrics:** Accuracy, latency, and efficiency

We explore both **baseline** and **advanced** prompting methods and compare them against a **fine-tuned model**.

---

## 🧠 Techniques Implemented

**Core Prompting Strategies**
1. **Zero-Shot Prompting** — Solve without examples
2. **Few-Shot Prompting** — Provide Q&A examples
3. **Chain-of-Thought (CoT)** — Step-by-step reasoning
4. **Self-Consistency** — Multiple CoT runs with majority voting
5. **Prolog-Style Reasoning** — Structured, logic-like problem solving *(novel variant for GSM8K)*

**Advanced / Hybrid Strategies**
- Enhanced Prolog
- Calculator-Augmented Reasoning
- Verification Chains
- Weighted Ensemble
- Specialized Solver with task routing

**Fine-Tuning**
- Quick fine-tuning on distilled GPT-2 or similar models
- LoRA/PEFT experiments

---

## 📂 Repository Structure

```plaintext
prompt-eng-gsm8k-gpt3.5-dspy/
├── src/gsm8k_bench/
│   ├── cli.py                 # CLI entry point
│   ├── data.py                # GSM8K loader
│   ├── utils.py               # Answer extraction, matching
│   ├── benchmark.py           # Benchmark runner
│   ├── viz.py                 # Plotting & results tables
│   ├── techniques/            # Core techniques
│   │   ├── zero_shot.py
│   │   ├── few_shot.py
│   │   ├── cot.py
│   │   ├── self_consistency.py
│   │   └── prolog_style.py
│   ├── improvements/          # Advanced techniques
│   └── finetune/              # Fine-tuning pipeline
├── configs/                   # YAML configs for experiments
├── examples/                  # Notebooks & Colab demos
│   └── gsm8k_colab.ipynb
├── tests/                     # Unit tests
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## 🔧 Installation

**Clone the repository**
```bash
git clone https://github.com/Meenatchisundari/prompt-eng-gsm8k-gpt3.5-dspy.git
cd prompt-eng-gsm8k-gpt3.5-dspy
```

**Create and activate virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

**Set your API key**
```bash
export OPENAI_API_KEY="sk-..."   # Linux / Mac
# PowerShell (Windows):
# $env:OPENAI_API_KEY="sk-..."
```

---

## ▶️ Usage

**Run benchmarks with default config**
```bash
python -m gsm8k_bench.cli run --config configs/default.yaml
```

**Run with custom parameters**
```bash
python -m gsm8k_bench.cli run   --n-samples 50   --techniques zero-shot few-shot cot prolog   --model gpt-3.5-turbo   --temperature 0.0
```

**Run fine-tuning pipeline**
```bash
python -m gsm8k_bench.cli finetune --config configs/finetune.yaml
```

---

## 📊 Example Results

| Technique                  | Accuracy (%) | Avg Time (s) | Efficiency |
|----------------------------|--------------|--------------|------------|
| Zero-Shot                  | 55.0         | 1.2          | 45.8       |
| Few-Shot                   | 60.5         | 1.3          | 46.5       |
| Chain-of-Thought           | 68.0         | 2.5          | 27.2       |
| Self-Consistency (n=5)     | 72.5         | 8.0          | 9.0        |
| Prolog-Style               | 71.0         | 3.0          | 23.7       |
| Enhanced Prolog            | **74.5**     | 3.5          | 21.3       |


---

## 📈 Visualisations

The `viz.py` module generates:
- Accuracy comparison bar charts
- Response time plots
- Accuracy vs Time scatter plots
- Heatmaps for multi-technique analysis

---

## 📚 References

- [GSM8K Dataset](https://huggingface.co/datasets/openai/gsm8k)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [DSPy GitHub](https://github.com/stanfordnlp/dspy)
- [Google Prompt Engineering Docs](https://developers.google.com/machine-learning/resources/prompt-eng)
- [Microsoft Prompt Docs](https://learn.microsoft.com/en-us/semantic-kernel/prompts/)

---

## 📜 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---
