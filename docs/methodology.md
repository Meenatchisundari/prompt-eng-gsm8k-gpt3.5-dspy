# Research Methodology

Comprehensive methodology for evaluating prompting techniques on mathematical reasoning tasks.

## Overview

This document details the rigorous methodology used to evaluate prompting techniques on the GSM8K dataset, ensuring reproducible and statistically valid results.

## Experimental Design

### Research Questions

1. **Primary**: Which prompting technique achieves the highest accuracy on GSM8K mathematical reasoning tasks?
2. **Secondary**: What is the cost-effectiveness trade-off between different techniques?
3. **Exploratory**: How do techniques perform across different problem categories and difficulty levels?

### Hypotheses

- **H1**: Structured reasoning approaches (Prolog-Style) will outperform unstructured approaches (Zero-Shot)
- **H2**: Multi-sample techniques (Self-Consistency) will achieve higher accuracy but at increased computational cost
- **H3**: Few-shot learning will significantly improve performance over zero-shot baselines

## Dataset Description

### GSM8K Dataset Characteristics

```
Total Problems: 8,792 (training) + 1,319 (test)
Domain: Grade school mathematics (ages 6-12)
Problem Types:
  - Arithmetic operations (35%)
  - Multi-step word problems (40%)
  - Money/time calculations (15%)
  - Basic geometry (10%)

Difficulty Distribution:
  - Simple (1-2 steps): 45%
  - Medium (3-4 steps): 40%
  - Complex (5+ steps): 15%

Answer Format: Numerical values (integers and decimals)
Language: English
Quality: Human-verified, professionally created
```

### Data Preprocessing

1. **Answer Extraction**: Extract numerical answers from solution text using regex patterns
2. **Validation**: Ensure all problems have valid numerical answers
3. **Categorization**: Classify problems by type using keyword analysis
4. **Quality Check**: Manual review of 100 random samples for accuracy

### Evaluation Split

- **Training Set**: Not used (zero-shot evaluation)
- **Test Set**: 1,319 problems (standard split)
- **Validation**: 10-fold cross-validation for statistical robustness
- **Sampling**: Stratified sampling to maintain problem type distribution

## Experimental Setup

### Model Configuration

```yaml
Model: GPT-3.5-turbo (gpt-3.5-turbo-0613)
Temperature: 0.0 (deterministic evaluation)
Max Tokens: 1,000
Top-p: 1.0
Frequency Penalty: 0.0
Presence Penalty: 0.0
API Version: 2023-07-01-preview
```

### Evaluation Protocol

#### Single Technique Evaluation
```python
for problem in test_set:
    try:
        # Generate prediction with timeout
        start_time = time.time()
        prediction = technique.forward(problem.question)
        response_time = time.time() - start_time
        
        # Extract and validate answer
        predicted_answer = extract_answer(prediction.answer)
        
        # Compute accuracy
        is_correct = math_accuracy(problem, prediction, tolerance=0.01)
        
        # Record metrics
        record_result(problem, prediction, is_correct, response_time)
        
    except TimeoutError:
        record_error(problem, "timeout")
    except APIError as e:
        record_error(problem, f"api_error: {e}")
```

#### Statistical Validation
```python
# Confidence intervals
ci_lower, ci_upper = calculate_confidence_interval(
    successes=correct_count,
    total=total_count,
    confidence_level=0.95
)

# Significance testing
p_value = fisher_exact_test(
    technique_a_results,
    technique_b_results
)

# Effect size calculation
effect_size = cohens_h(
    proportion_a=accuracy_a,
    proportion_b=accuracy_b
)
```

## Prompting Technique Implementation

### 1. Zero-Shot Prompting

**Design Principle**: Minimal prompting without examples or special instructions.

```python
prompt_template = """Solve this math word problem and provide only the numerical answer.

Problem: {question}
Answer:"""
```

**Rationale**: Establishes baseline performance for comparison with enhanced techniques.

### 2. Few-Shot Prompting

**Design Principle**: Learning from 4 carefully selected examples representing common problem types.

```python
examples = [
    ("Janet's ducks lay 16 eggs per day...", "18"),
    ("A robe takes 2 bolts of blue fiber...", "3"),
    ("Josh buys a house for $80,000...", "70000"),
    ("Sarah has 3 boxes of pencils...", "28")
]

prompt_template = """Solve math word problems using these examples:
{examples}
Now solve: {question}
Answer:"""
```

**Selection Criteria**:
- Diverse problem types
- Varying complexity levels
- Clear solution patterns
- Validated correct answers

### 3. Chain-of-Thought (CoT)

**Design Principle**: Explicit step-by-step reasoning to improve accuracy on complex problems.

```python
prompt_template = """Solve this math problem step by step, showing your detailed reasoning and calculations.

Problem: {question}
Step-by-step solution:"""
```

**Implementation Details**:
- Encourages explicit intermediate steps
- No specific format requirements
- Natural language reasoning
- Final answer extraction from complete response

### 4. Self-Consistency

**Design Principle**: Generate multiple reasoning paths and use majority voting for robustness.

```python
# Generate multiple samples
predictions = []
for i in range(n_samples):
    with temperature(0.7):  # Add randomness
        pred = chain_of_thought_predictor(question)
        predictions.append(extract_answer(pred))

# Majority voting
final_answer = Counter(predictions).most_common(1)[0][0]
confidence = majority_count / n_samples
```

**Parameters**:
- Sample size: 5 (optimal balance of accuracy vs. cost)
- Temperature: 0.7 (sufficient diversity without degradation)
- Voting: Simple majority with confidence scoring

### 5. Prolog-Style Logical Reasoning

**Design Principle**: Structured logical reasoning mimicking Prolog programming paradigm.

```python
prompt_template = """Solve this math problem using logical reasoning with facts, rules, and derivation.

Structure your response exactly as:
FACTS: [List what we know from the problem]
RULES: [State mathematical relationships and operations needed]  
QUERY: [What we want to find]
DERIVATION: [Step-by-step logical reasoning using facts and rules]
ANSWER: [Final numerical result]

Problem: {question}"""
```

**Rationale**:
- Forces systematic problem decomposition
- Separates known facts from logical operations
- Explicit reasoning chain
- Reduces calculation errors through structure

## Evaluation Metrics

### Primary Metrics

#### Accuracy
```python
accuracy = correct_predictions / total_predictions

# With confidence interval
ci_lower, ci_upper = binom.interval(
    confidence=0.95,
    n=total_predictions, 
    p=accuracy
)
```

#### Response Time
```python
avg_response_time = sum(response_times) / len(response_times)
median_response_time = np.median(response_times)
```

### Secondary Metrics

#### Cost Effectiveness
```python
cost_per_problem = (input_tokens * input_rate + output_tokens * output_rate)
cost_effectiveness = accuracy / cost_per_problem
```

#### Token Usage
```python
avg_input_tokens = sum(input_tokens) / len(problems)
avg_output_tokens = sum(output_tokens) / len(problems)
total_tokens = avg_input_tokens + avg_output_tokens
```

#### Error Rate
```python
api_error_rate = api_errors / total_attempts
timeout_error_rate = timeouts / total_attempts
total_error_rate = (api_errors + timeouts) / total_attempts
```

### Derived Metrics

#### Efficiency Score
```python
efficiency = (accuracy * 100) / avg_response_time
```

#### Reliability Score
```python
reliability = 1 - total_error_rate
```

## Statistical Analysis

### Descriptive Statistics

```python
# Central tendency
mean_accuracy = np.mean(accuracies)
median_accuracy = np.median(accuracies)
std_accuracy = np.std(accuracies)

# Distribution analysis
skewness = scipy.stats.skew(accuracies)
kurtosis = scipy.stats.kurtosis(accuracies)
```

### Inferential Statistics

#### Pairwise Comparisons
```python
# Fisher's exact test for categorical outcomes
def fisher_exact_test(results_a, results_b):
    """Test significance of accuracy difference between two techniques."""
    
    # Contingency table
    correct_a = sum(results_a)
    correct_b = sum(results_b)
    incorrect_a = len(results_a) - correct_a
    incorrect_b = len(results_b) - correct_b
    
    contingency = [[correct_a, incorrect_a], 
                   [correct_b, incorrect_b]]
    
    odds_ratio, p_value = fisher_exact(contingency)
    return p_value, odds_ratio
```

#### Multiple Comparisons Correction
```python
# Bonferroni correction for multiple pairwise tests
from statsmodels.stats.multitest import multipletests

p_values_corrected = multipletests(
    p_values_raw, 
    alpha=0.05, 
    method='bonferroni'
)[1]
```

#### Effect Size Analysis
```python
def cohens_h(p1, p2):
    """Calculate Cohen's h for proportions."""
    return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))

# Interpretation:
# |h| < 0.2: small effect
# 0.2 ≤ |h| < 0.5: medium effect  
# |h| ≥ 0.5: large effect
```

### Power Analysis

```python
from statsmodels.stats.power import ttest_power

# Determine required sample size
required_n = ttest_power(
    effect_size=0.2,    # Small-medium effect
    alpha=0.05,         # Type I error rate
    power=0.8,          # 80% power
    alternative='two-sided'
)
```

## Quality Assurance

### Reproducibility Measures

#### Random Seed Control
```python
# Fixed seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# DSPy deterministic configuration
dspy.configure(lm=dspy.LM(model="gpt-3.5-turbo", temperature=0.0))
```

#### Version Control
```python
# Log all dependency versions
requirements = {
    'dspy-ai': '2.4.9',
    'openai': '1.3.5',
    'pandas': '2.1.3',
    'numpy': '1.24.3'
}
```

#### Environment Documentation
```yaml
experimental_environment:
  platform: "Ubuntu 20.04 LTS"
  python_version: "3.10.12"
  hardware: "Intel Xeon E5-2686 v4"
  memory: "16GB RAM"
  api_endpoint: "api.openai.com"
  evaluation_date: "2024-01-15"
```

### Validation Procedures

#### Human Evaluation
```python
# Sample 100 random predictions for human validation
human_validation_sample = random.sample(all_predictions, 100)

# Human-AI agreement calculation
human_ai_agreement = sum(
    human_judgment == ai_judgment 
    for human_judgment, ai_judgment in validation_pairs
) / len(validation_pairs)
```

#### Cross-Validation
```python
# K-fold cross-validation for robustness
from sklearn.model_selection import KFold

kfold = KFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = []

for train_idx, test_idx in kfold.split(dataset):
    test_subset = [dataset[i] for i in test_idx]
    accuracy = evaluate_technique(technique, test_subset)
    cv_scores.append(accuracy)

cv_mean = np.mean(cv_scores)
cv_std = np.std(cv_scores)
```

### Error Analysis Methodology

#### Error Categorization
```python
def categorize_error(question, expected, predicted):
    """Categorize prediction errors by type."""
    
    try:
        exp_val = float(expected.replace(',', ''))
        pred_val = float(predicted.replace(',', ''))
        
        relative_error = abs(exp_val - pred_val) / max(abs(exp_val), 1)
        
        if relative_error < 0.1:
            return "calculation_error"
        elif relative_error > 2.0:
            return "comprehension_error"
        else:
            return "reasoning_error"
            
    except ValueError:
        return "format_error"
```

#### Problem Difficulty Assessment
```python
def assess_difficulty(question):
    """Assess problem difficulty based on linguistic features."""
    
    # Count mathematical operations
    operations = len(re.findall(r'\b(add|subtract|multiply|divide|plus|minus|times|divided)\b', question.lower()))
    
    # Count numerical values
    numbers = len(re.findall(r'\d+', question))
    
    # Sentence complexity
    sentences = len(question.split('.'))
    
    # Difficulty score
    difficulty = (operations * 2 + numbers + sentences) / 10
    
    if difficulty < 0.3:
        return "easy"
    elif difficulty < 0.7:
        return "medium"
    else:
        return "hard"
```

## Limitations and Assumptions

### Methodological Limitations

1. **Model Dependency**: Results specific to GPT-3.5-turbo architecture and training
2. **Language Constraints**: English-only evaluation limits generalizability
3. **Domain Specificity**: Mathematical reasoning may not transfer to other domains
4. **Temporal Sensitivity**: Model capabilities may change with updates

### Evaluation Constraints

1. **Exact Match**: Binary accuracy metric doesn't capture partial correctness
2. **Single Dataset**: GSM8K may not represent all mathematical reasoning tasks
3. **Cost Consideration**: API costs limit large-scale statistical power
4. **Human Baseline**: Limited human evaluation data for comparison

### Statistical Assumptions

1. **Independence**: Assumes problem solutions are independent
2. **Stationarity**: Model performance assumed stable across evaluation period
3. **Normality**: Some statistical tests assume normal distribution of errors
4. **Homoscedasticity**: Equal variance assumption in comparative tests

## Ethical Considerations

### Bias Assessment

#### Dataset Bias
- Demographic representation in problem contexts
- Cultural assumptions in word problems
- Socioeconomic bias in real-world scenarios

#### Model Bias
- Training data biases inherited from pre-trained models
- Systematic errors across problem categories
- Fairness across different mathematical domains

### Environmental Impact

```python
# Carbon footprint estimation
total_api_calls = sum(calls_per_technique.values())
estimated_energy_per_call = 0.002  # kWh (estimate)
total_energy = total_api_calls * estimated_energy_per_call
carbon_footprint = total_energy * 0.5  # kg CO2 (US grid average)
```

## Future Work

### Methodological Extensions

1. **Multi-Modal Evaluation**: Include diagram-based mathematical problems
2. **Longitudinal Studies**: Track performance changes over time
3. **Cross-Domain Validation**: Test on other reasoning datasets (MATH, MathQA)
4. **Human-AI Collaboration**: Study human-in-the-loop improvements

### Technical Improvements

1. **Adaptive Techniques**: Dynamic technique selection based on problem type
2. **Meta-Learning**: Learning to learn optimal prompting strategies
3. **Uncertainty Quantification**: Confidence estimation for predictions
4. **Explanation Quality**: Systematic evaluation of reasoning quality

### Statistical Enhancements

1. **Bayesian Analysis**: Uncertainty quantification in technique comparisons
2. **Causal Inference**: Understanding causal mechanisms in prompting effectiveness
3. **Sequential Testing**: Early stopping criteria for efficient evaluation
4. **Non-Parametric Methods**: Robust statistical tests for non-normal distributions
