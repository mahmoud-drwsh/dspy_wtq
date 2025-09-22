# 🚧 DSPy + WikiTableQuestions (WTQ) Experiment 🚧

> **Status**: Work in Progress - This is an experimental implementation of table question answering using DSPy with ReAct agents and custom table analysis tools.

## Overview

This project implements a table question answering system using:
- **DSPy 3.0+** for language model orchestration
- **ReAct agents** with custom table analysis tools
- **WikiTableQuestions (WTQ)** dataset for evaluation
- **LM Studio** for local model inference

The system uses a ReAct agent equipped with three specialized tools for table analysis:
- `count_column_values_tool`: Count exact matches in columns
- `count_column_contains_tool`: Count partial matches in columns  
- `get_row_by_condition_tool`: Retrieve specific rows based on conditions

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
uv venv --python 3.11

# Activate environment (if needed)
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows

# Install dependencies
uv pip install -r requirements.txt
```

### 2. Configuration

Edit `config.json` to customize your setup:

```json
{
  "model": "openai/qwen/qwen3-4b-2507",
  "api_base": "http://10.1.11.218:1234/v1",
  "api_key": "local",
  "model_type": "chat",
  "temperature": 1,
  "max_tokens": 16384,
  "test_questions_limit": 500
}
```

**Key Configuration Options:**
- `model`: Model identifier for your LM Studio setup
- `api_base`: LM Studio API endpoint
- `test_questions_limit`: Number of test questions to evaluate (default: 500)
- `max_iters`: Maximum ReAct iterations (default: 5)
- `tools`: Available table analysis tools

### 3. Dataset Setup

```bash
# Extract and preview WTQ dataset
uv run setup/extract_wtq_test_data.py
```

This will:
- Download the WikiTableQuestions dataset to `./.cache/`
- Print the first 10 test questions with their tables
- Ensure data is ready for evaluation

### 4. Run the Experiment

```bash
# Run the main evaluation
uv run main.py
```

This will:
- Load the configured model via LM Studio
- Run a smoke test to verify connectivity
- Load WTQ test questions (up to the configured limit)
- Execute the ReAct agent on each question
- Save detailed results and reasoning analysis

## Project Structure

```
dspy_wtq/
├── main.py                 # Main experiment runner
├── config.json            # Configuration file
├── requirements.txt       # Python dependencies
├── setup/                 # Dataset extraction scripts
├── utils/                 # Utility modules
│   ├── dataset_loader.py  # WTQ data loading
│   ├── dspy_utils.py      # DSPy configuration
│   ├── eval_utils.py      # Evaluation logic
│   └── ...
├── examples/              # Example scripts
├── docs/                  # Documentation
├── reports/               # Analysis reports
└── run_results/           # Experiment results
```

## Usage Examples

### Load WTQ Data in Your Code

```python
from utils import load_wtq_test_questions_with_tables

# Load test questions with tables
examples = load_wtq_test_questions_with_tables(limit=100)

# Each example contains:
# {
#   "id": "nt-0",
#   "question": "which country had the most cyclists...",
#   "answers": ["Italy"],
#   "table_name": "cycling_race_results",
#   "table": {
#     "header": ["Rank", "Cyclist", "Country", "Time"],
#     "rows": [["1", "Alejandro Valverde (ESP)", "Spain", "4:23:15"], ...],
#     "name": "cycling_race_results"
#   }
# }
```

### Run Example Scripts

```bash
# Preview dataset structure
uv run examples/wtq_iter_example.py

# Test DSPy + LM Studio connection
uv run examples/hello_dspy_lmstudio.py
```

## Latest Run Results Analysis

### Performance Summary (Latest Run: 2025-09-22 09:06:32)

**Configuration:**
- Model: `openai/qwen/qwen3-4b-2507`
- Test Questions: 393 (incremental run)
- Temperature: 0.0
- Max Iterations: 5

**Results:**
- **Accuracy: 46.3%** (182/393 correct)
- **Total Questions: 393**
- **Correct Answers: 182**

### Sample Question Analysis

#### ✅ Question 1: "which country had the most cyclists finish within the top 10?"
- **Expected**: Italy
- **Predicted**: Italy
- **Result**: ✅ **CORRECT**
- **Tool Calls**: 20
- **Reasoning**: Successfully used `get_row_by_condition_tool` to retrieve individual cyclist rows, extracted country codes from cyclist names (ESP, RUS, ITA), and correctly identified Italy as having the most cyclists

#### ✅ Question 2: "how many people were murdered in 1940/41?"
- **Expected**: 100,000
- **Predicted**: 100000
- **Result**: ✅ **CORRECT**
- **Tool Calls**: 12
- **Reasoning**: Successfully located the "Murdered" row in the "1940/41" column and extracted the correct value

#### ❌ Question 3: "how long did it take for the new york americans to win the national cup after 1936?"
- **Expected**: 17 years
- **Predicted**: I don't know
- **Result**: ❌ **INCORRECT**
- **Tool Calls**: 4
- **Reasoning**: Could not find relevant data about the New York Americans team in the provided table

### Question Types and Performance Patterns

The run included diverse question types:
- **Counting questions**: "how many people were murdered in 1940/41?" ✅
- **Comparative questions**: "which country had the most cyclists..." ✅
- **Temporal questions**: "how long did it take for the new york americans..." ❌
- **Aggregation questions**: "what is the total number of films with the language of kannada..." 
- **Ranking questions**: "who came immediately after sebastian porto in the race?"
- **Date/time questions**: "when was his first 1st place record?"

### Key Observations

1. **Tool Functionality**: The system successfully used table analysis tools (`get_row_by_condition_tool`, `count_column_values_tool`, `count_column_contains_tool`) to extract data from tables.

2. **Systematic Approach**: The agent demonstrated systematic reasoning by retrieving individual rows and analyzing data step-by-step.

3. **Answer Format Handling**: The system correctly handles various answer formats (country names, numbers, dates) and provides appropriate "I don't know" responses when data is insufficient.

4. **Large Sample Size**: This was a substantial incremental run (393 questions), providing statistically meaningful results.

### Historical Performance Context

From the most recent full run (2025-09-22 08:42:11):
- **Full Dataset**: 200 questions
- **Accuracy**: 49.0% (98/200 correct)
- **Configuration**: Temperature 0.0, 200 test questions

The current incremental run shows slightly lower performance (46.3% vs 49.0%) on a larger sample, suggesting some consistency in the system's capabilities.

### Technical Issues Identified

1. **Table State Management**: Global `current_table_data` variable may not be properly synchronized with the ReAct agent's tool calls.

2. **Tool Error Handling**: Tools return generic error messages instead of graceful degradation when table data is unavailable.

3. **Reasoning Trajectory**: The agent shows good reasoning capabilities but relies heavily on sample data when tools fail.

### Recommendations for Improvement

1. **Fix Table State**: Ensure proper table data loading and synchronization with tool functions
2. **Enhanced Error Handling**: Implement better fallback mechanisms when tools fail
3. **Tool Validation**: Add validation to ensure table data is available before tool execution
4. **Larger Evaluation**: Run full evaluation sets to get statistically significant results
5. **Temperature Tuning**: Experiment with different temperature settings for better consistency

---

*Last updated: 2025-09-22*
*Results from run: 20250922_090632*
