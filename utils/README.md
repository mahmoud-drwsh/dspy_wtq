# DSPy WTQ Utilities

A comprehensive utility library for working with WikiTableQuestions (WTQ) dataset using DSPy. This package provides streamlined tools for data loading, table processing, model configuration, and evaluation.

## Overview

This utility library is designed to facilitate table-based question answering using the WikiTableQuestions dataset with DSPy framework. It includes:

- **Dataset Management**: Automatic downloading and caching of WTQ dataset
- **Table Processing**: Token-efficient table formatting and serialization
- **Model Configuration**: Easy setup for various LLM providers (Ollama, LM Studio, OpenRouter)
- **Evaluation Tools**: Normalized answer comparison and accuracy metrics
- **I/O Utilities**: Robust file handling and data loading with fallback mechanisms

## Installation

The utilities are designed to work with Python 3.8+ and require the following dependencies:

```bash
pip install dspy requests
```

For specific model backends:
- **Ollama**: Install Ollama and pull your preferred models
- **LM Studio**: Download and set up LM Studio
- **OpenRouter**: Obtain an API key from OpenRouter

## Quick Start

### Basic Usage

```python
from utils import (
    ensure_wtq_data,
    load_wtq_test_questions_with_tables,
    configure_dspy,
    format_table_token_efficient,
    denotation_accuracy
)

# Ensure WTQ dataset is available (auto-downloads if needed)
data_dir = ensure_wtq_data()

# Load test examples with tables
examples = load_wtq_test_questions_with_tables(data_dir, limit=10)

# Configure DSPy with Ollama
dspy = configure_dspy(model="gemma3:4b", api_base="http://localhost:11434")

# Format a table for efficient token usage
table = examples[0]["table"]
formatted_table = format_table_token_efficient(table, question=examples[0]["question"])

print(f"Table formatted in {len(formatted_table)} characters")
```

### Advanced Configuration

```python
from utils import configure_dspy_lm_studio, configure_dspy_openrouter

# Configure with LM Studio
lm = configure_dspy_lm_studio(
    model_name="deepseek/deepseek-r1-0528-qwen3-8b",
    api_base="http://localhost:1234/v1",
    temperature=0.1,
    max_tokens=1024
)

# Configure with OpenRouter
lm = configure_dspy_openrouter(
    model_name="openrouter/deepseek/deepseek-r1-0528-qwen3-8b",
    max_tokens=1024,
    context_length=131072
)
```

## Module Documentation

### Dataset Utilities (`dataset_loader.py`)

Handles automatic downloading and management of the WTQ dataset.

#### Key Functions:

- `ensure_wtq_data()`: Ensures WTQ dataset is available locally, downloading if necessary
- `download_wtq_zip()`: Downloads the WTQ compact zip file
- `get_wtq_root_dir()`: Returns the root directory of the WTQ dataset
- `is_wtq_data_available()`: Checks if WTQ data is already available

**Example:**
```python
from utils.dataset_loader import ensure_wtq_data, is_wtq_data_available

if not is_wtq_data_available():
    print("Downloading WTQ dataset...")

data_dir = ensure_wtq_data()
print(f"WTQ data available at: {data_dir}")
```

### Splits Loading (`splits_loader.py`)

Provides functions to load different WTQ dataset splits with associated table data.

#### Key Functions:

- `load_wtq_splits()`: Load all available splits (train, validation, test)
- `load_wtq_splits_with_tables()`: Load splits with full table data included
- `load_wtq_test_questions_with_tables()`: Load test split with tables (backward compatibility)
- `get_split_summary()`: Get quick summary of available splits

**Example:**
```python
from utils import load_wtq_splits_with_tables, get_split_summary

# Get summary without loading full data
summary = get_split_summary()
print(f"Dataset summary: {summary}")

# Load all splits with tables
splits = load_wtq_splits_with_tables(limit={'train': 100, 'test': 50})
print(f"Loaded {len(splits['train'])} training examples")
```

### Table Utilities (`table_utils.py`)

Provides table formatting and serialization utilities optimized for LLM consumption.

#### Key Functions:

- `format_table_token_efficient()`: Most token-efficient table formatting
- `serialize_table_for_prompt()`: Traditional table serialization
- `human_table_preview()`: Human-readable table preview

**Example:**
```python
from utils.table_utils import format_table_token_efficient, human_table_preview

table = {
    "header": ["Name", "Age", "City"],
    "rows": [["Alice", 25, "New York"], ["Bob", 30, "San Francisco"]]
}

# Token-efficient format (recommended for LLMs)
compact_table = format_table_token_efficient(
    table,
    question="What cities are listed?",
    delimiter="|"
)
print("Compact format:")
print(compact_table)

# Human-readable format
preview = human_table_preview(table, n=2)
print("\nHuman preview:")
print(preview)
```

### DSPy Configuration (`dspy_utils.py`)

Provides model configuration utilities for various LLM providers.

#### Key Functions:

- `configure_dspy()`: Configure DSPy with Ollama
- `configure_dspy_lm_studio()`: Configure DSPy with LM Studio
- `configure_dspy_openrouter()`: Configure DSPy with OpenRouter
- `ping_ollama()`: Check if Ollama is available
- `print_token_usage()`: Display token usage statistics
- `build_module()`: Build DSPy modules (ChainOfThought or Predict)

**Example:**
```python
from utils.dspy_utils import (
    configure_dspy_lm_studio,
    ping_ollama,
    print_token_usage,
    build_module
)

# Check if Ollama is available
is_ollama_ok, model_info = ping_ollama()
print(f"Ollama status: {is_ollama_ok}, Models: {model_info}")

# Configure LM Studio
lm = configure_dspy_lm_studio(
    model_name="deepseek/deepseek-r1-0528-qwen3-8b",
    temperature=0.1
)

# Build a DSPy module
module = build_module(dspy, use_cot=True)

# Use the module
result = module(table_text=formatted_table, question="What is the average age?")
print_token_usage(result)
```

### Evaluation Utilities (`eval_utils.py`)

Provides evaluation metrics for table question answering.

#### Key Functions:

- `denotation_accuracy()`: Compute denotation accuracy between predictions and gold answers
- `normalize_token()`: Normalize tokens for comparison
- `split_prediction()`: Split prediction text into individual answer tokens

**Example:**
```python
from utils.eval_utils import denotation_accuracy, normalize_token

# Gold answers and predictions
gold_answers = [["New York", "NYC"], ["25 years"], ["San Francisco"]]
predictions = [["New York"], ["25"], ["San Francisco", "SF"]]

# Calculate accuracy
accuracy = denotation_accuracy(gold_answers, predictions)
print(f"Denotation accuracy: {accuracy:.2%}")

# Normalize individual tokens
print(f"Normalized: {normalize_token('$1,000.50%')}")
```

### I/O Utilities (`io_utils.py`)

Provides robust file handling and data loading utilities.

#### Key Functions:

- `eprint()`: Print to stderr
- `ensure_output_dir()`: Create output directory if it doesn't exist
- `load_examples_fallback()`: Load examples with multiple fallback mechanisms
- `load_examples_repo_utils()`: Load examples using repository utilities
- `read_csv_table()`: Read CSV/TSV table files

**Example:**
```python
from utils.io_utils import (
    ensure_output_dir,
    load_examples_fallback,
    eprint
)

# Create output directory
output_dir = ensure_output_dir("results/experiment_1")
eprint(f"Output directory created: {output_dir}")

# Load examples with fallback
examples = load_examples_fallback(
    wtq_dir=data_dir.parent,
    examples_jsonl=None,  # Auto-detect
    limit=5,
    col_limit=10
)
```

## Example Outputs

Here are actual outputs from running the WTQ utilities:

### Dataset Loading Example Output

```bash
============================================================
WTQ Dataset Loading Example
============================================================

1. Checking initial data availability:
   Data initially available: True

2. Ensuring WTQ data is available...
   Data directory: /Users/mahmoud/Documents/GitHub/dspy_wtq/.cache/WikiTableQuestions/data
   Root directory: /Users/mahmoud/Documents/GitHub/dspy_wtq/.cache/WikiTableQuestions

3. Getting split summary:
   Available splits: {'train': 14152, 'validation': 3537, 'test': 4344}

4. Loading test examples with tables (limit=3):
   Loaded 3 test examples

   Example 1:
     ID: nu-0
     Question: which country had the most cyclists finish within the top 10?
     Answers: ['Italy']
     Table name: csv/203-csv/733.csv
     Table header: ['Rank', 'Cyclist', 'Team', 'Time', 'UCI ProTour\\nPoints']
     Table rows: 10
     Sample row: ['1', 'Alejandro Valverde\xa0(ESP)', "Caisse d'Epargne", '5h 29\' 10"', '40']

   Example 2:
     ID: nu-1
     Question: how many people were murdered in 1940/41?
     Answers: ['100,000']
     Table name: csv/204-csv/149.csv
     Table header: ['Description Losses', '1939/40', '1940/41', '1941/42', '1942/43', '1943/44', '1944/45', 'Total']
     Table rows: 7
     Sample row: ['Direct War Losses', '360,000', '', '', '', '', '183,000', '543,000']

   Example 3:
     ID: nu-2
     Question: how long did it take for the new york americans to win the national cup after 1936?
     Answers: ['17 years']
     Table name: csv/203-csv/435.csv
     Table header: ['Year', 'Division', 'League', 'Reg. Season', 'Playoffs', 'National Cup']
     Table rows: 27
     Sample row: ['1931', '1', 'ASL', '6th (Fall)', 'No playoff', 'N/A']

5. Loading all splits with tables (limit=2 each):
   train: 2 examples
     Sample question: what was the last year where this team was a part of the usl a-league?
   validation: 2 examples
     Sample question: how long did grand blanc high school participate for?
   test: 2 examples
     Sample question: which country had the most cyclists finish within the top 10?
```

### Table Formatting Example Output

```bash
============================================================
Table Formatting Example
============================================================

1. Loading test examples...

========================================
Example 1
========================================

Question: which country had the most cyclists finish within the top 10?
Expected Answers: ['Italy']
Table Name: csv/203-csv/733.csv

2. Human-readable preview:
Header (5 cols): ['Rank', 'Cyclist', 'Team', 'Time', 'UCI ProTour\\nPoints']
Rows: 10 total; showing first 3
  ['1', 'Alejandro Valverde\xa0(ESP)', "Caisse d'Epargne", '5h 29\' 10"', '40']
  ['2', 'Alexandr Kolobnev\xa0(RUS)', 'Team CSC Saxo Bank', 's.t.', '30']
  ['3', 'Davide Rebellin\xa0(ITA)', 'Gerolsteiner', 's.t.', '25']

3. Traditional serialization:
Table: csv/203-csv/733.tsv
Header: Rank | Cyclist | Team | Time | UCI ProTour\nPoints
Row: 1 | Alejandro Valverde (ESP) | Caisse d'Epargne | 5h 29' 10" | 40
Row: 2 | Alexandr Kolobnev (RUS) | Team CSC Saxo Bank | s.t. | 30
Row: 3 | Davide Rebellin (ITA) | Gerolsteiner | s.t. | 25
Row: 4 | Paolo Bettini (ITA) | Quick Step | s.t. | 20
Row: 5 | Franco Pellizotti (ITA) | Liquigas | s.t. | 15
... (5 more rows truncated)
   Length: 418 characters

4. Token-efficient formatting:
Rank|Cyclist|Team|Time|UCI ProTour\nPoints
1|Alejandro Valverde (ESP)|Caisse d'Epargne|5h 29' 10"|40
2|Alexandr Kolobnev (RUS)|Team CSC Saxo Bank|s.t.|30
3|Davide Rebellin (ITA)|Gerolsteiner|s.t.|25
4|Paolo Bettini (ITA)|Quick Step|s.t.|20
5|Franco Pellizotti (ITA)|Liquigas|s.t.|15
6|Denis Menchov (RUS)|Rabobank|s.t.|11
7|Samuel Sánchez (ESP)|Euskaltel-Euskadi|s.t.|7
8|Stéphane Goubert (FRA)|Ag2r-La Mondiale|+ 2"|5
9|Haimar Zubeldia (ESP)|Euskaltel-Euskadi|+ 2"|3
10|David Moncoutié (FRA)|Cofidis|+ 2"|1
   Length: 507 characters

   Token savings: -89 characters (-21.3%)

========================================
Question-based filtering example
========================================

5. Without question filtering:
Name|Age|City|Country|Year|Population
New York|394|USA|2020|8336817
Los Angeles|379|USA|2020|3979576
Chicago|271|USA|2020|2693976
Houston|232|USA|2020|2320268
Phoenix|168|USA|2020|1680992
   Length: 187 characters

6. With question filtering (What cities have population over 1 million?):
Country|Year|Population
2020|8336817
2020|3979576
2020|2693976
2020|2320268
2020|1680992
   Length: 88 characters
```

### Evaluation Utilities Example Output

```bash
============================================================
Evaluation Utilities Example
============================================================

1. Loading test examples...
Loaded 5 examples for evaluation demo

2. Token normalization examples:
   'New York' -> 'new york'
   '1,000.50' -> '1000.5'
   '$500' -> '500'
   '75%' -> '75'
   '  extra  spaces  ' -> 'extra spaces'
   '"quoted"' -> 'quoted'
   'mixed-CASE' -> 'mixed-case'

3. Prediction splitting examples:
   'New York|NYC|Manhattan' (gold_count=1) -> ['new york|nyc|manhattan']
   'New York|NYC|Manhattan' (gold_count=3) -> ['new york', 'nyc', 'manhattan']
   'New York, NYC, Manhattan' (gold_count=1) -> ['new york, nyc, manhattan']
   'New York, NYC, Manhattan' (gold_count=3) -> ['new york', 'nyc', 'manhattan']

4. Computing denotation accuracy:
   Accuracy: 60.00% (3/5)

5. Detailed comparison:
   Example 1: ✅
     Gold normalized: ['italy']
     Pred normalized: ['italy']
   Example 2: ❌
     Gold normalized: ['100000']
     Pred normalized: ['some alternative']
   Example 3: ✅
     Gold normalized: ['17 years']
     Pred normalized: ['17 years']

6. Edge case testing:
   Number normalization: ✅ (accuracy: 100.00%)
   Percentage normalization: ✅ (accuracy: 100.00%)
   Case normalization: ✅ (accuracy: 100.00%)
   Space normalization: ✅ (accuracy: 100.00%)
   Quote normalization: ✅ (accuracy: 100.00%)
```

## Complete Example

Here's a complete example demonstrating the full workflow:

```python
import dspy
from utils import (
    ensure_wtq_data,
    load_wtq_test_questions_with_tables,
    configure_dspy_lm_studio,
    format_table_token_efficient,
    build_module,
    denotation_accuracy
)
from utils.io_utils import ensure_output_dir, eprint

def main():
    # 1. Setup and data loading
    eprint("Setting up WTQ dataset...")
    data_dir = ensure_wtq_data()

    eprint("Loading test examples...")
    examples = load_wtq_test_questions_with_tables(data_dir, limit=10)

    # 2. Model configuration
    eprint("Configuring DSPy...")
    lm = configure_dspy_lm_studio(
        model_name="deepseek/deepseek-r1-0528-qwen3-8b",
        temperature=0.1,
        max_tokens=512
    )

    # 3. Build DSPy module
    eprint("Building DSPy module...")
    module = build_module(dspy, use_cot=True)

    # 4. Process examples
    predictions = []
    gold_answers = []

    for example in examples[:5]:  # Process first 5 examples
        # Format table efficiently
        table_text = format_table_token_efficient(
            example["table"],
            question=example["question"]
        )

        # Get prediction
        result = module(table_text=table_text, question=example["question"])
        prediction = result.answer

        predictions.append([prediction])
        gold_answers.append(example["answers"])

        eprint(f"Q: {example['question']}")
        eprint(f"Predicted: {prediction}")
        eprint(f"Expected: {example['answers']}")
        eprint("---")

    # 5. Evaluate
    accuracy = denotation_accuracy(gold_answers, predictions)
    eprint(f"Final accuracy: {accuracy:.2%}")

    # 6. Save results
    output_dir = ensure_output_dir("results")
    with open(output_dir / "results.txt", "w") as f:
        f.write(f"Accuracy: {accuracy:.2%}\n")

    eprint("Processing complete!")

if __name__ == "__main__":
    main()
```

## API Reference

### Complete Function List

All utilities are available through the main `utils` package:

```python
from utils import (
    # WTQ utilities
    "ensure_wtq_data",
    "load_wtq_test_questions_with_tables",

    # I/O utilities
    "eprint",
    "ensure_output_dir",
    "load_examples_fallback",
    "load_examples_repo_utils",

    # Table utilities
    "human_table_preview",
    "serialize_table_for_prompt",

    # Evaluation utilities
    "denotation_accuracy",
    "normalize_token",
    "split_prediction",

    # DSPy utilities
    "build_module",
    "configure_dspy",
    "configure_dspy_openrouter",
    "configure_dspy_lm_studio",
    "print_token_usage",
    "ping_ollama",
)
```

### Configuration Options

#### Ollama Configuration
```python
dspy = configure_dspy(
    model="gemma3:4b",
    api_base="http://localhost:11434",
    temperature=0.1,
    max_tokens=512,
    disable_cache=False
)
```

#### LM Studio Configuration
```python
lm = configure_dspy_lm_studio(
    model_name="deepseek/deepseek-r1-0528-qwen3-8b",
    api_base="http://localhost:1234/v1",
    api_key="local",
    temperature=0.1,
    max_tokens=1024,
    track_usage=False
)
```

#### OpenRouter Configuration
```python
lm = configure_dspy_openrouter(
    model_name="openrouter/deepseek/deepseek-r1-0528-qwen3-8b",
    max_tokens=1024,
    context_length=131072,
    track_usage=False
)
```

## Troubleshooting

### Common Issues

1. **Dataset Download Fails**
   - Check internet connection
   - Verify disk space in project directory
   - Ensure write permissions in project folder

2. **Ollama Connection Issues**
   - Verify Ollama is running: `ollama serve`
   - Check if model is available: `ollama list`
   - Pull model if needed: `ollama pull gemma3:4b`

3. **LM Studio Issues**
   - Ensure LM Studio is running
   - Verify API endpoint is accessible
   - Check model is loaded in LM Studio

4. **Memory Issues**
   - Reduce `max_tokens` parameter
   - Use smaller models (e.g., `gemma3:4b` instead of larger models)
   - Process fewer examples at once

### Debug Mode

Enable debug output:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Use verbose error messages
try:
    examples = load_wtq_test_questions_with_tables(data_dir)
except Exception as e:
    print(f"Error loading examples: {e}")
    import traceback
    traceback.print_exc()
```

## Contributing

This utility library is designed to be extensible. To contribute:

1. Follow existing code style and patterns
2. Add comprehensive docstrings for new functions
3. Include type hints for all parameters and return values
4. Test with multiple model backends
5. Ensure backward compatibility

## License

This project follows the license terms of the original WikiTableQuestions dataset and DSPy framework. Please ensure compliance with both when using these utilities.