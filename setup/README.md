# WTQ Dataset Setup and Preparation

This directory contains tools and instructions for setting up and preparing the WikiTableQuestions (WTQ) dataset for use with DSPy and other machine learning frameworks.

## Overview

WikiTableQuestions (WTQ) is a large-scale dataset for question answering on semi-structured tables. The dataset contains approximately 22,000 question-answer pairs across 2,100 Wikipedia tables, with test tables disjoint from training tables to stress compositional generalization to unseen schemas.

## Files in this Directory

- `WikiTableQuestions-1.0.2-compact.zip` - The original WTQ dataset archive
- `extract_wtq_test_data.py` - Python script to extract and process the test split
- `README.md` - This documentation file

## Prerequisites

Before running the setup script, ensure you have the following installed:

```bash
# Install required Python packages
pip install requests
```

Or if using uv (recommended for this project):

```bash
uv add requests
```

## Dataset Structure

The WTQ dataset contains the following splits:
- **Training data**: Multiple random splits (random-split-1 through random-split-5)
- **Validation data**: Development sets corresponding to each random split
- **Test data**: `pristine-unseen-tables.tsv` - Unseen tables for final evaluation

## Quick Start

### 1. Extract and Process Test Data

Run the setup script to extract the dataset and create a JSON log file with all test questions and answers:

```bash
# From the project root directory
uv run setup/extract_wtq_test_data.py
```

This will:
- Extract the `WikiTableQuestions-1.0.2-compact.zip` file to `../.cache/`
- Process the test split (`pristine-unseen-tables.tsv`)
- Create a JSON log file at `../test/wtq_test_qa.log`

### 2. Verify the Output

Check that the log file was created successfully:

```bash
# Count the number of examples processed
wc -l test/wtq_test_qa.log

# View the first few examples
head -5 test/wtq_test_qa.log
```

Expected output:
- **4,345 lines** total (1 header + 4,344 examples)
- Each line contains a JSON object with `question`, `answers`, and `id` fields

## Detailed Process

### What the Script Does

1. **Extraction**: 
   - Extracts the zip file to `../.cache/WikiTableQuestions/`
   - Preserves the original directory structure

2. **Data Processing**:
   - Reads `pristine-unseen-tables.tsv` from the extracted data
   - Parses each line to extract question, table reference, and answers
   - Handles pipe-separated multiple answers

3. **Output Generation**:
   - Creates a header line with dataset metadata
   - Logs each question-answer pair as a separate JSON line
   - Saves to `../test/wtq_test_qa.log`

### Output Format

Each line in the log file is a JSON object with the following structure:

```json
{
  "question": "which country had the most cyclists finish within the top 10?",
  "answers": ["Italy"],
  "id": 0
}
```

For questions with multiple correct answers:
```json
{
  "question": "which mayors had more than 2 consecutive terms?",
  "answers": ["Mr B.Melman", "Mr P.Venter", "Mrs E.Myer"],
  "id": 3438
}
```

## Directory Structure After Setup

```
dspy_wtq/
├── .cache/                          # Extracted dataset (created by script)
│   └── WikiTableQuestions/
│       ├── data/
│       │   ├── pristine-unseen-tables.tsv
│       │   ├── random-split-1-train.tsv
│       │   ├── random-split-1-dev.tsv
│       │   └── csv/                 # Table files
│       └── ...
├── setup/                           # Setup tools
│   ├── WikiTableQuestions-1.0.2-compact.zip
│   ├── extract_wtq_test_data.py
│   └── README.md
├── test/                            # Output directory (created by script)
│   └── wtq_test_qa.log             # Processed test data
└── ...
```

## Using the Processed Data

### With DSPy

The processed JSON log file can be easily loaded for DSPy experiments:

```python
import json

# Load the processed test data
test_examples = []
with open('test/wtq_test_qa.log', 'r') as f:
    for line in f:
        if line.strip():  # Skip empty lines
            example = json.loads(line)
            if 'question' in example:  # Skip header line
                test_examples.append(example)

print(f"Loaded {len(test_examples)} test examples")
```

### For Evaluation

The JSON format makes it easy to evaluate model predictions:

```python
def exact_match(gold_answers, predicted_answer):
    """Check if predicted answer matches any gold answer."""
    return any(gold.lower().strip() == predicted_answer.lower().strip() 
               for gold in gold_answers)

# Example evaluation
gold = ["Italy"]
predicted = "Italy"
print(exact_match(gold, predicted))  # True
```

## Troubleshooting

### Common Issues

1. **Permission Errors**: Ensure you have write permissions to create the `.cache` and `test` directories
2. **Missing Dependencies**: Install required packages with `pip install requests` or `uv add requests`
3. **Path Issues**: Run the script from the project root directory, not from within the setup directory

### Manual Extraction

If the script fails, you can manually extract the dataset:

```bash
# Extract to .cache directory
mkdir -p .cache
cd .cache
unzip ../setup/WikiTableQuestions-1.0.2-compact.zip
```

## Dataset Information

- **Source**: [WikiTableQuestions GitHub Repository](https://github.com/ppasupat/WikiTableQuestions)
- **Paper**: [Compositional Semantic Parsing on Semi-Structured Tables](https://arxiv.org/abs/1508.00305)
- **License**: Creative Commons Attribution Share Alike 4.0 International
- **Version**: 1.0.2

## Next Steps

After running the setup script, you can:

1. **Use with DSPy**: Load the processed data for training and evaluation
2. **Analyze the Data**: Explore question types, answer patterns, and table structures
3. **Build Models**: Implement table question answering models using the test data
4. **Evaluate Performance**: Use the exact match metric for model evaluation

For more information about using WTQ with DSPy, see the main project documentation in `../docs/wtq.md`.
