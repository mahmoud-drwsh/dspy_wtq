# WikiTableQuestions (WTQ) Dataset - Comprehensive Analysis Report

*Generated using the new modular WTQ utilities*

## Executive Summary

The WikiTableQuestions (WTQ) dataset is a comprehensive benchmark for table question answering, containing **22,033 total examples** across three splits. This report provides detailed statistics and insights into the dataset's characteristics, structure, and complexity.

## Dataset Overview

### Split Distribution
| Split | Examples | Percentage | Description |
|-------|----------|------------|-------------|
| **Train** | 14,152 | 64.3% | Training examples with seen tables |
| **Validation** | 3,537 | 16.1% | Validation examples with seen tables |
| **Test** | 4,344 | 19.7% | Test examples with **unseen tables** |
| **Total** | 22,033 | 100% | Complete dataset |

### Key Characteristics
- **Total Tables**: 2,108 unique table files
- **Total Cells**: 3,691,610 cells across all tables
- **Average Table Size**: 168 cells per table
- **Question Types**: Primarily "what", "how", and "which" questions
- **Answer Types**: Mix of numeric (44.6%), textual (51.3%), and date-like (4.1%) answers

## Question Analysis

### Question Length Statistics

#### Overall Dataset
| Metric | Words | Characters |
|--------|-------|------------|
| **Minimum** | 1 | 7 |
| **Maximum** | 61 | 352 |
| **Mean** | 9.97 | 54.8 |
| **Median** | 9 | 52 |
| **Standard Deviation** | 3.4 | 19.5 |

#### By Split
| Split | Mean Words | Mean Characters | Max Words | Max Characters |
|-------|------------|-----------------|-----------|----------------|
| Train | 9.98 | 54.9 | 61 | 352 |
| Validation | 10.01 | 55.0 | 61 | 350 |
| Test | 9.99 | 54.8 | 61 | 352 |

### Question Types (Top 10 Starters)
| Question Starter | Count | Percentage |
|------------------|-------|------------|
| **what** | 5,960 | 27.1% |
| **how** | 5,538 | 25.1% |
| **which** | 3,718 | 16.9% |
| **who** | 1,572 | 7.1% |
| **in** | 395 | 1.8% |
| **the** | 316 | 1.4% |
| **name** | 283 | 1.3% |
| **did** | 236 | 1.1% |
| **when** | 156 | 0.7% |
| **was** | 139 | 0.6% |

## Answer Analysis

### Answer Distribution
| Metric | Value |
|--------|-------|
| **Total Answers** | 23,742 |
| **Unique Answers** | 9,886 |
| **Average Answers per Question** | 1.08 |
| **Maximum Answers per Question** | 56 |

### Answer Length Statistics
| Metric | Words | Characters |
|--------|-------|------------|
| **Minimum** | 1 | 1 |
| **Maximum** | 36 | 228 |
| **Mean** | 1.6 | 8.0 |
| **Median** | 1.0 | 5.0 |
| **Standard Deviation** | 1.2 | 8.8 |

### Answer Types
| Type | Count | Percentage |
|------|-------|------------|
| **Numeric** | 10,588 | 44.6% |
| **Textual/Other** | 12,178 | 51.3% |
| **Date-like** | 976 | 4.1% |

## Table Analysis

### Table Dimensions

#### Overall Statistics
| Metric | Rows | Columns |
|--------|------|---------|
| **Minimum** | 4 | 3 |
| **Maximum** | 753 | 25 |
| **Mean** | 25.4 | 6.4 |
| **Median** | 14.0 | 6.0 |
| **Standard Deviation** | 40.1 | 2.0 |

#### Size Distribution
| Metric | Total Cells |
|--------|-------------|
| **Minimum** | 20 |
| **Maximum** | 3,832 |
| **Mean** | 162.3 |
| **Median** | 90.0 |
| **Standard Deviation** | 255.5 |

### Cell Content Analysis
| Metric | Value |
|--------|-------|
| **Total Cells** | 3,691,610 |
| **Empty Cells** | 192,427 |
| **Empty Cell Percentage** | 5.21% |

#### Cell Content Length
| Metric | Words | Characters |
|--------|-------|------------|
| **Minimum** | 1 | 1 |
| **Maximum** | 360 | 2,009 |
| **Mean** | 1.9 | 10.7 |
| **Median** | 1.0 | 7.0 |
| **Standard Deviation** | 3.0 | 19.9 |

### Header Analysis
| Metric | Words per Header |
|--------|------------------|
| **Minimum** | 0 |
| **Maximum** | 15 |
| **Mean** | 1.32 |
| **Median** | 1.0 |
| **Standard Deviation** | 0.74 |

### Table-Level Content Analysis
| Metric | Words per Table | Characters per Table |
|--------|-----------------|---------------------|
| **Minimum** | 34 | 120 |
| **Maximum** | 6,374 | 40,732 |
| **Mean** | 304.4 | 1,704.9 |
| **Median** | 168.0 | 960.0 |
| **Standard Deviation** | 408.2 | 2,285.1 |

#### By Split
| Split | Mean Words | Mean Characters | Max Words | Max Characters |
|-------|------------|-----------------|-----------|----------------|
| Train | 304.0 | 1,708.0 | 6,374 | 40,732 |
| Validation | 298.3 | 1,679.4 | 5,648 | 37,046 |
| Test | 310.7 | 1,715.4 | 5,675 | 33,675 |

### Token Count Estimates
| Tokenization Method | Min Tokens | Max Tokens | Mean Tokens | Median Tokens |
|---------------------|------------|------------|-------------|---------------|
| **Simple (Word-based)** | 34 | 6,374 | 304.4 | 168.0 |
| **Tiktoken (GPT-style)** | 46 | 9,099 | 425.0 | 240.0 |
| **SentencePiece** | 58 | 11,452 | 540.7 | 305.0 |

*Note: Token counts are estimates based on common tokenization patterns. Actual counts may vary depending on the specific tokenizer implementation.*

#### Token Counts by Split
| Split | Simple Tokens (Mean) | Tiktoken Est. (Mean) | SentencePiece Est. (Mean) |
|-------|---------------------|---------------------|---------------------------|
| Train | 304.0 | 426.5 | 542.7 |
| Validation | 298.3 | 420.3 | 534.5 |
| Test | 310.7 | 424.0 | 539.2 |

## Topic Distribution

### Top Table Topics (by CSV directory)
| Topic | Count | Percentage |
|-------|-------|------------|
| **204-csv** | 7,749 | 36.8% |
| **203-csv** | 5,892 | 28.0% |
| **202-csv** | 255 | 1.2% |
| **201-csv** | 167 | 0.8% |
| **200-csv** | 89 | 0.4% |

*Note: Topics are derived from the CSV directory structure in the original dataset.*

## Complexity Analysis

### Dataset Complexity Indicators

1. **Question Complexity**
   - Questions range from simple 1-word queries to complex 61-word questions
   - Average question length of ~10 words indicates moderate complexity
   - High variety in question starters suggests diverse reasoning patterns

2. **Table Complexity**
   - Tables vary dramatically in size (20 to 3,832 cells)
   - Large standard deviation in row counts (40.1) indicates high variability
   - Relatively consistent column counts (mean: 6.4, std: 2.0)

3. **Answer Complexity**
   - 44% numeric answers suggest significant mathematical reasoning requirements
   - Multi-answer questions (up to 56 answers) indicate complex aggregation tasks
   - Short average answer length (1.6 words) suggests precise, factual answers

4. **Generalization Challenge**
   - Test set uses **unseen tables**, creating a true generalization challenge
   - 2,108 unique table files across diverse topics
   - 4.98% empty cells add noise and require robust handling

## Key Insights

### Strengths of the Dataset
1. **Scale**: Large dataset with 22K+ examples provides robust training data
2. **Diversity**: Wide range of table sizes, question types, and topics
3. **Real-world**: Based on Wikipedia tables with natural language questions
4. **Generalization**: Test set with unseen tables tests true generalization ability

### Challenges for Models
1. **Size Variability**: Tables range from 20 to 3,832 cells
2. **Question Complexity**: Up to 61-word questions requiring complex reasoning
3. **Answer Types**: Mix of numeric, textual, and date answers
4. **Data Quality**: 5.21% empty cells and varying table structures
5. **Multi-answer**: Some questions have up to 56 correct answers
6. **Token Limits**: Tables can require 46-11,452 tokens depending on tokenization method

### Recommended Model Considerations
1. **Variable Input Handling**: Models must handle tables of vastly different sizes
2. **Multi-step Reasoning**: Complex questions may require multi-hop reasoning
3. **Type Awareness**: Different answer types may require different processing
4. **Robustness**: Handle missing data and varying table structures
5. **Precision**: Short, precise answers require accurate extraction
6. **Token Management**: Consider token limits when processing large tables (up to 11K+ tokens)

## Technical Notes

### Data Loading
This analysis was generated using the new modular WTQ utilities:
- `utils.dataset_loader`: Handles dataset downloading and extraction
- `utils.splits_loader`: Loads and processes different dataset splits

### Analysis Methodology
- **Question Analysis**: Word and character count statistics, question type classification
- **Answer Analysis**: Length statistics, type classification (numeric/date/textual)
- **Table Analysis**: Dimension statistics, cell content analysis, header analysis
- **Topic Analysis**: Distribution based on CSV directory structure

### Limitations
- Topic classification is based on directory structure, not semantic analysis
- Answer type classification uses simple heuristics
- Cell content analysis includes headers and data cells together

---

*Report generated on: September 21, 2025*  
*Dataset Version: WikiTableQuestions v1.0.2*  
*Analysis Tools: Custom Python scripts using new modular WTQ utilities*
