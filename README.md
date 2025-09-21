# CRUCIAL INSTRUCTIONS

- Create venv: `uv venv --python 3.11`
- Configure: Edit `config.json` to customize model, data paths, and other settings
- Run main: `uv run main.py`

## Configuration

All settings are now managed through `config.json` instead of command-line arguments:

- **Model settings**: model name, API base URL, temperature, max tokens
- **Program settings**: use Chain of Thought, disable caching
- **Data settings**: data directory, row/column limits
- **Output settings**: output directory
- **Repro settings**: random seed

Example configuration:
```json
{
  "model": {
    "name": "gemma3:4b",
    "api_base": "http://localhost:11434",
    "temperature": 0.1,
    "max_tokens": 512
  },
  "program": {
    "use_cot": true,
    "disable_cache": false
  },
  "data": {
    "data_dir": null,
    "test_limit": 200,
    "row_limit": 30,
    "col_limit": 10
  }
}
```

## WTQ Dataset Setup / Preview

- Quick preview: `uv run setup/extract_wtq_test_data.py`
  - Ensures the WikiTableQuestions dataset is present under `./.cache`.
  - Prints the first 10 test questions with their associated tables to the terminal (JSON lines).

## Load WTQ in Your Code

- High-level loader is exposed via `utils`:
  - `from utils import load_wtq_test_questions_with_tables`
  - `examples = load_wtq_test_questions_with_tables()`
    - Ensures data if missing, and returns a list of dicts:
      `{id, question, answers, table_name, table}` where `table = {header, rows, name}`.

## Example Script

- Run example iteration: `uv run examples/wtq_iter_example.py`
  - Loads a small sample and prints a concise preview of each example and table.
