# CRUCIAL INSTRUCTIONS

- Create venv: `uv venv --python 3.11`
- Run main: `uv run main.py`

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
