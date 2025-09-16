#!/usr/bin/env python3
"""
Chain-of-thought WTQ example (top-10)

- Loads the first 10 WikiTableQuestions test examples with their tables using
  `utils.load_wtq_test_questions_with_tables`.
- Uses DSPy with an OpenRouter-backed LM to generate chain-of-thought rationale
  and a final answer for each question.

Setup (see ./docs/ for project notes):
- Create venv: `uv venv --python 3.11`
- Install deps (if needed): `uv pip install -r requirements.txt`
- Configure OpenRouter:
  - Export an API key: `export OPENROUTER_API_KEY=sk-or-...`
  - Optional: this script maps `OPENROUTER_API_KEY` to `OPENAI_API_KEY` for DSPy.
  - Base URL: this script sets `OPENAI_BASE_URL=https://openrouter.ai/api/v1` if not set.

Run:
- `uv run examples/dspy_cot_wtq_top10.py`
"""

from __future__ import annotations

import os
from typing import List, Dict

import dspy

# Ensure OpenRouter works with DSPy's OpenAI backend
if not os.environ.get("OPENAI_BASE_URL"):
    os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

if not os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENROUTER_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.environ["OPENROUTER_API_KEY"]

MODEL_NAME = "openai/gpt-oss-20b:free"


def table_to_text(table: Dict, row_limit: int = 20) -> str:
    """Convert a WTQ table dict {header, rows, name} into a text snippet.

    Limits rows to `row_limit` for brevity.
    """
    header = table.get("header", [])
    rows: List[List[str]] = table.get("rows", [])
    name = table.get("name", "")

    lines = [f"Table: {name}"]
    if header:
        lines.append("Header: | " + " | ".join(header) + " |")
    preview = rows[: max(0, row_limit)]
    for r in preview:
        lines.append("Row: | " + " | ".join(r) + " |")
    if len(rows) > len(preview):
        lines.append(f"[... truncated {len(rows) - len(preview)} rows ...]")
    return "\n".join(lines)


class WTQCoTSignature(dspy.Signature):
    """Answer a question given its table context.

    question: the user's question about the table
    table_text: a compact, readable text form of the table
    rationale: chain-of-thought steps that reason to the answer
    answer: the final answer string
    """

    question = dspy.InputField(desc="question about the table")
    table_text = dspy.InputField(desc="compact table text, with header and rows")
    rationale = dspy.OutputField(desc="step-by-step reasoning")
    answer = dspy.OutputField(desc="final answer")


def main() -> None:
    # Configure DSPy with OpenRouter via OpenAI-compatible client
    # Use dspy.LM with OpenAI-compatible settings pointing to OpenRouter.
    lm = dspy.LM(MODEL_NAME)
    dspy.settings.configure(lm=lm)

    from utils import load_wtq_test_questions_with_tables

    print("Loading first 10 WTQ test examples with tables...")
    examples: List[Dict] = load_wtq_test_questions_with_tables(limit=10)
    print(f"Loaded {len(examples)} examples\n")

    cot = dspy.ChainOfThought(WTQCoTSignature)

    for i, ex in enumerate(examples, start=1):
        q = ex["question"]
        table_text = table_to_text(ex["table"], row_limit=20)

        print("-" * 60)
        print(f"Example {i}")
        print(f"ID: {ex.get('id')}  Table: {ex.get('table_name')}")
        print(f"Question: {q}")

        pred = cot(question=q, table_text=table_text)

        # Print chain-of-thought rationale and final answer
        print("Rationale:")
        print(pred.rationale)
        print("Answer:")
        print(pred.answer)


if __name__ == "__main__":
    main()
