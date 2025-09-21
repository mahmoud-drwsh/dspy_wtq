#!/usr/bin/env python3
"""
Example: iterate WTQ test questions with their tables.

This uses utils.splits_loader.load_wtq_test_questions_with_tables to load ready-to-use
examples. It prints a short summary plus a small peek at each example's table.

Run:
  python examples/wtq_iter_example.py

Notes:
- If the WTQ data is not present locally, the utils will attempt to download
  and extract it under ./.cache.
"""

from __future__ import annotations

from typing import List, Dict
from pathlib import Path
import sys

# Ensure project root is on sys.path for local imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import load_wtq_test_questions_with_tables


def main() -> None:
    # Load a small sample to keep output concise
    examples: List[Dict] = load_wtq_test_questions_with_tables(limit=5)
    print(f"Loaded {len(examples)} WTQ test examples with tables")

    for i, ex in enumerate(examples, start=1):
        q = ex["question"]
        answers = ex["answers"]
        table = ex["table"]
        header = table.get("header", [])
        rows = table.get("rows", [])
        tname = table.get("name")

        print("-" * 60)
        print(f"Example {i}")
        print(f"ID: {ex.get('id')}")
        print(f"Table: {tname}")
        print(f"Question: {q}")
        print(f"Answers: {answers}")
        print(f"Header ({len(header)} cols): {header}")
        print(f"Rows: {len(rows)} total")
        if rows:
            preview = rows[:2]
            print("First 2 rows:")
            for r in preview:
                print("  ", r)


if __name__ == "__main__":
    main()
