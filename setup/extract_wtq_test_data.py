#!/usr/bin/env python3
"""
WTQ quick setup and preview

- Uses the shared loader in `utils` to ensure data is available and then prints
  a preview (first 10 questions with their associated tables) to the terminal.
- Intention: help developers verify the dataset is ready without duplicating logic.

Configuration is in-script via module constants (no CLI flags).
"""

import json
import logging
from pathlib import Path
from typing import Optional
import sys

# Compute project root and ensure it's importable before importing utils
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import load_wtq_test_questions_with_tables

# Configuration (edit these as needed)
DATA_DIR: Optional[Path] = None  # If None, utils will auto-extract into .cache
LOG_TO_STDOUT: bool = True  # Always logs to terminal for verification


def configure_json_logger(name: str, to_stdout: bool = True) -> logging.Logger:
    """Create a simple logger that writes raw JSON lines to stdout."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    if to_stdout:
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)
    return logger

def main():
    """Print a preview (first 10 questions with tables) to the terminal.

    Uses `utils.load_wtq_test_questions_with_tables` to ensure data and load entries.
    """
    logger = configure_json_logger("wtq_setup_preview", to_stdout=LOG_TO_STDOUT)
    print("Printing first 10 questions and their tables to terminal...")
    examples = load_wtq_test_questions_with_tables(data_dir=DATA_DIR, limit=10)
    logger.info(json.dumps({
        "dataset": "WikiTableQuestions",
        "source": "https://github.com/ppasupat/WikiTableQuestions",
        "split": "pristine-unseen-tables",
        "preview_count": len(examples),
        "description": "Preview: first N questions with tables",
    }))
    for ex in examples:
        logger.info(json.dumps(ex, ensure_ascii=False))
    for ex in examples:
        logger.info(json.dumps(ex, ensure_ascii=False))
    print("Completed! Preview printed to terminal.")

if __name__ == "__main__":
    main()
