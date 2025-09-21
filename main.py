#!/usr/bin/env python3
"""
WTQ × DSPy × Ollama — single-file baseline (repo-integrated, no HF)

- **Data loading:** uses your repo utilities:
    from utils import load_wtq_test_questions_with_tables
  This loads WTQ data from the cache (shape: id, question, answers, table:{header,rows,name}).

- Single DSPy module (Predict by default; optional CoT) over a local Ollama model.
- Writes predictions (predictions.txt, predictions.jsonl) and computes Denotation Accuracy (metrics.json).

Configuration:
  All settings are loaded from config.json. Edit this file to customize behavior.

Usage:
  # Ensure Ollama is running and model is pulled:
  ollama serve
  ollama pull llama3.2
  
  # Run with default config.json:
  uv run main.py
  
  # Run with custom config file:
  uv run main.py --config custom_config.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# Import utilities
from utils import (
    build_module,
    configure_dspy,
    denotation_accuracy,
    eprint,
    ensure_output_dir,
    human_table_preview,
    load_wtq_test_questions_with_tables,
    ping_ollama,
    serialize_table_for_prompt,
    split_prediction,
)

# ---------------------------
# Global configuration knobs
# ---------------------------

# Model / generation
OLLAMA_MODEL: str = "gemma3:4b"         # e.g., "llama3.2", "llama3.1:8b-instruct"
OLLAMA_URL: str = "http://localhost:11434"
TEMPERATURE: float = 0.1
MAX_TOKENS: int = 512
CTX_SIZE: int = 8192                    # informational; DSPy/Ollama handle context internally

# Program toggle
USE_COT: bool = True                   # if True, use dspy.ChainOfThought instead of dspy.Predict

# Data & IO
DATA_DIR: Optional[str] = None          # explicit '.cache/WikiTableQuestions/data' if desired
ROW_LIMIT: int = 30                     # max rows serialized from each table
COL_LIMIT: int = 10                     # max columns serialized from each table
TEST_LIMIT: Optional[int] = 200         # None or -1 for full test set; start small for smoke test
OUTPUT_DIR: str = "./outputs_wtq_dspy_ollama"

# Repro / caching
SEED: int = 42
DISABLE_DSPY_CACHE: bool = False        # DSPy caches calls by default; set True to bypass

# ---------------------------
# Main routine
# ---------------------------

def load_config(config_path: str = "config.json") -> dict:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        eprint(f"Error: Config file '{config_path}' not found.")
        eprint("Please create a config.json file or specify a valid config file with --config.")
        raise SystemExit(1)
    except json.JSONDecodeError as e:
        eprint(f"Error: Invalid JSON in config file '{config_path}': {e}")
        raise SystemExit(1)

def main(argv: Optional[Sequence[str]] = None) -> int:
    # Simple argument parser for config file path only
    parser = argparse.ArgumentParser(description="Answer WTQ test questions with DSPy + Ollama and evaluate Denotation Accuracy (repo-integrated, no HF).")
    parser.add_argument("--config", default="config.json", help="Path to configuration JSON file")
    args = parser.parse_args(argv)
    
    # Load configuration
    config = load_config(args.config)
    
    # Extract configuration values with fallbacks to global defaults
    model_config = config.get("model", {})
    program_config = config.get("program", {})
    data_config = config.get("data", {})
    output_config = config.get("output", {})
    repro_config = config.get("repro", {})
    
    # Model settings
    model_name = model_config.get("name", OLLAMA_MODEL)
    api_base = model_config.get("api_base", OLLAMA_URL)
    temperature = model_config.get("temperature", TEMPERATURE)
    max_tokens = model_config.get("max_tokens", MAX_TOKENS)
    ctx_size = model_config.get("ctx_size", CTX_SIZE)
    
    # Program settings
    use_cot = program_config.get("use_cot", USE_COT)
    disable_cache = program_config.get("disable_cache", DISABLE_DSPY_CACHE)
    
    # Data settings
    data_dir = data_config.get("data_dir", DATA_DIR)
    row_limit = data_config.get("row_limit", ROW_LIMIT)
    col_limit = data_config.get("col_limit", COL_LIMIT)
    test_limit = data_config.get("test_limit", TEST_LIMIT)
    
    # Output settings
    output_dir = output_config.get("output_dir", OUTPUT_DIR)
    
    # Repro settings
    seed = repro_config.get("seed", SEED)
    
    # Resolve limit
    limit_val: Optional[int] = None if test_limit is None or int(test_limit) < 0 else int(test_limit)

    outdir = ensure_output_dir(output_dir)

    # Friendly Ollama check
    ok, info = ping_ollama(api_base)
    if not ok:
        eprint(f"[warn] Could not reach Ollama at {api_base} ({info}).")
        eprint("       Ensure `ollama serve` is running and the model is pulled:")
        eprint(f"         ollama pull {model_name}")
    else:
        eprint(f"[ok] Ollama reachable. Local models: {info}")

    # Load data from WTQ cache
    try:
        data_dir_path = Path(data_dir) if data_dir else None
        examples = load_wtq_test_questions_with_tables(limit=limit_val, data_dir=data_dir_path)
    except Exception as ex:
        eprint(f"[error] Failed to load WTQ data: {type(ex).__name__}: {ex}")
        eprint("       Ensure the WTQ dataset is available in the cache.")
        eprint("       Run: uv run setup/extract_wtq_test_data.py")
        raise

    print(f"Loaded {len(examples)} WTQ test examples with tables")

    # Print a small preview like the example script
    preview_n = min(5, len(examples))
    for i in range(preview_n):
        ex = examples[i]
        print("-" * 60)
        print(f"Example {i+1}")
        print(f"ID: {ex.get('id')}")
        print(f"Table: {ex['table'].get('name')}")
        print(f"Question: {ex['question']}")
        print(f"Answers: {ex['answers']}")
        print(human_table_preview(ex["table"], n=2))

    # Configure DSPy + module
    try:
        dspy = configure_dspy(
            model=model_name,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            disable_cache=bool(disable_cache),
        )
    except Exception as ex:
        eprint("Failed to configure DSPy. Did you install `dspy` and run Ollama?")
        raise

    module = build_module(dspy, use_cot=bool(use_cot))

    # Inference
    preds_list: List[List[str]] = []
    golds_list: List[List[str]] = []
    predictions_txt_lines: List[str] = []
    jsonl_path = outdir / "predictions.jsonl"

    with open(jsonl_path, "w", encoding="utf-8") as fjsonl:
        for idx, ex in enumerate(examples, start=1):
            table_text = serialize_table_for_prompt(ex["table"], row_limit=row_limit, col_limit=col_limit)
            q = ex["question"]
            gold = list(ex["answers"])
            try:
                pred = module(table_text=table_text, question=q)
                ans_text = getattr(pred, "answer", "")
            except Exception as ex_call:
                ans_text = ""
                eprint(f"[warn] Inference failed for example {ex.get('id')}: {type(ex_call).__name__}: {ex_call}")
            pred_items = split_prediction(ans_text, gold_count=len(gold))
            preds_list.append(pred_items)
            golds_list.append(gold)
            predictions_txt_lines.append("|".join(pred_items) if pred_items else "")
            rec = {
                "id": ex.get("id"),
                "question": q,
                "gold": gold,
                "pred_text": ans_text,
                "pred_items": pred_items,
                "table_name": ex["table"].get("name"),
            }
            fjsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if idx % 50 == 0:
                print(f"... processed {idx} examples")

    # Save predictions.txt
    pred_path = outdir / "predictions.txt"
    with open(pred_path, "w", encoding="utf-8") as ftxt:
        for line in predictions_txt_lines:
            ftxt.write(line + "\n")

    # Evaluate DA
    da = denotation_accuracy(golds_list, preds_list)
    metrics = {
        "denotation_accuracy": da,
        "n": len(golds_list),
        "multi_answer_count": sum(1 for g in golds_list if len(g) > 1),
        "config": {
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "use_cot": bool(use_cot),
            "row_limit": row_limit,
            "col_limit": col_limit,
        },
    }
    metrics_path = outdir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fm:
        json.dump(metrics, fm, indent=2)

    print("-" * 60)
    print(f"Saved: {pred_path}")
    print(f"Saved: {jsonl_path}")
    print(f"Saved: {metrics_path}")
    print(f"Denotation Accuracy (string-set): {da:.4f} over {len(golds_list)} examples")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())