"""
Splits loading utilities for WikiTableQuestions (WTQ).

Provides functions to load different WTQ dataset splits (train, validation, test)
and return them as a dictionary for easy access.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

from .dataset_loader import ensure_wtq_data, get_wtq_root_dir


def read_table_from_file(table_name: str, root_dir: str) -> Dict[str, object]:
    """Read a normalized table (.tsv) from the WTQ dataset root directory.

    Args:
        table_name: Table name like `csv/203-csv/733.csv`; will be normalized to `.tsv`
        root_dir: WTQ root directory (contains `csv/` and `data/`)
        
    Returns:
        Dict with keys: header (List[str]), rows (List[List[str]]), name (str)
    """
    def extract_table_content(line: str) -> List[str]:
        vals = [val.replace("\n", " ").strip() for val in line.strip("\n").split("\t")]
        return vals

    rows: List[List[str]] = []
    # Normalize extension
    normalized = table_name.replace(".csv", ".tsv")

    # If the name is prefixed with 'csv/', strip for joining under root/csv
    if normalized.startswith("csv/"):
        rel_under_csv = normalized[len("csv/"):]
    else:
        rel_under_csv = normalized

    csv_dir = os.path.join(root_dir, "csv")
    candidates = [
        os.path.join(csv_dir, rel_under_csv),               # <root>/csv/<...>.tsv
        os.path.join(root_dir, normalized),                 # <root>/<normalized>
        os.path.join(root_dir, rel_under_csv),              # <root>/<...>.tsv
    ]

    table_path = None
    for cand in candidates:
        if os.path.exists(cand):
            table_path = cand
            break

    if table_path is None:
        # As a last resort, search under csv_dir
        for root, dirs, files in os.walk(csv_dir):
            if os.path.basename(rel_under_csv) in files:
                table_path = os.path.join(root, os.path.basename(rel_under_csv))
                break

    if table_path is None:
        raise FileNotFoundError(f"Could not resolve table path for {table_name} under {root_dir}")

    with open(table_path, "r", encoding="utf8") as table_f:
        table_lines = table_f.readlines()
        if not table_lines:
            header = []
        else:
            header = extract_table_content(table_lines[0])
            for line in table_lines[1:]:
                rows.append(extract_table_content(line))

    return {"header": header, "rows": rows, "name": normalized}


def load_wtq_split(data_dir: Path, split_name: str) -> List[Dict[str, object]]:
    """Load and parse a specific WTQ split from the data directory.
    
    Args:
        data_dir: Path to the WTQ data directory
        split_name: Name of the split file (e.g., 'training', 'pristine-unseen-tables', 'pristine-seen-tables')
        
    Returns:
        List of examples, each containing: id, question, answers, table_name
    """
    split_file_path = Path(data_dir) / f"{split_name}.tsv"
    
    if not split_file_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_file_path}")

    split_data: List[Dict[str, object]] = []
    with open(split_file_path, encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            example_id, question, table_name, answer = line.strip("\n").split("\t")
            answers = answer.split("|")
            split_data.append(
                {
                    "id": example_id,
                    "question": question,
                    "answers": answers,
                    "table_name": table_name,
                }
            )
    return split_data


def load_wtq_splits(data_dir: Optional[Path] = None) -> Dict[str, List[Dict[str, object]]]:
    """Load all available WTQ splits and return them as a dictionary.
    
    Args:
        data_dir: Path to the WTQ data directory. If None, ensures data is available.
        
    Returns:
        Dictionary with keys: 'train', 'validation', 'test'
        - 'train': training examples (from training.tsv)
        - 'validation': validation examples (from pristine-seen-tables.tsv) 
        - 'test': test examples (from pristine-unseen-tables.tsv)
    """
    data_dir = data_dir or ensure_wtq_data()
    
    splits = {}
    
    # Load training split
    try:
        splits['train'] = load_wtq_split(data_dir, 'training')
    except FileNotFoundError:
        splits['train'] = []
    
    # Load validation split (pristine-seen-tables)
    try:
        splits['validation'] = load_wtq_split(data_dir, 'pristine-seen-tables')
    except FileNotFoundError:
        splits['validation'] = []
    
    # Load test split (pristine-unseen-tables)
    try:
        splits['test'] = load_wtq_split(data_dir, 'pristine-unseen-tables')
    except FileNotFoundError:
        splits['test'] = []
    
    return splits


def load_wtq_splits_with_tables(
    data_dir: Optional[Path] = None,
    limit: Optional[Dict[str, int]] = None,
) -> Dict[str, List[Dict[str, object]]]:
    """Load all WTQ splits with their associated tables.
    
    Args:
        data_dir: Path to the WTQ data directory. If None, ensures data is available.
        limit: Optional dict with split names as keys and limits as values.
               E.g., {'train': 100, 'test': 50} to limit examples per split.
               
    Returns:
        Dictionary with keys: 'train', 'validation', 'test'
        Each split contains examples with full table data:
        {id, question, answers, table_name, table: {header, rows, name}}
    """
    data_dir = data_dir or ensure_wtq_data()
    root_dir = get_wtq_root_dir(data_dir)
    
    # Load basic splits
    splits = load_wtq_splits(data_dir)
    
    # Apply limits if specified
    if limit:
        for split_name, split_limit in limit.items():
            if split_name in splits and split_limit is not None:
                splits[split_name] = splits[split_name][:max(0, split_limit)]
    
    # Join with table data
    splits_with_tables = {}
    for split_name, examples in splits.items():
        joined: List[Dict[str, object]] = []
        for ex in examples:
            try:
                table = read_table_from_file(ex["table_name"], root_dir=str(root_dir))
            except Exception as e:
                table = {"error": f"Failed to load table: {e}", "name": ex.get("table_name")}
            
            joined.append({**ex, "table": table})
        splits_with_tables[split_name] = joined
    
    return splits_with_tables


def get_split_info(splits: Dict[str, List[Dict[str, object]]]) -> Dict[str, int]:
    """Get information about the loaded splits.
    
    Args:
        splits: Dictionary of splits as returned by load_wtq_splits()
        
    Returns:
        Dictionary with split names as keys and example counts as values
    """
    return {split_name: len(examples) for split_name, examples in splits.items()}


def get_split_summary(data_dir: Optional[Path] = None) -> Dict[str, int]:
    """Get a quick summary of all available splits without loading full data.
    
    Args:
        data_dir: Path to the WTQ data directory. If None, ensures data is available.
        
    Returns:
        Dictionary with split names as keys and example counts as values
    """
    data_dir = data_dir or ensure_wtq_data()
    splits = load_wtq_splits(data_dir)
    return get_split_info(splits)


# Compatibility function to maintain the same interface as the old wtq.py
def load_wtq_test_questions_with_tables(
    data_dir: Optional[Path] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, object]]:
    """Compatibility function for the old wtq.py interface.
    
    Loads only the test split with table data, maintaining the same signature
    as the original function for backward compatibility.
    
    Args:
        data_dir: Path to the WTQ data directory. If None, ensures data is available.
        limit: Optional limit on number of examples to return.
        
    Returns:
        List of test examples with full table data
    """
    limit_dict = {'test': limit} if limit is not None else None
    splits_with_tables = load_wtq_splits_with_tables(data_dir=data_dir, limit=limit_dict)
    return splits_with_tables.get('test', [])
