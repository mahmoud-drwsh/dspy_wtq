"""
File I/O utilities for handling data loading, CSV reading, and file operations.
"""

from __future__ import annotations

import csv
import gzip
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def eprint(*args, **kwargs):
    """Print to stderr."""
    print(*args, file=sys.stderr, **kwargs)


def ensure_output_dir(path: str) -> Path:
    """Create output directory if it doesn't exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def open_maybe_gzip(path: Path, mode: str = "rt", encoding: str = "utf-8"):
    """Open a file, handling gzip compression if the file ends with .gz."""
    if str(path).endswith(".gz"):
        return gzip.open(path, mode=mode, encoding=encoding)  # type: ignore
    return open(path, mode=mode, encoding=encoding)


def find_examples_jsonl(root: Path) -> Optional[Path]:
    """Find the examples JSONL file in the given directory."""
    candidates = [
        "test.examples.with-tables.jsonl",
        "test.examples.with_tables.jsonl",
        "test.examples.jsonl",
        "pristine-unseen-tables.examples.jsonl",
        "test.jsonl",
        "test.examples.with-tables.jsonl.gz",
        "test.examples.jsonl.gz",
    ]
    for name in candidates:
        p = root / name
        if p.exists():
            return p
    for p in sorted(root.glob("*.jsonl*")):
        if "test" in p.name or "pristine" in p.name:
            return p
    return None


def read_csv_table(csv_root: Path, name: str, col_limit: int) -> Tuple[List[str], List[List[str]]]:
    """Read a CSV/TSV table file and return header and rows."""
    rel = name.replace("\\", "/").lstrip("./")
    path = csv_root / rel
    if not path.exists():
        if not rel.startswith("csv/"):
            alt = csv_root / "csv" / rel
            if alt.exists():
                path = alt
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found for table name '{name}' under '{csv_root}'")
    
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
    
    if not rows:
        return [], []
    
    header = [str(x) for x in rows[0][:col_limit]]
    body = [[str(x) for x in r[:col_limit]] for r in rows[1:]]
    return header, body


def load_examples_fallback(
    wtq_dir: Path, 
    examples_jsonl: Optional[Path], 
    limit: Optional[int], 
    col_limit: int
) -> List[Dict]:
    """Load examples from JSONL file with fallback CSV loading."""
    wtq_dir = wtq_dir.resolve()
    src = examples_jsonl if examples_jsonl else find_examples_jsonl(wtq_dir)
    if src is None:
        raise FileNotFoundError(
            f"Could not find a test examples JSONL under '{wtq_dir}'. "
            "Provide --examples-jsonl or place a file like 'test.examples.with-tables.jsonl'."
        )
    
    src = src.resolve()
    csv_root = wtq_dir
    examples: List[Dict] = []
    
    with open_maybe_gzip(src, mode="rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            j = json.loads(line)
            _id = j.get("id") or j.get("example_id") or j.get("qid")
            q = j.get("question")
            answers = j.get("answers") or j.get("answer")
            if isinstance(answers, str):
                answers = [answers]
            if answers is None:
                answers = []
            
            t = j.get("table", {})
            tname = t.get("name") or j.get("table_id") or j.get("table_name")
            theader = t.get("header")
            trows = t.get("rows")
            
            if (not theader or not isinstance(theader, list)) or (not trows or not isinstance(trows, list)):
                if not tname:
                    raise ValueError(f"Example {_id} missing table.name; cannot load CSV")
                theader, trows = read_csv_table(csv_root, tname, col_limit=col_limit)
            
            ex = {
                "id": _id,
                "question": q,
                "answers": list(answers),
                "table": {
                    "header": list(theader),
                    "rows": [list(r) for r in trows],
                    "name": tname,
                },
            }
            examples.append(ex)
            if limit is not None and len(examples) >= int(limit):
                break
    
    return examples


def load_examples_repo_utils(limit: Optional[int], data_dir: Optional[Path]) -> List[Dict]:
    """
    Use user's repo utilities to load WTQ test examples with tables.
    Falls back to ensure_wtq_data() when data_dir is None.
    """
    try:
        from utils import ensure_wtq_data, load_wtq_test_questions_with_tables  # type: ignore
    except Exception as ex:
        raise RuntimeError(f"Repo utilities not available: {type(ex).__name__}: {ex}")
    
    if data_dir is None:
        data_dir = ensure_wtq_data()
    
    examples = load_wtq_test_questions_with_tables(data_dir=data_dir, limit=limit)
    
    # Normalize keys in case repo returns 'table_name' alongside 'table'
    for ex in examples:
        if "table" not in ex or not ex["table"]:
            # If only table_name is present, attempt to read via repo's read_table (not importing here)
            # but typically repo loader fills 'table' already.
            ex["table"] = {"header": [], "rows": [], "name": ex.get("table_name")}
    
    return examples
