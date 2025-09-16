"""
Utilities for loading the WikiTableQuestions (WTQ) test split.

Exposes a high-level function to load all test examples with their associated
tables, so downstream scripts can consume ready-to-use question/answers + table
data without custom extraction or file handling.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import requests


def _download_wtq_zip() -> Path:
    """Download the WTQ compact zip into ./setup/ and return its path.

    If the file already exists in ./setup/, no download occurs.
    """
    data_url = (
        "https://github.com/ppasupat/WikiTableQuestions/releases/download/"
        "v1.0.2/WikiTableQuestions-1.0.2-compact.zip"
    )

    project_root = Path(__file__).resolve().parent.parent  # utils/ -> project root
    setup_dir = project_root / "setup"
    setup_dir.mkdir(exist_ok=True)
    zip_file_path = setup_dir / "WikiTableQuestions-1.0.2-compact.zip"

    if zip_file_path.exists():
        return zip_file_path

    # Stream to file to avoid holding the entire zip in memory
    with requests.get(data_url, stream=True) as resp:
        resp.raise_for_status()
        with open(zip_file_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return zip_file_path


def ensure_wtq_data(zip_file_path: Optional[Path] = None, cache_dir: Optional[Path] = None) -> Path:
    """Ensure WTQ data is extracted locally and return the `data/` directory path.

    - Uses `zip_file_path` if provided; otherwise looks for the zip next to caller script.
    - Downloads the zip if missing.
    - Extracts into project `.cache` (or provided `cache_dir`).
    - Returns the path to `.../WikiTableQuestions/data` directory.
    """
    # Determine a sensible default root based on this file location
    script_dir = Path(__file__).resolve().parent.parent  # utils/ -> project root
    project_root = script_dir

    if zip_file_path is None:
        # Assume a copy may live next to the setup script (project_root/setup/)
        candidate = project_root / "setup" / "WikiTableQuestions-1.0.2-compact.zip"
        zip_file_path = candidate

    cache_dir = cache_dir or (project_root / ".cache")
    cache_dir.mkdir(exist_ok=True)

    existing_data_dir = cache_dir / "WikiTableQuestions" / "data"
    if existing_data_dir.exists():
        return existing_data_dir

    # Ensure a zip exists (download if missing)
    if not Path(zip_file_path).exists():
        zip_file_path = _download_wtq_zip()

    # Extract the zip file to .cache
    import zipfile

    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(cache_dir)

    data_dir = cache_dir / "WikiTableQuestions" / "data"

    return data_dir


def read_table_from_file(table_name: str, root_dir: str) -> Dict[str, object]:
    """Read a normalized table (.tsv) from the WTQ dataset root directory.

    - `root_dir` should be the WTQ root (the directory that contains `csv/` and `data/`).
    - `table_name` is usually like `csv/203-csv/733.csv`; we normalize to `.tsv`.
    Returns: {header: List[str], rows: List[List[str]], name: str}
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


def load_wtq_test_split(data_dir: Path) -> List[Dict[str, object]]:
    """Load and parse the WTQ test split from a given `data_dir` (no extraction)."""
    test_file_path = Path(data_dir) / "pristine-unseen-tables.tsv"

    test_data: List[Dict[str, object]] = []
    with open(test_file_path, encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            example_id, question, table_name, answer = line.strip("\n").split("\t")
            answers = answer.split("|")
            test_data.append(
                {
                    "id": example_id,
                    "question": question,
                    "answers": answers,
                    "table_name": table_name,
                }
            )
    return test_data


def load_wtq_test_questions_with_tables(
    data_dir: Optional[Path] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, object]]:
    """Return test examples joined with their full tables.

    - Ensures WTQ data if `data_dir` not provided.
    - Returns a list of dicts: {id, question, answers, table_name, table}.
    - If `limit` is provided, returns only the first `limit` examples.
    """
    data_dir = data_dir or ensure_wtq_data()
    examples = load_wtq_test_split(data_dir)
    if limit is not None:
        examples = examples[: max(0, limit)]

    joined: List[Dict[str, object]] = []
    for ex in examples:
        try:
            # WTQ table files live under the dataset root (sibling of `data/`)
            table = read_table_from_file(ex["table_name"], root_dir=str(Path(data_dir).parent))
        except Exception as e:
            table = {"error": f"Failed to load table: {e}", "name": ex.get("table_name")}

        joined.append({**ex, "table": table})
    return joined
