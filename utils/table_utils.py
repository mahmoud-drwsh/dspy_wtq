"""
Table serialization and formatting utilities.
"""

from __future__ import annotations

from typing import Dict, List


def serialize_table_for_prompt(table: Dict, row_limit: int = 30, col_limit: int = 10) -> str:
    """Serialize a table dictionary into a formatted string for use in prompts."""
    header: List[str] = list(table.get("header", []))[:col_limit]
    rows: List[List[str]] = [list(map(str, r[:col_limit])) for r in table.get("rows", [])[:row_limit]]
    lines: List[str] = []
    name = table.get("name") or "table"
    
    lines.append(f"Table: {name}")
    lines.append("Header: " + " | ".join(map(str, header)))
    for r in rows:
        lines.append("Row: " + " | ".join(map(str, r)))
    
    if len(table.get("rows", [])) > row_limit:
        lines.append(f"... ({len(table.get('rows', [])) - row_limit} more rows truncated)")
    if any(len(r) > col_limit for r in table.get("rows", [])):
        lines.append(f"... (columns truncated to first {col_limit})")
    
    return "\n".join(lines)


def human_table_preview(table: Dict, n: int = 2) -> str:
    """Create a human-readable preview of a table for display purposes."""
    header = table.get("header", [])
    rows = table.get("rows", [])[:n]
    out = []
    out.append(f"Header ({len(header)} cols): {header}")
    out.append(f"Rows: {len(table.get('rows', []))} total; showing first {len(rows)}")
    for r in rows:
        out.append(f"  {r}")
    return "\n".join(out)
