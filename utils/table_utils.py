"""
Table serialization and formatting utilities.
"""

from __future__ import annotations

from typing import Dict, List, Optional


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


def format_table_token_efficient(
    table: Dict, 
    question: Optional[str] = None, 
    delimiter: str = "|", 
    max_rows: int = 100
) -> str:
    """
    Format table as token-efficient delimited data with optional pruning.
    
    This is the most token-efficient format for LLM consumption:
    - No JSON/Markdown overhead (quotes, braces, borders)
    - Minimal scaffolding (no "Table:", "Headers:", "Data:" labels)
    - Smart column/row filtering based on question keywords
    - Value normalization (removes thousands separators, normalizes dates)
    - Clean delimited format (header row + data rows)
    
    Args:
        table: Table dict with 'header' and 'rows' keys
        question: Optional question for column/row filtering
        delimiter: Field separator (default "|")
        max_rows: Maximum number of rows to include
        
    Returns:
        Token-efficient delimited table string
    """
    headers = [h.strip() for h in table.get("header", [])]
    rows = table.get("rows", [])
    
    def clean_value(x):
        """Clean and normalize cell values for token efficiency."""
        if x is None:
            return ""
        x = str(x).strip()
        # Remove newlines and replace delimiter with space
        x = x.replace("\n", " ").replace(delimiter, " ")
        # Remove thousands separators for numeric efficiency
        x = x.replace(",", "")
        # Normalize dates (basic pattern)
        if "/" in x and len(x.split("/")) == 3:
            parts = x.split("/")
            if len(parts[2]) == 4:  # Assume MM/DD/YYYY format
                x = f"{parts[2]}-{parts[0].zfill(2)}-{parts[1].zfill(2)}"
        return x
    
    # Optional: Filter columns based on question keywords
    if question:
        question_words = set(question.lower().split())
        relevant_headers = []
        relevant_indices = []
        
        for i, header in enumerate(headers):
            header_words = set(header.lower().split())
            # Keep column if header matches question or if it's numeric/date-like
            # Also keep columns that might contain ranking/position data
            if (question_words.intersection(header_words) or 
                any(word in header.lower() for word in ['date', 'year', 'number', 'count', 'total', 'amount', 'value', 'placing', 'place', 'rank', 'position', 'finish', 'result'])):
                relevant_headers.append(header)
                relevant_indices.append(i)
        
        # If we filtered too aggressively, keep all columns
        if len(relevant_headers) < 2:
            relevant_headers = headers
            relevant_indices = list(range(len(headers)))
    else:
        relevant_headers = headers
        relevant_indices = list(range(len(headers)))
    
    # Filter rows based on question keywords (optional)
    if question and len(rows) > max_rows:
        question_words = set(question.lower().split())
        relevant_rows = []
        
        for row in rows:
            row_text = " ".join(str(cell) for cell in row).lower()
            if any(word in row_text for word in question_words):
                relevant_rows.append(row)
        
        # If filtering left us with too few rows, take first N rows instead
        if len(relevant_rows) < 10:
            relevant_rows = rows[:max_rows]
        else:
            relevant_rows = relevant_rows[:max_rows]
    else:
        relevant_rows = rows[:max_rows]
    
    # Build the compact format
    lines = []
    
    # Header row
    header_line = delimiter.join(clean_value(h) for h in relevant_headers)
    lines.append(header_line)
    
    # Data rows
    for row in relevant_rows:
        # Only include columns we're keeping
        filtered_row = [row[i] for i in relevant_indices if i < len(row)]
        row_line = delimiter.join(clean_value(v) for v in filtered_row)
        lines.append(row_line)
    
    return "\n".join(lines)
