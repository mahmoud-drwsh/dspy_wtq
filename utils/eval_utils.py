"""
Evaluation utilities for normalizing predictions and computing metrics.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import List


def normalize_token(s: str) -> str:
    """Normalize a token for comparison purposes."""
    s0 = s.strip().strip('"').strip("'").lower()
    s0 = " ".join(s0.split())
    s0 = s0.strip(" .,")
    t = s0.replace(",", "")
    if t.endswith("%"):
        t = t[:-1]
    if t.startswith("$"):
        t = t[1:]
    try:
        d = Decimal(t)
        return format(d, "f").rstrip("0").rstrip(".") if "." in str(d) else str(d)
    except (InvalidOperation, ValueError):
        pass
    return s0


def split_prediction(pred_text: str, gold_count: int) -> List[str]:
    """Split a prediction text into individual answer tokens."""
    if not isinstance(pred_text, str):
        pred_text = str(pred_text)
    txt = pred_text.strip()
    if gold_count <= 1:
        return [normalize_token(txt)]
    
    raw = [x.strip() for x in txt.split("|") if x.strip()]
    if len(raw) <= 1:
        raw = [x.strip() for x in txt.split(",") if x.strip()]
    
    return [normalize_token(x) for x in raw] if raw else [normalize_token(txt)]


def denotation_accuracy(golds: List[List[str]], preds: List[List[str]]) -> float:
    """Compute denotation accuracy by comparing normalized answer sets."""
    assert len(golds) == len(preds)
    correct = 0
    for g, p in zip(golds, preds):
        gset = {normalize_token(x) for x in g}
        pset = {normalize_token(x) for x in p}
        if gset == pset:
            correct += 1
    return correct / len(golds) if golds else 0.0
