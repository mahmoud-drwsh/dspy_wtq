"""
Evaluation utilities for normalizing predictions and computing metrics.
"""

from __future__ import annotations

import re
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


def normalize_answer(text):
    """Normalize text for comparison - simplified version of evaluator.py normalize function."""
    if not text:
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower().strip()
    
    # Remove common punctuation and formatting
    # Remove commas from numbers (100,000 -> 100000)
    text = re.sub(r'(\d),(\d)', r'\1\2', text)
    # Remove trailing periods
    text = text.rstrip('.')
    # Normalize dashes (en-dash, em-dash to regular dash)
    text = re.sub(r'[–—−]', '-', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def is_answer_correct(predicted, expected_answers):
    """Check if predicted answer matches any expected answer using simplified evaluation."""
    if not predicted or not expected_answers:
        return False
    
    # Normalize predicted answer
    pred_norm = normalize_answer(predicted)
    
    # Handle "I don't know" responses
    if "i don't know" in pred_norm or "don't know" in pred_norm:
        # Check if any expected answer is also "I don't know" or similar
        for expected in expected_answers:
            exp_norm = normalize_answer(expected)
            if "i don't know" in exp_norm or "don't know" in exp_norm or "unknown" in exp_norm:
                return True
        # If expected answers are not "I don't know", then "I don't know" is incorrect
        return False
    
    # Check against each expected answer
    for expected in expected_answers:
        exp_norm = normalize_answer(expected)
        
        # Direct match
        if pred_norm == exp_norm:
            return True
        
        # Try to parse as numbers and compare
        try:
            pred_num = float(pred_norm.replace(',', ''))
            exp_num = float(exp_norm.replace(',', ''))
            if abs(pred_num - exp_num) < 1e-6:
                return True
        except (ValueError, TypeError):
            pass
    
    return False
