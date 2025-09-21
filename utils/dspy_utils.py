"""
DSPy configuration and model setup utilities.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Optional, Tuple

import dspy  # type: ignore


def ping_ollama(api_base: str = "http://localhost:11434") -> Tuple[bool, Optional[str]]:
    """Best-effort check that Ollama is reachable and list models/tags."""
    try:
        req = urllib.request.Request(f"{api_base.rstrip('/')}/api/tags")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        tags = [t.get("name") for t in data.get("models", []) if isinstance(t, dict)]
        return True, ", ".join(tags) if tags else "(no local models found)"
    except Exception as ex:  # broad: we only need a hint
        return False, f"{type(ex).__name__}: {ex}"


def configure_dspy(
    model: str = "gemma3:4b",
    api_base: str = "http://localhost:11434",
    temperature: float = 0.1,
    max_tokens: int = 512,
    disable_cache: bool = False,
):
    """Configure DSPy with Ollama model settings."""
    if disable_cache:
        try:
            dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
        except Exception:
            pass
    
    lm_id = f"ollama_chat/{model}"
    lm = dspy.LM(lm_id, api_base=api_base, temperature=temperature, max_tokens=max_tokens)
    dspy.configure(lm=lm)
    return dspy


def build_module(dspy, use_cot: bool = True):
    """Build a DSPy module (ChainOfThought or Predict)."""
    signature = "table_text, question -> answer"
    if use_cot:
        return dspy.ChainOfThought(signature)
    else:
        return dspy.Predict(signature)
