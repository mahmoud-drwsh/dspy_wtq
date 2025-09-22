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

def configure_dspy_openrouter(
    model_name: str = "openrouter/deepseek/deepseek-r1-0528-qwen3-8b",
    max_tokens: int = 1024,
    context_length: int = 131072,
    track_usage: bool = False,
):
    """
    Configure DSPy with OpenRouter model settings.
    
    Args:
        model_name: The model to use (default: DeepSeek R1 via OpenRouter)
        max_tokens: Maximum tokens for generation
        context_length: Context window size
        track_usage: Whether to track token usage (can cause errors)
    
    Returns:
        The configured language model
    """
    # Configure the language model
    lm = dspy.LM(model_name, max_tokens=max_tokens, context_length=context_length, cache=False)
    dspy.configure(lm=lm)
    
    # Configure usage tracking and cache
    dspy.settings.configure(track_usage=track_usage)
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
    
    print(f"✅ DSPy configured with model: {model_name}")
    return lm


def print_token_usage(result):
    """Print token usage statistics from a DSPy result."""
    print("\nToken Usage Statistics:")
    try:
        usage_stats = result.get_lm_usage()
        if usage_stats:
            for model_name, stats in usage_stats.items():
                print(f"Model: {model_name}")
                print(f"Input tokens: {stats.get('prompt_tokens', 'N/A')}")
                print(f"Output tokens: {stats.get('completion_tokens', 'N/A')}")
                print(f"Total tokens: {stats.get('total_tokens', 'N/A')}")
                print(f"API calls: {stats.get('calls', 'N/A')}")
        else:
            print("No usage statistics available (tracking disabled)")
    except Exception as e:
        print(f"Usage tracking error: {e}")


def build_module(dspy, use_cot: bool = True):
    """Build a DSPy module (ChainOfThought or Predict)."""
    signature = "table_text, question -> answer"
    if use_cot:
        return dspy.ChainOfThought(signature)
    else:
        return dspy.Predict(signature)
