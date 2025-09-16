# DSPy modules — Predict, ChainOfThought, and ReAct (with examples from the official docs)

Below is a practical, doc-accurate guide to the three core modules you asked about. I’ll show how to declare each one, call it, and use its most important options—**all aligned with DSPy’s official documentation** (citations after each part).

---

## Quick setup (one time)

```python
import dspy

# pick any provider supported by LiteLLM; OpenAI shown as an example
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)
```

This is the documented way to configure your LM in DSPy; you can also pass an API key directly when constructing `dspy.LM`. ([DSPy][1])

---

# 1) `dspy.Predict`

### What it does

`Predict` is the **basic** DSPy module: you give it a **Signature** (your I/O contract) and it prompts the model once to map inputs → outputs. You can set default LM kwargs at construction (e.g., `temperature`, `n`), and even override them per call via a `config` dict. It also supports async (`acall`) and parallel batching via `.batch(...)`. ([DSPy][2])

### Minimal usage

```python
import dspy

class Sentiment(dspy.Signature):
    sentence: str = dspy.InputField()
    sentiment: bool = dspy.OutputField()

clf = dspy.Predict(Sentiment, temperature=0.0)   # default LM kwargs

out = clf(sentence="it's a charming and often affecting journey.")
print(out.sentiment)   # -> True/False
```

That usage pattern—**declare with a signature → call with inputs → read outputs**—is the standard, and the docs show an equivalent sentiment example. ([DSPy][3])

### Per-call overrides and multiple completions

```python
# request 3 completions on this call only
out = clf(
    sentence="surprisingly flat and meandering",
    config={"n": 3, "temperature": 0.7}
)

# out.completions.sentiment is a list (one per completion)
print(out.completions.sentiment)
```

Passing `config={...}` at call-time overrides the defaults set on the module (e.g., `n`, `temperature`). The `Predict` API doc shows the `config` override explicitly. ([DSPy][2])

### Batch processing (parallel)

```python
examples = [
    dspy.Example(sentence="delightful!") ,
    dspy.Example(sentence="a slog"),
]
results = clf.batch(examples)   # returns processed Examples
```

Batch execution uses the shared `.batch(...)` provided on modules, processing multiple `dspy.Example` items in parallel. ([DSPy][2])

---

# 2) `dspy.ChainOfThought`

### What it does

`ChainOfThought` is a thin wrapper that **adds a reasoning field** to your signature and instructs the model to “think step by step” before producing the final outputs—often a drop-in improvement over `Predict`. You can still request multiple completions with `n`, and you can inspect the model’s reasoning and the list of completions. ([DSPy][4])

### Minimal usage + inspecting reasoning

```python
qa = dspy.ChainOfThought("question -> answer", n=5)
pred = qa(question="What's something great about the ColBERT retrieval model?")

print(pred.reasoning)      # auto-injected rationale text
print(pred.answer)         # final answer
print(pred.completions.answer)  # list of answers from the n completions
```

The docs demonstrate (1) swapping in `ChainOfThought('question -> answer', n=5)`, (2) accessing `reasoning`, and (3) reading the `completions` lists. ([DSPy][3])

### How the “reasoning” field is added

Internally, the module prepends a `reasoning` output field (often with a prompt prefix like “Let’s think step by step…”) to your signature and calls a `Predict` under the hood—so it remains composable and compatible with any signature. ([DSPy][4])

---

# 3) `dspy.ReAct`

### What it does

`ReAct` (Reasoning + Acting) builds a **tool-using agent** around your signature. You provide a list of tools (plain functions or `dspy.Tool`), and the module iteratively (up to `max_iters`) decides a next thought, a tool to call with JSON args, observes the result, and repeats—then **extracts the final outputs** for your signature. It’s generic over any signature (thanks to signature polymorphism). ([DSPy][5])

### A tiny tool-using example

```python
import dspy
from typing import List

# Tool 1: tiny “DB” of facts
DB = {"Tokyo": "sunny", "Jakarta": "humid"}
def get_weather(city: str) -> str:
    return f"The weather in {city} is {DB.get(city, 'unknown')}."

# Build an agent: inputs→outputs contract, plus your tools.
react = dspy.ReAct("question -> answer", tools=[get_weather], max_iters=5)

pred = react(question="What's the weather in Tokyo?")
print(pred.answer)

# You can also inspect the full agent trajectory:
print(pred.trajectory)  # thoughts, chosen tools, args, and observations
```

The official API page shows the same constructor (`ReAct(signature, tools, max_iters=10)`) and a weather example. The module wraps each tool as a `dspy.Tool` (if you pass a plain function), injects a special **`finish`** tool, and builds an internal signature with fields like `trajectory`, `next_thought`, `next_tool_name` (a `Literal` over tool names), and `next_tool_args` (a `dict`), plus a fallback extraction step using `ChainOfThought`. ([DSPy][5])

### Async usage

All three modules support async via `acall(...)`/`aforward(...)`. With `ReAct`, you can also pass `max_iters` at call-time; the implementation pops it from call args. ([DSPy][5])

---

## Shared building blocks you’ll use with all modules

### 1) Signatures (string or class)

Signatures are **declarative I/O contracts** that tell the model what you want (inputs/outputs, types, and optional descriptions). You can define them as a short string (`"question -> answer: float"`) or a class with typed fields (`dspy.InputField` / `dspy.OutputField`). Multiple outputs are supported. ([DSPy][6])

```python
class BasicQA(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="often between 1 and 5 words")
```

The cheatsheet and “Signatures” pages show both styles and stress that Signatures specify **what** to produce, not **how** to prompt. ([DSPy][7])

### 2) Module config & completions

You can pass LM kwargs at construction (e.g., `temperature`, `n`) and **override per-call** via `config={...}`. When `n>1`, you get `.completions` (lists or a list of `Prediction` objects). This is documented for `Predict` and illustrated for `ChainOfThought`. ([DSPy][2])

### 3) Usage tracking, caching, streaming (optional)

* **Usage tracking**: turn it on and read token usage from any `Prediction`.

  ```python
  dspy.settings.configure(track_usage=True)
  usage = pred.get_lm_usage()
  ```

  ([DSPy][3])
* **Caching**: enable memory/disk caching to skip repeated calls.

  ```python
  dspy.configure_cache(enable_memory_cache=True, enable_disk_cache=True)
  ```

  ([DSPy][8])
* **Streaming**: wrap your program with `dspy.streamify(...)` and attach listeners to stream tokens from string fields. ([DSPy][9])

### 4) Saving & loading

Modules (and full programs) can be saved and later loaded—handy after compiling/optimizing. See the “Saving and Loading” tutorial and each module’s `load(...)`/`load_state(...)`. ([DSPy][10])

---

## Putting it together: one file, three modules

```python
import dspy

# LM config
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# --- Predict ---
class Sentiment(dspy.Signature):
    sentence: str = dspy.InputField()
    sentiment: bool = dspy.OutputField()

sentiment = dspy.Predict(Sentiment, temperature=0.0)

# --- ChainOfThought ---
cot = dspy.ChainOfThought("question -> answer", n=3)

# --- ReAct ---
def get_weather(city: str) -> str:
    return {"Tokyo": "sunny", "Jakarta": "humid"}.get(city, "unknown")

agent = dspy.ReAct("question -> answer", tools=[get_weather], max_iters=5)

# Run all three
print(sentiment(sentence="it's a charming and often affecting journey.").sentiment)
print(cot(question="Two dice are tossed. Probability sum equals two?").answer)
print(agent(question="What's the weather in Tokyo?").answer)
```

* The **usage patterns and options** mirror the official “Modules” page and each module’s API reference; the math-and-agents examples shown there are analogous to these. ([DSPy][3])
* The **constructor parameters and behaviors** (e.g., `config` overrides, reasoning field for CoT, ReAct’s internal agent loop and special fields) follow the API pages for each module. ([DSPy][2])

---

## Where to read more in the official docs

* **Modules overview & examples** (how to declare, call, completions, CoT swap-in, usage tracking): Modules page. ([DSPy][3])
* **`Predict` API** (constructor, `config` overrides, batch, async): Predict reference. ([DSPy][2])
* **`ChainOfThought` API** (auto “reasoning” field, internals): ChainOfThought reference. ([DSPy][4])
* **`ReAct` API** (tools list, `finish` tool, trajectory, fallback extraction): ReAct reference. ([DSPy][5])
* **Signatures** (string vs class, typed fields, multiple outputs): Signatures & Cheatsheet. ([DSPy][6])
* **LM configuration**: Language Models guide. ([DSPy][1])

[1]: https://dspy.ai/learn/programming/language_models/?utm_source=chatgpt.com "Language Models"
[2]: https://dspy.ai/api/modules/Predict/ "Predict - DSPy"
[3]: https://dspy.ai/learn/programming/modules/ "Modules - DSPy"
[4]: https://dspy.ai/api/modules/ChainOfThought/ "ChainOfThought - DSPy"
[5]: https://dspy.ai/api/modules/ReAct/ "ReAct - DSPy"
[6]: https://dspy.ai/learn/programming/signatures/?utm_source=chatgpt.com "Signatures"
[7]: https://dspy.ai/cheatsheet/?utm_source=chatgpt.com "DSPy Cheatsheet"
[8]: https://dspy.ai/api/utils/configure_cache/?utm_source=chatgpt.com "configure_cache"
[9]: https://dspy.ai/tutorials/streaming/?utm_source=chatgpt.com "Streaming"
[10]: https://dspy.ai/tutorials/saving/?utm_source=chatgpt.com "Saving and Loading"
