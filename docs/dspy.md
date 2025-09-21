# DSPy.ai — Complete & Detailed Usage Guide (v3.0.x)

> **Latest stable**: `dspy==3.0.3` (released Aug 31, 2025). Requires Python 3.10–3.13. ([PyPI][1])

---

## Table of Contents

* [What is DSPy?](#what-is-dspy)
* [Install & Upgrade](#install--upgrade)
* [Hello, DSPy (Quickstart)](#hello-dspy-quickstart)
* [Core Concepts](#core-concepts)

  * [Language Models (LMs)](#language-models-lms)
  * [Signatures](#signatures)
  * [Modules](#modules)
  * [Adapters & Types (3.0)](#adapters--types-30)
* [Programming Patterns](#programming-patterns)
* [Evaluation & Metrics](#evaluation--metrics)
* [Optimization (formerly “Teleprompters”)](#optimization-formerly-teleprompters)
* [Retrieval-Augmented Generation (RAG)](#retrieval-augmented-generation-rag)
* [Agents with Tools (ReAct)](#agents-with-tools-react)
* [Streaming, Async & Concurrency](#streaming-async--concurrency)
* [Caching & Cost Control](#caching--cost-control)
* [Saving, Loading & Deployment](#saving-loading--deployment)
* [Observability & MLflow](#observability--mlflow)
* [Best Practices & Gotchas](#best-practices--gotchas)
* [Cheatsheet & Further Reading](#cheatsheet--further-reading)

---

## What is DSPy?

**DSPy** is a **declarative framework** for building modular AI software. You write structured Python (not brittle prompt strings), and DSPy compiles your programs into effective prompts and/or fine-tuned weights. It scales from simple classifiers to sophisticated **RAG** pipelines and **agent** loops. ([Medium][2])

**DSPy 3.0 Highlights** (Aug 2025):

* New optimizers: **GRPO** (via Arbor), **GEPA**, **SIMBA**.
* Extensibility: **Adapters** (`ChatAdapter`, `JSONAdapter`, `TwoStepAdapter`) & **Types** (`Image`, `Audio`, Pydantic models, `ToolCalls`).
* Production: native **MLflow 3.0** integration; better async, caching, and save/load stability. ([GitHub][3])

---

## Install & Upgrade

```bash
pip install -U dspy
# Optional extras:
# pip install -U "dspy[anthropic]" "dspy[weaviate]" "dspy[langchain]" "dspy[mcp]"
```

* Verify the **latest** version and Python constraints on PyPI. ([PyPI][1])

---

## Hello, DSPy (Quickstart)

```python
import dspy

# 1) Pick a model provider & configure
lm = dspy.LM("openai/gpt-4o-mini")   # or anthropic/..., gemini/..., azure/..., ollama_chat/...
dspy.configure(lm=lm)

# 2) Define a task with a Signature, then use a Module
summarize = dspy.ChainOfThought("document -> summary")
out = summarize(document="DSPy lets you build AI with code, not brittle prompts.")
print(out.summary)
print(out.reasoning)   # ChainOfThought adds reasoning
```

* `dspy.configure(lm=...)` sets your default LM. Works across OpenAI, Gemini, Anthropic, Databricks, local backends (SGLang, Ollama), and OpenAI-compatible endpoints. ([DSPy][4])
* `dspy.ChainOfThought` augments your signature with a `reasoning` field. ([DSPy][5])

---

## Core Concepts

### Language Models (LMs)

Configure once; swap freely without changing your program.

```python
import dspy

# OpenAI
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini", api_key="..."))

# Gemini (Google AI Studio)
dspy.configure(lm=dspy.LM("gemini/gemini-2.5-pro-preview-03-25", api_key="..."))

# Anthropic
dspy.configure(lm=dspy.LM("anthropic/claude-3-opus-20240229", api_key="..."))

# Databricks
dspy.configure(lm=dspy.LM("databricks/databricks-meta-llama-3-1-70b-instruct"))

# Local via SGLang (OpenAI-compatible)
lm = dspy.LM(
    "openai/meta-llama/Meta-Llama-3-8B-Instruct",
    api_base="http://localhost:7501/v1", api_key="", model_type="chat"
)
dspy.configure(lm=lm)

# Local via Ollama
dspy.configure(lm=dspy.LM("ollama_chat/llama3.2", api_base="http://localhost:11434", api_key=""))
```

Examples and provider notes are in **Language Models**. ([DSPy][4])

**Generation controls & cache flag**:

```python
gpt4o = dspy.LM("openai/gpt-4o-mini", temperature=0.7, max_tokens=2048, cache=False)
dspy.configure(lm=gpt4o)
```

By default DSPy caches LM calls; set `cache=False` to bypass. ([DSPy][6])

**Advanced caching control**: `rollout_id` differentiates otherwise identical cached calls. ([DSPy][7])

---

### Signatures

A **Signature** declaratively specifies **inputs → outputs**. Use inline strings or class-based types.

```python
# Inline
classify = dspy.Predict("sentence -> sentiment: bool")
print(classify(sentence="I loved it").sentiment)

# Class-based with typing & docs
from typing import Literal
class Emotion(dspy.Signature):
    """Classify emotion."""
    sentence: str = dspy.InputField()
    sentiment: Literal["sadness","joy","love","anger","fear","surprise"] = dspy.OutputField()

predict = dspy.Predict(Emotion)
print(predict(sentence="this is amazing").sentiment)
```

Why signatures? They keep code modular and enable DSPy to optimize prompts/weights later. ([DSPy][8])

---

### Modules

**Modules** are building blocks implementing prompting strategies against a signature:

* `dspy.Predict` – basic predictor.
* `dspy.ChainOfThought` – step-by-step reasoning; adds `reasoning`.
* `dspy.ProgramOfThought` – code-first reasoning.
* `dspy.ReAct` – tool-using agent for the given signature.
* `dspy.MultiChainComparison` – compare multiple CoT outputs.
* `dspy.majority` / `BestOfN` – simple voting/selection utilities. ([DSPy][5])

Example:

```python
qa = dspy.ChainOfThought("question -> answer", n=5)  # multiple candidates
resp = qa(question="Why is ColBERT notable?")
print(resp.answer)
print(resp.completions.answer)  # all candidates
```

See **Modules** for composition patterns and more examples. ([DSPy][5])

---

### Adapters & Types (3.0)

**Adapters** control how DSPy formats & parses messages to/from LMs:

* Default: `ChatAdapter` (field-marked messages).
* `JSONAdapter` for structured JSON I/O (great with Pydantic types).
* `TwoStepAdapter` and other advanced flows.
  Configure globally or per-context:

```python
import dspy, pydantic

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"), adapter=dspy.ChatAdapter())

class Item(pydantic.BaseModel):
    sku: str
    price: float

catalog = dspy.Predict("query -> results: list[Item]")  # Typed outputs!
print(catalog(query="bestsellers").results)
```

Adapters, typed I/O (`Image`, `Audio`, Pydantic models) and tool-call types are part of 3.0’s extensibility. ([DSPy][9])

---

## Programming Patterns

### Compose Programs

Compose multiple modules like regular Python; DSPy traces calls at compile-time.

```python
class SearchThenAnswer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.search = dspy.Predict("query -> passages: list[str]")
        self.answer = dspy.ChainOfThought("context: list[str], question -> answer")

    def forward(self, question: str):
        ctx = self.search(query=question).passages
        return self.answer(context=ctx, question=question)

app = SearchThenAnswer()
print(app("Who inherited Kinnairdy Castle?").answer)
```

Modules are composable and return typed `Prediction` objects. ([DSPy][5])

---

## Evaluation & Metrics

Evaluate systematically before/after optimization.

```python
from dspy.evaluate import Evaluate
from dspy.metrics import answer_exact_match, SemanticF1

def metric(example, pred) -> float:
    # choose a metric appropriate to your task
    return answer_exact_match(example.answer, pred.answer)

evaluator = Evaluate(devset=my_examples, metric=metric)
score = evaluator(app)  # higher is better
print(score)
```

DSPy provides `Evaluate` plus metrics like `answer_exact_match` and `SemanticF1`. ([DSPy][4])

---

## Optimization (formerly “Teleprompters”)

**Optimizers** tune your program’s prompts and/or model weights from a small train/dev set:

* Few-shot: `LabeledFewShot`, `BootstrapFewShot`, `KNNFewShot`
* Instruction: `MIPROv2`, `COPRO`, `GEPA`, `SIMBA`
* Finetuning: `BootstrapFinetune`
* Ensembles & program transforms: `Ensemble`, `BetterTogether` ([DSPy][10])

```python
from dspy.teleprompt import MIPROv2

optimizer = MIPROv2(metric=metric, max_bootstrapped_demos=40)
optimized_program = optimizer.compile(app, trainset=train_examples)

print(optimized_program("…").answer)
optimized_program.save("search_then_answer.optimized.json")
```

Optimizers accept your **program**, **metric**, and a few **train inputs** (5–10 often works). You can save/load optimized programs. ([DSPy][10])

> **New in 3.0:** RL (**GRPO** via Arbor), upgraded **MIPROv2**, reflective **GEPA**, and feedback-driven **SIMBA**. ([GitHub][3])

---

## Retrieval-Augmented Generation (RAG)

Build RAG with a retriever (e.g., **ColBERTv2**) plus a reasoning module.

```python
def retrieve(query: str) -> list[str]:
    return [x["text"] for x in dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")(query, k=3)]

rag = dspy.ChainOfThought("context: list[str], question -> answer")
question = "What's the name of the castle that David Gregory inherited?"
pred = rag(context=retrieve(question), question=question)
print(pred.answer)
```

Module-level RAG example (with ColBERTv2) and full RAG tutorial are available. ([DSPy][5])

---

## Agents with Tools (ReAct)

Use tools (Python, search, APIs) inside a **ReAct** loop:

```python
def evaluate_math(expr: str) -> float:
    return dspy.PythonInterpreter({}).execute(expr)

def search_wikipedia(query: str) -> list[str]:
    return [x["text"] for x in dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")(query, k=3)]

agent = dspy.ReAct("question -> answer: float", tools=[evaluate_math, search_wikipedia])
print(agent(question="What is 9362158 divided by David Gregory's birth year?").answer)
```

See tutorials on **Agents** and **Multi-Hop Retrieval** for larger, optimized agentic systems. ([DSPy][5])

---

## Streaming, Async & Concurrency

**Streaming**: stream tokens or status messages from any layer (adapter, module). (See Adapters and Streaming guides.) ([DSPy][9])

**Async**: wrap any program to call it concurrently:

```python
from dspy.utils import asyncify
async_app = asyncify(app)
# await async_app(question="…")
```

The async wrapper propagates your current DSPy settings (lm, adapter, etc.). ([DSPy][11])

---

## Caching & Cost Control

DSPy uses a **3-layer cache** (memory, disk, and provider-side prompt cache). Control globally or per-LM:

```python
import dspy, os

# Disable caches globally
dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)

# Or tweak policy/limits
dspy.configure_cache(disk_cache_dir=os.path.join(os.getcwd(), "cache"), memory_max_entries=250_000)

# Per-LM: disable cache for a model
nocache = dspy.LM("openai/gpt-4o-mini", cache=False)
dspy.configure(lm=nocache)
```

See the **Cache** tutorial (`configure_cache`) and FAQ for toggling/exporting caches. ([DSPy][12])

---

## Saving, Loading & Deployment

**Save/Load full program** (architecture + state):

```python
optimized_program.save("my_program.pkl")         # or .json (state-only via .save(save_program=False))
loaded = dspy.load("my_program.pkl")
print(loaded("…"))
```

* `dspy.load(path)` loads a program saved with `save_program=True`.
* `Module.load(path)` loads **state** into an existing module. ([DSPy][13])

Production deployment guidance (packaging, environments, endpoints) is covered in the **Deployment** docs. ([DSPy][14])

---

## Observability & MLflow

Track runs, traces, and optimizer training with **MLflow**:

```python
import dspy
# Configure LM and (optionally) MLflow tracking via env vars or mlflow.start_run(...)
# Then run programs/optimizers as usual; DSPy emits rich traces.
```

DSPy 3.0 integrates with **MLflow 3.0** for tracing and optimizer tracking; see **Tracking DSPy Optimizers**. ([DSPy][15])

---

## Best Practices & Gotchas

* **Start with a clear Signature**; keep field names semantically meaningful (e.g., `question` vs `answer`). ([DSPy][8])
* Prefer **typed outputs** (e.g., `list[Item]`, Pydantic models) with `JSONAdapter` when you need strict structure. ([DSPy][9])
* Use **ChainOfThought** or **ProgramOfThought** for complex reasoning; use **MultiChainComparison**/**BestOfN** to compare/vote. ([DSPy][5])
* **Evaluate** first; **optimize** with MIPROv2/GEPA/SIMBA; consider **BootstrapFinetune** when you have more data. ([DSPy][10])
* Control **caching** carefully (global + per-LM) for reproducibility, cost, and speed; use `rollout_id` when sampling variants. ([DSPy][16])
* **Save** your optimized programs and version them; load with `dspy.load()` in prod. ([DSPy][13])

---

## Cheatsheet & Further Reading

* **Cheatsheet** (quick snippets, cache config, module tips). ([DSPy][17])
* **Learning DSPy** (overview of stages & curriculum). ([DSPy][18])
* **Language Models** (all providers, OpenAI-compatible endpoints). ([DSPy][4])
* **Signatures / Modules / Adapters** (core programming docs). ([DSPy][8])
* **Optimizers** (MIPROv2, GEPA, SIMBA, etc.). ([DSPy][10])
* **RAG** tutorial & **Agents** tutorial. ([DSPy][19])
* **Saving & Loading** and **Deployment** guides. ([DSPy][20])

---

*This guide tracks **DSPy 3.0.x** (current: 3.0.3). For release notes and 3.0 upgrades, see GitHub Releases.* ([PyPI][1])

[1]: https://pypi.org/project/dspy/ "dspy · PyPI"
[2]: https://medium.com/%40anyuanay/building-an-llm-based-research-assistant-agent-using-dspy-8435ae35ae15?utm_source=chatgpt.com "Building an LLM-Based Research Assistant Agent Using ..."
[3]: https://github.com/stanfordnlp/dspy/releases "Releases · stanfordnlp/dspy · GitHub"
[4]: https://dspy.ai/learn/programming/language_models/ "Language Models - DSPy"
[5]: https://dspy.ai/learn/programming/modules/ "Modules - DSPy"
[6]: https://dspy.ai/learn/programming/language_models/?utm_source=chatgpt.com "Language Models"
[7]: https://dspy.ai/api/models/LM/?utm_source=chatgpt.com "dspy.LM"
[8]: https://dspy.ai/learn/programming/signatures/ "Signatures - DSPy"
[9]: https://dspy.ai/learn/programming/adapters/ "Adapters - DSPy"
[10]: https://dspy.ai/learn/optimization/optimizers/ "Optimizers - DSPy"
[11]: https://dspy.ai/api/utils/asyncify/?utm_source=chatgpt.com "asyncify"
[12]: https://dspy.ai/tutorials/cache/?utm_source=chatgpt.com "Cache"
[13]: https://dspy.ai/api/utils/load/?utm_source=chatgpt.com "load"
[14]: https://dspy.ai/tutorials/deployment/ "Deployment - DSPy"
[15]: https://dspy.ai/tutorials/optimizer_tracking/ "Tracking DSPy Optimizers - DSPy"
[16]: https://dspy.ai/api/utils/configure_cache/?utm_source=chatgpt.com "configure_cache"
[17]: https://dspy.ai/cheatsheet/?utm_source=chatgpt.com "DSPy Cheatsheet"
[18]: https://dspy.ai/learn/?utm_source=chatgpt.com "Learning DSPy"
[19]: https://dspy.ai/tutorials/rag/?utm_source=chatgpt.com "Tutorial: Retrieval-Augmented Generation (RAG)"
[20]: https://dspy.ai/tutorials/saving/?utm_source=chatgpt.com "Saving and Loading"
