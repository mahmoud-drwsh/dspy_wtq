Here’s a compact, academic take on WTQ—and then a dead-simple DSPy+Python walkthrough you can run.

# What is WTQ?

**WikiTableQuestions (WTQ)** is a benchmark for *semantic parsing and question answering over semi-structured tables*. Each example pairs a natural-language question with a single HTML/TSV table from Wikipedia; the system must compute the correct denotation (answer) by inducing a latent program (logical form) that operates over the table. The original release contains **≈22k question–answer pairs across ≈2.1k Wikipedia tables**, with test tables disjoint from training tables to stress compositional generalization to unseen schemas. ([arXiv][1])

### Why it’s academically interesting

* **Weak supervision:** Only answers (denotations) are provided; logical forms are not. Models must search over programs consistent with the answer, e.g., with dynamic programming on denotations. ([arXiv][1])
* **Compositionality:** Questions require multi-step operations—row/column selection, filters, superlatives (argmax/argmin), aggregation (count/sum), arithmetic, and date/ordinal reasoning—rather than single-cell lookup. ([Panupong Pasupat][2])
* **Evaluation:** Accuracy is computed by *denotation exact match* against gold answer(s); generalization is measured on unseen tables. ([Stanford NLP][3])

### Practical dataset shape (HF Datasets)

Using the Hugging Face loader, each record includes:

* `question: str`
* `answers: List[str]`
* `table: { header: List[str], rows: List[List[str]], name: str }`
  Splits are provided for train/validation/test (default config `random-split-1`). ([Hugging Face][4])

---

# Minimal “Hello WTQ” in Python with **DSPy**

Below is a tiny, end-to-end program that:

1. loads WTQ from 🤗 Datasets,
2. serializes tables into a compact text block,
3. defines a DSPy module (`Predict` / CoT) to answer questions from the table, and
4. optionally uses **BootstrapFewShot** to auto-compose a few-shot prompt from a handful of labeled examples.

> This is deliberately simple to illustrate DSPy plumbing. It relies on an LLM reading the serialized table text; it is **not** a semantic parser and won’t match SOTA. (For stronger baselines, see TAPAS models fine-tuned on WTQ.) ([DSPy][5])

## 0) Install & set up

```bash
pip install dspy-ai datasets
# choose an LM backend; examples:
# - OpenAI-compatible:
#   pip install openai
# - Hugging Face Inference API client:
#   pip install huggingface_hub
```

## 1) Code (single file)

```python
import dspy
from datasets import load_dataset
import re

# ---------- Configure an LM for DSPy ----------
# Option A: OpenAI-compatible endpoint (replace with your model + key)
# import os; os.environ["OPENAI_API_KEY"] = "sk-..."
# dspy.settings.configure(lm=dspy.OpenAI(model="gpt-4o-mini"))  # or any compatible model

# Option B: Hugging Face Inference API (server-hosted)
# from huggingface_hub import InferenceClient
# dspy.settings.configure(lm=dspy.HFClient("mistralai/Mixtral-8x7B-Instruct"))  # requires HF token in env

# ---------- Load a small slice of WTQ ----------
ds = load_dataset("stanfordnlp/wikitablequestions")  # default: random-split-1
train = ds["train"]
val = ds["validation"]

def serialize_table(tbl, max_rows=12, max_cols=6):
    header = tbl["header"][:max_cols]
    rows = [r[:max_cols] for r in tbl["rows"][:max_rows]]
    lines = [" | ".join(header)]
    lines.append("-" * min(80, 3 * len(" | ".join(header))))
    for r in rows:
        lines.append(" | ".join(r))
    return "\n".join(lines)

# ---------- Define a DSPy signature & module ----------
class WTQAnswer(dspy.Signature):
    """Answer a question using only the given table."""
    question: str = dspy.InputField()
    table: str = dspy.InputField()
    rationale: str = dspy.OutputField(desc="brief chain-of-thought over the table")
    answer: str = dspy.OutputField(desc="final answer string found in/derived from the table")

class WTQProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(WTQAnswer)  # or dspy.Predict(WTQAnswer) for no rationale

    def forward(self, question: str, table: dict):
        table_txt = serialize_table(table)
        pred = self.predict(question=question, table=table_txt)
        # Post-process very lightly (strip punctuation, collapse spaces)
        ans = re.sub(r"\s+", " ", pred.answer).strip()
        return dspy.Prediction(rationale=pred.rationale, answer=ans)

prog = WTQProgram()

# ---------- Tiny training set for BootstrapFewShot (optional) ----------
def normalize(s):
    return re.sub(r"\s+", " ", s).strip().lower()

def exact_match(gold_list, pred):
    return any(normalize(g) == normalize(pred) for g in gold_list)

train_examples = []
for ex in train.select(range(12)):  # 12 supervised demos
    train_examples.append(
        dspy.Example(
            question=ex["question"],
            table=serialize_table(ex["table"]),
            answer=ex["answers"][0] if ex["answers"] else ""
        ).with_inputs("question", "table")
    )

# Use DSPy's few-shot teleprompter to compile the program with demos
from dspy.optimizers import BootstrapFewShot
teleprompter = BootstrapFewShot(metric=lambda gold, pred: exact_match([gold], pred))
compiled = teleprompter.compile(
    prog,
    trainset=train_examples,
)

# ---------- Quick evaluation on a tiny validation slice ----------
n, correct = 20, 0
subset = val.select(range(n))
for ex in subset:
    out = compiled(question=ex["question"], table=ex["table"])
    correct += int(exact_match(ex["answers"], out.answer))

print(f"Accuracy on {n} val examples: {correct/n:.2%}")

# ---------- Try an interactive example ----------
i = 0
ex = val[i]
out = compiled(question=ex["question"], table=ex["table"])
print("\nQ:", ex["question"])
print("\nTABLE:\n", serialize_table(ex["table"], max_rows=8))
print("\nMODEL:", out.answer, "\nGOLD:", ex["answers"])
print("\nRATIONALE:\n", out.rationale)
```

## 2) Notes & tips

* **Dataset fields:** The HF loader gives you the *full table content* (`header`, `rows`), so you don’t need to fetch tables separately. ([Hugging Face][4])
* **Prompt length:** Tables can be large. Start with `max_rows/cols` caps (as above). For better scaling, consider retrieval over columns/rows or learned table selectors.
* **Metrics:** WTQ evaluation is denotation-based. Even simple **exact match** with light normalization is a good first check.
* **Going beyond this toy:** For stronger results, move toward *neuro-symbolic* execution or specialized table models (e.g., TAPAS fine-tuned on WTQ) and integrate them as DSPy modules. ([Hugging Face][6])
* **DSPy docs:** `Predict` and `BootstrapFewShot` APIs are here if you want to customize signatures, add callbacks, or swap optimizers. ([DSPy][5])

[1]: https://arxiv.org/pdf/1508.00305?utm_source=chatgpt.com "arXiv:1508.00305v1 [cs.CL] 3 Aug 2015"
[2]: https://ppasupat.github.io/WikiTableQuestions/?utm_source=chatgpt.com "Compositional Semantic Parsing on Semi-Structured Tables"
[3]: https://nlp.stanford.edu/blog/wikitablequestions-a-complex-real-world-question-understanding-dataset/?utm_source=chatgpt.com "a Complex Real-World Question Understanding Dataset"
[4]: https://huggingface.co/datasets/stanfordnlp/wikitablequestions/blame/main/wikitablequestions.py "wikitablequestions.py · stanfordnlp/wikitablequestions at main"
[5]: https://dspy.ai/api/modules/Predict/?utm_source=chatgpt.com "Predict"
[6]: https://huggingface.co/google/tapas-medium-finetuned-wtq?utm_source=chatgpt.com "google/tapas-medium-finetuned-wtq"
