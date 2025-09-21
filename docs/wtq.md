# WikiTableQuestions (WTQ) — Complete Report (Latest Version)

## Overview

WikiTableQuestions (WTQ) is a benchmark for answering natural-language questions over semi-structured Wikipedia tables. The dataset spans **\~2.1k tables** and **\~22k questions**, with test tables disjoint from training tables to stress generalization. ([Stanford NLP][1])

## Version & Provenance (Latest)

* **Canonical latest release:** **v1.0.2** (Oct 4, 2016; latest GitHub release tagged Feb 16, 2017). Curated by Panupong Pasupat & Percy Liang. **License:** CC BY-SA 4.0. ([GitHub][2])
* **Hugging Face dataset:** `stanfordnlp/wikitablequestions`, mirrors WTQ with a convenient processing script and split mapping; license noted as **Creative Commons Attribution Share Alike 4.0 International**. ([Hugging Face][3])

## Task & Evaluation

* **Task:** Given a question and a table, predict the **denotation** (answer string/set) from the table. ([Stanford NLP][1])
* **Primary metric:** **Denotation Accuracy (DA)**—prediction is correct if the set of predicted values equals the gold set (order-invariant). This is the de-facto WTQ metric used in recent work. ([ACL Anthology][4])
* **Official evaluator:** `evaluator.py` (provided by WTQ maintainers). It expects a `.tagged` dataset file and a predictions file (one line per example). Note: SEMPRE’s evaluator additionally enforces type constraints, so scores can differ slightly.

  * Usage (from README): `evaluator.py <tagged_dataset_path> <prediction_path>`. ([GitHub][2])

## Splits & Sizes

**Canonical WTQ (v1.0.2):** 22,033 total examples

* `training`: **14,152**
* `pristine-unseen-tables` (**test**): **4,344** (tables unseen in training)
* `pristine-seen-tables`: **3,537** (not commonly used for official evaluation) ([GitHub][2])

**Hugging Face processed splits (derived from canonical):**

* `train`: **11,321**
* `validation`: **2,831**
* `test` (maps to `pristine-unseen-tables`): **4,344** ([Hugging Face][3])

> **Note:** The HF `train+validation` (**11,321 + 2,831 = 14,152**) together reproduce the canonical `training` portion; `test` corresponds to the official `pristine-unseen-tables`. ([Hugging Face][3])

## Data Structure

### Processed schema (Hugging Face)

Each example is a JSON with:

```json
{
  "id": "string",
  "question": "string",
  "answers": ["string", "..."],
  "table": {
    "header": ["string", "..."],
    "rows": [["string", "..."], ["..."]],
    "name": "csv/<subdir>/<file>.csv"
  }
}
```

Fields and an example instance are shown on the dataset card. ([Hugging Face][3])

### Raw WTQ files (canonical repo)

* **Questions & answers (TSV) in `data/`**

  * Fields per line: `id`, `utterance` (question), `context` (table id), `targetValue` (answer; `|`-separated if multiple).
  * Multiple split files provided (e.g., `training.tsv`, `pristine-unseen-tables.tsv`). ([GitHub][2])
* **Tables & webpages**

  * `csv/…/*.csv|*.tsv|*.html`: extracted tables (CSV/TSV plus a table-only HTML).
  * `page/…/*.html|*.json`: raw page HTML and metadata (URL, page title, chosen table index). ([GitHub][2])
* **CoreNLP “tagged” annotations**

  * **Question-level** (`tagged/data/*.tagged`): tokens, lemmas, POS, NER, normalized numeric/date spans, canonical target type (`number|date|string|mixed`).
  * **Cell-level** (`tagged/*-tagged/*.tagged`): for each table cell—row/col indices, token/lemma/POS/NER; optional **number**, **date**, **num2** (e.g., “1-2”), **list**. ([GitHub][2])
* **TSV conventions**: special escaping (e.g., `|` → `\p`, newline → `\n`) and whitespace normalization to ease parsing. ([GitHub][2])

## Example (from HF card)

A typical validation example:

```json
{
  "id": "nt-0",
  "question": "what was the last year where this team was a part of the usl a-league?",
  "answers": ["2004"],
  "table": {
    "header": ["Year", "Division", "League", "..."],
    "name": "csv/204-csv/590.csv",
    "rows": [
      ["2001", "2", "USL A-League", "..."],
      ["2002", "2", "USL A-League", "..."]
    ]
  }
}
```

(Structure and fields as documented on HF.) ([Hugging Face][3])

## Access & Loading

* **Hugging Face Datasets:** `load_dataset("stanfordnlp/wikitablequestions")` (provides `train`, `validation`, `test` with the schema above). See the dataset card for specifics and models fine-tuned on WTQ (e.g., TAPAS/TAPEX checkpoints). ([Hugging Face][3])
* **Canonical GitHub:** contains raw TSV/CSV/HTML plus evaluator and tagged artifacts; use when you need exact v1.0.2 files or the official evaluator. ([GitHub][2])

## Evaluation Protocol (Recommended)

1. Use the **`test`** split as defined above (`pristine-unseen-tables`). ([GitHub][2])
2. Report **Denotation Accuracy** with the **official `evaluator.py`** (note that SEMPRE’s evaluator may enforce type matching). ([GitHub][2])

## Best-Scoring Papers on WTQ (Last 5 Years)

> Sorted roughly by **test** performance on the standard WTQ test set; all within Sep-2020 → Sep-2025.

* **BINDER (ICLR 2023)** — *Training-free neural-symbolic with Codex*: **64.6** (test), **65.0** (dev). The paper also reports **OmniTab (2022) 63.3**, **TaCube (2022) 61.3**, **TAPEX (2021) 59.1**, **T5-3B/UnifiedSKG (2022) 50.6/51.9** for reference. *(Authors evaluate with an execution-aware variant and re-evaluate baselines consistently.)*&#x20;
* **Readi (Findings of ACL 2024)** — *LLM planning + editing for TableQA*: **61.7 (GPT-3.5)** and **61.3 (GPT-4)** denotation accuracy on WTQ, outperforming prior inference-based methods and competitive with trained models.&#x20;
* **OmniTab (EMNLP 2022)** — *Pretraining with natural & synthetic data*: **63.3** (test) as reported in BINDER’s consolidated table.&#x20;
* **TaCube (EMNLP 2022)** — *Pre-computed numerical “data cubes” for reasoning*: **61.3** (test) (paper also cites \~59.6/61+ in variants).&#x20;
* **TAPEX (NeurIPS 2021)** — *SQL-executor-style pretraining*: **59.1** (test) on WTQ (and widely used as a strong baseline in subsequent work).&#x20;

> **Metric note:** WTQ results are reported as **Denotation/Execution Accuracy** on the official test set; some papers apply lightweight normalization to align semantic correctness (e.g., yes/no mapping) but re-score all baselines with the same evaluator for fairness. When comparing across works, prefer tables that aggregate under a uniform evaluator (as above).&#x20;

## Known Caveats & Tips

* **Generalization:** Test tables are **unseen** during training; avoid table leakage in preprocessing or retrieval components. ([Stanford NLP][1])
* **Typing & normalization:** Minor formatting (dates, currencies, booleans) can affect DA; consider type-aware post-processing or the WTQ/SEMPRE evaluator if appropriate. ([GitHub][2])
* **Answer sets:** DA ignores ordering for multi-item answers—ensure your evaluator and prediction format list all items on one line per example. ([ACL Anthology][4])

## References & Resources

* **Dataset card (HF):** schema, splits, license; links to models. ([Hugging Face][3])
* **Canonical WTQ repo:** v1.0.2, file layout, “tagged” annotations, evaluator. ([GitHub][2])
* **Stanford overview post:** counts and task description. ([Stanford NLP][1])
* **Representative recent papers with strong WTQ results:**

  * **BINDER (ICLR 2023)**: consolidated dev/test table including 64.6 (test).&#x20;
  * **Readi (Findings of ACL 2024)**: 61.7/61.3 DA on WTQ.&#x20;
  * **TaCube (EMNLP 2022)**: pre-computation for numerical reasoning. ([ACL Anthology][5])

---

*This report reflects the WTQ **latest canonical release (v1.0.2)** and contemporary results as of **September 20, 2025**.* ([GitHub][2])

[1]: https://nlp.stanford.edu/blog/wikitablequestions-a-complex-real-world-question-understanding-dataset/?utm_source=chatgpt.com "a Complex Real-World Question Understanding Dataset"
[2]: https://github.com/ppasupat/WikiTableQuestions "GitHub - ppasupat/WikiTableQuestions: A dataset of complex questions on semi-structured Wikipedia tables"
[3]: https://huggingface.co/datasets/stanfordnlp/wikitablequestions "stanfordnlp/wikitablequestions · Datasets at Hugging Face"
[4]: https://aclanthology.org/2025.findings-acl.121.pdf?utm_source=chatgpt.com "Structural Deep Encoding for Table Question Answering"
[5]: https://aclanthology.org/2022.emnlp-main.145/?utm_source=chatgpt.com "TaCube: Pre-computing Data Cubes for Answering ..."
