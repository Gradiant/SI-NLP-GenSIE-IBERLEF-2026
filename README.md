# GRADIANT - GenSIE 2026

##  How to run for eval

The [task instructions](https://github.com/gia-uh/gensie/blob/main/docs/submission.md) have been followed to run the evaluation.

The commands used are:
1. Ensure the .env file is properly configured:
```bash
OPENAI_API_KEY="your_api_key" # if using OpenAI models
OPENAI_BASE_URL="http://localhost:1234/v1" # Optional: for local LLMs (Ollama, etc.)
```
You can copy the `.env.example` file and fill in the required values.
```
cp .env.example .env
```

2. Start the server:
```bash
docker compose up agent
```
3. In other terminal, install and run the evaluation (with baseline, stable and experimental):
```bash
uv sync

uv run gensie eval --data data/starter/ --url http://localhost:8000 --pipeline baseline --model gpt-4o-mini
```

4. Check leaderboard:
```bash
uv run gensie leaderboard results/
```




---

# GRADIANT Agents

This section describes the extraction agents developed by the GRADIANT team for the GenSIE 2026 competition. Both agents live under `src/gensie/agents/` and inherit from `GenSIEAgent`. They are served as the `stable` and `experimental` pipelines respectively.

## Repository layout

```
src/gensie/agents/
├── utils.py                     — JSON Schema normalisation helpers (strict mode)
├── stable_agent.py              — Production agent (StableAgent)
├── experimental_agent.py        — Experimental agent (ExperimentalAgent)
└── experimental/
    ├── grad_llm.py              — LLM wrapper with specialised call methods
    ├── categorizer.py           — Schema field type analysis & TypeScript-like rendering
    ├── corector.py              — Post-extraction fuzzy-match corrector
    ├── planner.py               — Builds a per-field execution plan
    ├── prompts.yaml             — All system/user prompts
    └── strategies/
        ├── Strategy.py          — Abstract base class (StrategyV2)
        ├── Direct.py            — Simple scalar fields (string, int, bool)
        ├── Categorical.py        — Enum fields defined via $ref
        ├── FixedEntities.py     — Arrays of $ref-typed objects
        ├── SoftEntities.py      — Arrays of plain strings
        └── Complex.py           — Nested object arrays (hardest case)
```

---

## StableAgent (`stable_agent.py`)

Every task must be answered within the 60-second wall-clock budget set by the competition rules.

### Pipeline

**Step 0 — Text pre-processing**
Before any extraction the agent runs two preparatory LLM calls on the raw input text:
1. An *analysis* call identifies which entity types are present and where they appear.
2. A *tagging* call copies the original text verbatim and wraps the relevant fragments in `**bold**`.

The enriched text (original + tagged copy + analysis) replaces `task.input_text` for all downstream extraction calls, giving the extraction model a clearer signal without modifying the source content.

**Path 1 — Single-field verbatim extraction**
Triggered when the target schema has exactly one string field whose description contains `"verbatim"` or `"fragment"`. These fields are scored with exact match by the evaluator, so returning a paraphrase scores 0.

Strategy: `_quote_then_extract` with `verbatim=True`.
- Step 1: the model locates and copies the exact sentence(s) from the text that contain the answer.
- Step 2: skipped — the grounded span *is* the answer.

If grounding returns nothing a standard single-call extraction is used as a fallback.

**Path 2 — Full-schema extraction**
Used for all other schemas (multi-field, arrays, nested objects). A single LLM call is made with the complete schema and a conservative anti-hallucination system prompt in Spanish.

Decomposing the schema into per-field calls was ruled out after empirical testing: it caused over-extraction in array fields and loss of cross-field context in string fields.

**Post-processing**
- If more than 17 seconds remain, a `_verification_pass` is run: the model reviews its own output against the source text and corrects any ungrounded values.
- Enum normalisation: case-insensitive lookup fixes `"positive"` → `"POSITIVE"` with no extra token cost.
- Array sorting: array items are sorted by their enum-typed field following the enum definition order in the schema, so positional keys align with the gold annotations.

### Unused methods
- `_self_consistency`: majority vote over N runs for boolean/enum fields.
- `_null_review`: targeted retry for fields that were left null on the first pass.

---

## ExperimentalAgent (`experimental_agent.py`)

A planner-based agent that classifies each schema field by its JSON type and routes it to the most appropriate extraction strategy.

### Pipeline

**1. Planning — `planner.py`**
For each field in the schema, `categorizer.py` inspects its type (following `$ref` and `anyOf` pointers) and assigns one of five categories. The planner then instantiates the matching strategy, calls `estimate()` to compute the expected token/time cost, and sorts the plan so required fields are processed first.

**2. Field type classification — `categorizer.py`**

| Category | Field type |
|---|---|
| `direct` | `string`, `int`, `bool`, `number` scalars |
| `categorical` | scalar with a `$ref` enum constraint |
| `fixed_entities` | array whose items resolve to a `$ref` object |
| `soft_entities` | array of plain strings |
| `complex` | array of nested objects |

The same module also renders schema fragments as TypeScript-like type strings (used in LLM prompts) and as plain-text descriptions.

**3. Strategies — `strategies/`**

- **Direct / Categorical**: single LLM call with the field schema rendered as a TypeScript type. Falls back to `direct_call` if the response cannot be parsed.
- **FixedEntities / SoftEntities**: same as Direct with a slightly higher temperature.
- **Complex**: a two-step approach — first a *candidates* call extracts a free-text list of likely values from the source text; then a second call uses those hints as additional context for the structured extraction.

**4. Post-extraction correction — `corector.py`**
Every strategy passes its output through `correct_in_text`, which applies three corrections in order:
1. **Type coercion**: unwraps malformed dicts like `{"value": 3}` to the expected scalar type.
2. **Enum normalisation**: case-insensitive canonicalisation against the schema's allowed values.
3. **Fuzzy span correction**: for short string values not found verbatim in the input text, a sliding-window search scores every candidate span using a weighted combination of fuzzy ratio (rapidfuzz/difflib) and character n-gram cosine similarity. If the best match scores ≥ 0.90, the span from the original text is used instead.

---

## Key differences at a glance

| | `StableAgent` | `ExperimentalAgent` |
|---|---|---|
| Architecture | Two fixed paths + text pre-processing | Planner + 5 type-specific strategies |
| Text pre-processing | Bold-tagging of relevant fragments | None |
| Per-field decomposition | Only for single verbatim fields | Always, grouped by type category |
| Post-extraction correction | Enum fix + array sort + verification pass | `correct_in_text` (fuzzy matching) |


---

# GenSIE 2026 Public Starter Kit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](./Dockerfile)

**GenSIE (General-purpose Schema-guided Information Extraction)** is a shared task at [IberLEF 2026](https://sites.google.com/view/iberlef-2026). This repository provides the official starter kit for participants.

## 🚀 Quick Start

### 1. Installation
We recommend using [**uv**](https://github.com/astral-sh/uv) for fast dependency management:

```bash
git clone <repository-url>
cd gensie
uv sync --group dev
```

### 2. Configuration
Create a `.env` file to configure your inference backend:

```bash
OPENAI_API_KEY="your-api-key"
OPENAI_BASE_URL="http://localhost:1234/v1" # Optional: for local LLMs
```

### 3. Serving your Agent
Start the FastAPI server:
```bash
uv run gensie serve --port 8000
```

### 4. Running Benchmarks
Evaluate your agent against the 40 starter instances:
```bash
uv run gensie eval --data data/starter/ --url http://localhost:8000 --pipeline baseline --model gpt-4o-mini
```

## 🛠️ How to Participate

1.  **Inherit from `GenSIEAgent`**: Implement your extraction logic in `src/gensie/`.
2.  **Register your Pipelines**: Configure up to 3 pipelines in `OfficialParticipant` (see `src/gensie/baseline.py`).
3.  **Submit**: Open a [**Competition Submission Issue**](https://github.com/gia-uh/gensie/issues/new?template=submission.md) to register your team and repository.
4.  **Dockerize**: Use the provided `Dockerfile` and `docker-compose.yml` for testing and final submission.

```bash
docker compose up --build
```

## 📊 Dataset & Metrics

The kit includes **40 silver-generated instances** for initial testing. Official metrics use **Flattened Schema Scoring** (Micro-F1), which combines exact matches for rigid fields and semantic similarity for free-text fields.

## 📜 Documentation

For more details, see our guides:
*   🚀 [**Starter Kit Guide**](./docs/starter-kit.md)
*   📂 [**Submission Guidelines**](./docs/submission.md)
*   📊 [**Task Description**](./docs/description.md)

## ⚖️ License

This starter kit is licensed under the **MIT License**.
