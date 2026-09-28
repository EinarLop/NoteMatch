# NoteMatch — Design

## 1. Overview

NoteMatch recommends notes PDFs from a catalog based on a student's free-text description of what they're studying. It returns up to 3 products, each explained by the catalog data that matched.

The system has two independent pipelines that share one index:

- **Ingestion (offline, run once per catalog change):** turns PDFs + metadata into a searchable index. Slow and expensive steps (LLM summaries, embeddings) live here.
- **Query (online, per request):** turns a user prompt into ranked, explained recommendations. Must stay under ~10 s on a local model, so it makes as few LLM calls as possible (target: exactly one).

**Out of scope:** user accounts, purchases, full-text search of paid content, catalog editing UI, PDFs without an embedded outline (see 4.2).

## 2. Architecture

```
INGESTION (offline)                                 QUERY (online)

 pdfs/*.pdf    metadata.csv                          user prompt
      │             │                                     │
      ▼             │                                     ▼
 ┌──────────────┐   │                            ┌──────────────────┐
 │ PDF extractor│   │                            │ Query parser     │  LLM call #1
 │ outline +    │   │                            │ prompt → JSON    │  (schema-validated,
 │ page-1 summary│  │                            └────────┬─────────┘   retry on fail)
 └──────┬───────┘   │                                     │ {topics[]}
        ▼           ▼                                     ▼
 ┌──────────────────────┐                        ┌──────────────────┐
 │ Summarizer (LLM)     │                        │ Retriever        │  one search
 │ one summary/product  │                        │ per topic        │  per topic
 └──────┬───────────────┘                        └────────┬─────────┘
        ▼                                                 │ (topic, chunk, score) hits
 ┌──────────────────────┐    ┌──────────────┐             ▼
 │ Embedder             │───▶│ Vector index │◀───  ┌──────────────────┐
 │ TOC entries + summary│    │ + product    │      │ Ranker           │  coverage × strength,
 └──────────────────────┘    │   store      │      │ threshold, top 3 │  threshold
                             └──────────────┘      └────────┬─────────┘
                                                            ▼
                                                   ┌──────────────────┐
                                                   │ Explainer        │  template, no LLM
                                                   │ covered/uncovered│  → no invented claims
                                                   └────────┬─────────┘
                                                            ▼
                                                   CLI output / POST /recommend JSON
```

**Key decisions**

| Decision | Choice | Why |
|---|---|---|
| Index unit | One vector per TOC entry, plus one per summary | Lets a match point to the exact TOC section (FR5). |
| Content protection | Extractor reads the PDF outline (metadata) and the text of page 1 only | Enforced at the source, so paid content (pages 2+) is never even opened (FR7). |
| Explanations | Template filled from retrieval results | Grounded by construction (FR5), and saves an LLM call (latency NFR). |
| LLM calls per query | 1 (query parsing) | Keeps local-model latency under 10 s. |

## 3. Data contracts

All contracts are Pydantic models in `src/notematch/schemas.py`. They double as validation (FR2) and as the API schema (FastAPI generates OpenAPI docs from them).

### 3.1 Catalog input — `catalog/metadata.csv`

One row per product. The PDF lives at `catalog/pdfs/<product_id>.pdf`.

```
product_id,title,subject,price,purchase_url
adversarial_search,Adversarial Search & Game Playing,ai,7.99,https://…
```

No `level` field: the notes aren't targeted at specific levels, so it would carry no signal. (This departs from FR1/FR2 in `requirements.md`.) `subject` is kept as catalog metadata and given to the summarizer, but it isn't used for filtering (5.2).

```python
class Product(BaseModel):
    product_id: str
    title: str
    subject: str            # lowercase, e.g. "ai"
    price: float
    purchase_url: HttpUrl
```

### 3.2 Ingestion output

```python
class TocEntry(BaseModel):
    text: str               # "Alpha-Beta Pruning"
    depth: int              # 1 = section, 2 = subsection, … (as returned by get_toc)
    page: int | None

class ProductRecord(Product):      # stored in products.json
    summary: str                    # LLM-generated at ingestion
    toc: list[TocEntry]

class IndexEntry(BaseModel):       # one row per vector
    entry_id: str                   # "adversarial_search:toc:3" / "adversarial_search:summary"
    product_id: str
    kind: Literal["toc", "summary"]
    text: str                       # what was embedded; shown in explanations
```

Vectors are stored separately as a numpy matrix, with row `i` corresponding to `entries[i]`.

### 3.3 Query parser output (LLM → JSON)

```python
class ParsedQuery(BaseModel):
    topics: list[str] = Field(max_length=8)
```

An empty `topics` list is valid and means "no study topics found". The request short-circuits to a no-match (FR6). Don't use `min_length=1` here: structured output *forces* the model to follow the schema, so a required item makes it invent a topic (a smoke test turned "hi, what's up?" into `["greeting"]`). The prompt tells the model to return `[]` when there's nothing academic.

There's no `subject` field. The catalog is single-subject, so a subject filter would do nothing (5.2), and every extra field costs output tokens.

### 3.4 API — `POST /recommend`

```python
class RecommendRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=1000)

class SectionMatch(BaseModel):
    text: str               # TOC entry or "summary"
    similarity: float

class TopicMatch(BaseModel):
    topic: str
    sections: list[SectionMatch]    # best matches in this product, max 3

class Recommendation(BaseModel):
    product: Product        # catalog fields only; summary and TOC stay server-side
    score: float
    covered_topics: list[TopicMatch]

class RecommendResponse(BaseModel):
    parsed_query: ParsedQuery | None   # returned for transparency/debugging
    recommendations: list[Recommendation]   # 0–3
    uncovered_topics: list[str]
    message: str            # e.g. "No product in the catalog covers these topics."
```

No match means `recommendations == []` plus an explanatory `message`. It's still a 200 response, not an error.

## 4. Ingestion pipeline

Command: `notematch ingest catalog/ --out index/`. It rebuilds the whole index on every run; at catalog scale that takes minutes, so there's no incremental update.

### 4.1 Load metadata
Parse `metadata.csv` into `Product` models. Fail loudly on an invalid row, a missing PDF or a duplicate `product_id`. A silently skipped product only shows up much later as a mysteriously bad recommendation.

### 4.2 TOC extraction
Library: **PyMuPDF** (`import pymupdf`). `doc.get_toc()` returns `[depth, title, page]` from the PDF's embedded outline (bookmarks). It reads **metadata, not page text**, so it doesn't touch the paid content.

The spike found an outline in 20/20 catalog PDFs (8–14 entries, depth ≤ 2). A PDF **without** an outline stops ingestion with a clear error naming the file. An LLM fallback (extracting a TOC from page text) is deferred until a real catalog needs it (9, #1).

Generic entries are dropped because they match everything weakly and add noise to ranking. This is a case-insensitive exact match against `toc_stoplist`, which includes the sections every note shares: `Exercises`, `Summary of Key Concepts`.

### 4.3 Page-1 summary
Page 1 of every note holds the title and an author-written **"Summary"** section. The extractor takes the text of **page 1 only** and keeps the lines between the `Summary` and `Contents` headings (the printed TOC follows the summary on page 1, and the outline already covers it). Line-break hyphenation (`en-\ncoding`) is joined back. A missing heading is an ingestion error, like a missing outline. All 20 catalog PDFs pass this check (700–1,350 characters of summary each). This text is used **only as input to the LLM summarizer** and is never embedded or stored.

**FR7 guarantee:** the extractor calls `get_toc()` and `doc[0].get_text()`, and nothing else. Pages 2+ are never read. This matters because the notes are only ~5 pages long: any "first N pages" rule would have read the whole document.

### 4.4 Summary (LLM)
One call per product, with temperature 0:

```
You are describing a study-notes product for a catalog.
Using ONLY the information below, write 2–4 sentences on what
the notes cover. Do not mention anything not present below.

Title: {title}
Subject: {subject}
Table of contents:
{toc_as_indented_text}
Author's summary:
{page1_summary}
```

### 4.5 Embedding
- Each TOC entry is embedded with **context**: `"{title} › {parent_section} › {entry}"`, e.g. `"Adversarial Search & Game Playing › Alpha-Beta Pruning"`. A bare "Alpha-Beta Pruning" gives the embedder less to go on, and a subsection like "Properties" is meaningless alone. `IndexEntry.text` still stores the bare entry, which is used for display.
- Vectors are L2-normalized, so a dot product equals cosine similarity.
- Some embedding models need **task prefixes** (e.g. nomic: `search_document:` / `search_query:`, e5: `passage:` / `query:`). These are set per model in the config. Forgetting them silently degrades retrieval.

### 4.6 Output — `index/`
```
products.json     list[ProductRecord]
entries.json      list[IndexEntry]     (row i ↔ vectors[i])
vectors.npy       float32 [n_entries, dim]
manifest.json     {embedding_model, dim, built_at, n_products}
```
Current catalog: 20 products → about 220 TOC entries + 20 summaries.

At startup the query side checks that `manifest.embedding_model` matches the config and refuses to run if it doesn't. Vectors from different models aren't comparable, and a mismatch would produce plausible-looking nonsense rather than an error.

## 5. Query pipeline

`recommend(prompt) -> RecommendResponse` is a single function. The CLI and FastAPI are thin wrappers around it.

### 5.1 Parse (the one LLM call)
- **Structured output:** the `ParsedQuery` JSON schema is passed to the model (Ollama `format=`, API "response format"/tool schema). Temperature 0.
- **Prompt:** instructions plus 2–3 few-shot examples showing how to split compound requests and when to return nothing:

  ```
  Extract the specific study topics the student needs help with.
  Topics are short noun phrases. Split combined requests into separate topics.
  If the message contains no study topic, return an empty list.

  Student: "struggling with minimax and alpha-beta for my game AI assignment"
  → {"topics": ["minimax", "alpha-beta pruning"]}
  Student: "hi, what's up?"
  → {"topics": []}
  ...
  Student: "{prompt}"
  ```
- **Validate and retry:** Pydantic validates the output. If validation fails, the call is retried up to `max_retries` (default 2), with the validation error appended to the prompt. If every attempt fails, the request ends as a no-match with the message "Couldn't identify any topics in your request."
- **Normalize:** topics are lowercased, stripped and de-duplicated. If the list is empty, return a no-match right away.

### 5.2 Retrieve (per topic, FR3)
1. Embed all `T` topics in one batch (with the query prefix) → `Q: [T, dim]`.
2. `sims = vectors @ Q.T` → `[n_entries, T]`. This is an exact, brute-force search over every entry.
3. For each (product, topic), take the **max** similarity over that product's entries → `S: [n_products, T]`. Keep the top 3 entries per cell for the explanation.

There's **no subject filter**. Every product is `ai`, so it would never exclude anything. Prompts outside the catalog ("organic chemistry") are rejected by the thresholds instead, and the eval's no-match cases check that they are.

The max is used rather than the mean because one strong matching section is what counts. A product shouldn't be penalized for its unrelated sections.

### 5.3 Rank (FR4)
```
covered[p,t] = S[p,t] >= τ_topic
score[p]     = mean_t( S[p,t] · covered[p,t] )
```
- Uncovered topics contribute 0, so the score rewards **how many** topics a product covers (coverage) as well as **how strongly** it covers them (similarity).
- A product is eligible if `score[p] >= τ_min`. The result is the top 3 eligible products by score.

Worked example (illustrative numbers), prompt about "backpropagation and convolution layers", `τ_topic = 0.55`:

| Product | backpropagation | convolution layers | score |
|---|---|---|---|
| convolutional_networks | 0.62 ✓ | 0.84 ✓ | 0.73 |
| neural_networks | 0.81 ✓ | 0.41 ✗ | 0.41 |
| reinforcement_learning | 0.30 ✗ | 0.20 ✗ | 0.00 |

**Thresholds depend on the embedding model.** Cosine distributions differ a lot between models (some put everything in 0.6–0.9). `τ_topic` and `τ_min` are therefore set per model in the config and tuned on the eval set (section 7), never chosen by feel.

### 5.4 Explain (FR5, FR6)
The explainer is pure templating over `S`, `covered` and the stored top entries, with no LLM involved:
- `covered_topics` is each covered topic with its top matching sections and their similarity scores.
- `uncovered_topics` is every topic not covered by **any returned** product.
- `message` comes from a fixed set of templates:
  - normal result: "Found N products covering K of T topics."
  - partial coverage: "... No product covers: X, Y."
  - no match: "No product in the catalog covers these topics."

### 5.5 Latency and logging
Budget on the M1 (8 GB) with `qwen3:4b-instruct`:

| Stage | Expected |
|---|---|
| Parse (LLM, ~20–50 output tokens) | ~2 s warm (measured); ~11 s on a cold start while the model loads |
| Embed topics | < 100 ms |
| Retrieve + rank + explain | < 10 ms |

The LLM call accounts for almost all of the latency. The server keeps the model warm by sending one dummy parse at startup. Each request writes one JSON log line: `{request_id, latency_ms: {parse, embed, rank, total}, prompt_tokens, completion_tokens, retries, n_results}`. Token counts come from the LLM response (Ollama: `prompt_eval_count` / `eval_count`).

## 6. Configuration

### 6.1 Format and loading
- **TOML**, read with the stdlib `tomllib` and validated into a Pydantic `Config` model with `extra="forbid"`, so a misspelled key is an error rather than a silently ignored setting.
- **One file per setup.** The file is chosen with `--config` or `NOTEMATCH_CONFIG` and defaults to `config.toml`. The comparison experiment (section 7) is simply two config files, with no profile system.
- **Secrets never go in the file.** `ANTHROPIC_API_KEY` comes from the environment.

### 6.2 Example — `config.toml`
```toml
[llm]
provider = "ollama"               # "ollama" | "anthropic"
model = "qwen3:4b-instruct"       # non-thinking variant; plain "qwen3:4b" is thinking-only (8 GB RAM M1)
think = false                     # safety net for reasoning models: thinking adds latency and can loop
# Ollama address comes from the OLLAMA_HOST env var (default localhost), see 8.4
temperature = 0.0
max_retries = 2

[embedding]                       # always via Ollama (see 6.3)
model = "nomic-embed-text"
query_prefix = "search_query: "
document_prefix = "search_document: "
# Thresholds are specific to this embedding model; re-tune whenever the model changes.
tau_topic = 0.55
tau_min = 0.30

[retrieval]
mode = "per_topic"                # "per_topic" | "whole_prompt" (section 7.4)
max_results = 3

[ingestion]
toc_stoplist = ["introduction", "preface", "index", "references", "bibliography",
                "contents", "exercises", "summary of key concepts"]

[paths]
catalog = "catalog"
index = "index"
```

To switch to an API model, change only the `[llm]` block:
```toml
[llm]
provider = "anthropic"
model = "claude-haiku-4-5-20251001"
temperature = 0.0
max_retries = 2
```

### 6.3 Provider switch (FR9)
`llm.py` exposes one function:

```python
def complete_json(prompt: str, schema: type[BaseModel], cfg: LLMConfig) -> tuple[BaseModel, Usage]
```

Inside it is a plain `if cfg.provider == ...` with one branch per provider (Ollama: `format=<json schema>`; Anthropic: the schema as a forced tool call). There's no provider class hierarchy for two backends. Retry and validation live in the caller, so they're shared by both.

**Embeddings always run through Ollama**, even when the LLM is an API model:
- This avoids `sentence-transformers`/`torch` (several GB), which matters for the 20 GB budget and the Docker image size.
- Local embedding is fast and free, so there's no reason to pay per call.
- Whichever model built the index has to embed the queries too, so the embedding model is always local.

The manifest check from 4.6 compares `embedding.model` against the index at startup.

## 7. Evaluation

Command: `notematch eval eval/cases.jsonl --config config.toml`. It prints the metrics and every failing case, and writes `eval/results/<config-name>.json`.

### 7.1 Eval set — `eval/cases.jsonl`
One case per line:
```json
{"id": "c01", "prompt": "my game bot is too slow, how do I skip branches in minimax?", "expected": ["adversarial_search"], "split": "dev"}
{"id": "c02", "prompt": "attention and transformers for text classification", "expected": ["sequence_models", "natural_language_processing"], "split": "dev"}
{"id": "c03", "prompt": "need help with organic chemistry reaction mechanisms", "expected": [], "split": "test"}
```
- `expected` lists **every** acceptable product (any one counts as a hit). An empty list marks a **no-match case**.
- **Overlapping products are the interesting cases.** For example `sequence_models` / `natural_language_processing`, `neural_networks` / `convolutional_networks`, `markov_decision_processes` / `reinforcement_learning`, and the search notes. Those cases should list every product that genuinely fits.
- **Mix** (for about 30 cases): about 20 match cases (single-topic, multi-topic and casual or misspelled wording) and about 6 no-match cases. The catalog is AI-only, so the no-match cases should include other subjects, **nearby-but-absent topics** (e.g. "SQL query optimization", "quantum computing"), and off-topic chatter. Add a few partial-coverage cases too.
- **Write the prompts like a student would, not from the TOCs.** Copying TOC wording into prompts leaks the answer into the test, the same way test data can leak into training, and inflates the scores.
- **Write the cases before tuning anything,** and don't edit them to make a failing case pass.

### 7.2 Metrics
Computed over the **match cases** (`expected` non-empty):
- **Hit@3** is the fraction of cases with at least one expected product in the results.
- **MRR** is the mean of `1 / rank` of the first expected product, or 0 if none is returned.

Computed over the **no-match cases**:
- **No-match accuracy** is the fraction of cases that return zero recommendations.

Operational metrics, from the request logs:
- p50/p95 latency and mean tokens per request.

With about 30 cases, **one case moves a metric by 3–5 points**. Differences smaller than that are noise, and the README should say so.

### 7.3 Threshold tuning
- `τ_topic` and `τ_min` are tuned by **grid search on the `dev` split** (about 20 cases). The objective is `Hit@3 + no-match accuracy`, which trades recall against false positives.
- The chosen thresholds are then scored **once on the `test` split** (about 10 cases). Those are the numbers that get reported.
- **Tuning is cheap:** the parsed query and the `S` matrix don't depend on the thresholds, so each case is parsed and retrieved once and cached. The grid then only re-runs ranking, with no LLM calls.

### 7.4 Experiment: per-topic vs whole-prompt retrieval
This tests the project's central design decision (FR3):
- **A — per-topic (default):** the approach described in section 5.
- **B — whole-prompt:** skip topic splitting and retrieve with `topics = [prompt]`, so `T = 1`. Everything else is unchanged. This is a one-line switch (`retrieval.mode = "whole_prompt"`).

Each variant gets its own tuned thresholds on `dev`, which keeps the comparison fair, and is reported on `test`:

| Setup | Hit@3 | MRR | No-match acc. | p50 latency |
|---|---|---|---|---|
| A: per-topic | | | | |
| B: whole-prompt | | | | |

The expectation is that per-topic wins on multi-topic prompts, since a whole-prompt vector averages the topics together. If it doesn't, that's a legitimate finding and gets reported as-is.

## 8. Project layout and tech stack

### 8.1 Layout
```
NoteMatch/
├── src/notematch/
│   ├── schemas.py      # all Pydantic models (section 3)
│   ├── config.py       # Config model + load()
│   ├── llm.py          # complete_json(), embed() — the only code that talks to models
│   ├── ingest.py       # section 4
│   ├── recommend.py    # section 5: parse, retrieve, rank, explain
│   ├── evaluate.py     # section 7: metrics, threshold grid search
│   ├── cli.py          # notematch ingest | recommend | eval | serve
│   └── api.py          # FastAPI app, POST /recommend
├── tests/
│   ├── test_parse.py
│   ├── test_rank.py
│   └── test_no_match.py
├── catalog/            # metadata.csv + pdfs/ (+ latex/ sources)
├── eval/cases.jsonl
├── index/              # generated by ingest, git-ignored
├── config.toml
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── README.md
└── design.md
```
The layout is one package (the `src/` layout that `uv init --package` creates) with one module per pipeline stage group. Split a module only if it outgrows a single screen of concepts.

**Paid PDFs are never committed.** The current catalog is self-authored, so it can ship with the repo as the demo catalog. A real paid catalog would be git-ignored.

### 8.2 Stack
| Need | Choice | Note |
|---|---|---|
| Python | 3.12, managed with `uv` | `tomllib` is in the stdlib |
| PDF | `pymupdf` | fast, has `get_toc()`; AGPL (fine for an open portfolio repo; `pypdf` is the permissive fallback) |
| Models | `ollama` client, `anthropic` SDK | Ollama serves both the LLM and embeddings |
| Validation | `pydantic` | schemas, config, API |
| Vectors | `numpy` | no vector DB (section 3.2) |
| API | `fastapi` + `uvicorn` | |
| CLI | `argparse` (stdlib) | four subcommands don't need a CLI framework |
| Logging | stdlib `logging`, one JSON line per request | |
| Tests | `pytest` | |

### 8.3 Unit tests (NFR)
No test calls a real model, so tests are fast and deterministic.
- **Seam:** `parse_query(prompt, complete=llm.complete_json)`. Tests pass a fake `complete` function in place of the real LLM call.
- **`test_parse.py`:** valid output is accepted, topics are normalized; invalid-then-valid succeeds after one retry; always-invalid ends in no-match after `max_retries`; empty topics ends in no-match without retrying.
- **`test_rank.py`:** the worked example from 5.3, as a hand-built `S` matrix, produces the expected order and scores; `max_results` caps the output.
- **`test_no_match.py`:** a matrix with every value below `τ_topic` gives an empty result plus the no-match message; a parse failure gives the same.

### 8.4 Docker — one command
**On an M1 Mac, Ollama inside Docker is CPU-only.** Docker on macOS can't use the Metal GPU, so the local model would miss the 10 s target by a wide margin. The setup is therefore:
- **Ollama runs natively on the host** (install it once).
- **`docker compose up`** runs the app container. It reaches Ollama at `OLLAMA_HOST=http://host.docker.internal:11434`, which the `ollama` client reads automatically.
- The container entrypoint does this, in order:
  1. pull the configured models if they're missing (`ollama.pull`);
  2. run `ingest` if `index/` is empty;
  3. start `serve`.
- `catalog/` and `index/` are mounted as volumes, so the index survives restarts.

**Storage estimate:** `qwen3:4b-instruct` is about 2.5 GB, `nomic-embed-text` about 0.3 GB, and the image (no torch) about 0.5 GB. Total is around 3.5 GB of the 20 GB budget.

## 9. Open questions

Each open question has a default, so none of them blocks work. The last column says how it gets settled, or how it was settled.

| # | Question | Status / default | Resolved by |
|---|---|---|---|
| 1 | Do the PDFs have embedded outlines? | ✅ 20/20 do. LLM fallback **deferred**; a missing outline is an ingestion error | Spike. Revisit when a real catalog is added |
| 2 | Are any PDFs scanned (no text layer)? | ✅ None | Spike |
| 3 | What language are the notes? | ✅ English → `nomic-embed-text` | Spike. Prompts assumed English too |
| 4 | Does the local model parse within ~10 s on the M1? | ✅ `qwen3:4b-instruct`: ~2 s warm, ~11 s cold (8 GB RAM rules out 8B) | Smoke test. Parse *quality* is still to be checked on the eval set |
| 5 | Is a subject filter worth it? | ✅ Removed: single-subject catalog | Spike |
| 6 | Should the top 3 be diverse? (e.g. prefer a product covering an uncovered topic over a third one covering the same topics) | No, pure score order | Only if eval shows multi-topic prompts leaving topics uncovered that the catalog does cover |
| 7 | Should the page-1 summary text be embedded too? | No, summarizer input only (4.3) | Only if eval shows recall misses it would catch |
| 8 | `uncovered_topics` = not covered by any *returned* product (5.4) | As stated | Confirm with product owner |
| 9 | `requirements.md` still lists `level` (FR1, FR2) | Design drops it (3.1) | Update `requirements.md` |
| 10 | Catalog size | ~240 index entries → brute-force numpy | Only above ~100k entries: switch to FAISS/Chroma |
| 11 | Few-shot copying: the 4B parser returned an example's topics for "hey what's up" (c29) | Known bug, accepted for now | Send few-shot examples as separate chat turns (needs a `messages` option in `complete_json`) |
| 12 | Threshold placeholder `tau_topic = 0.55` is too low: no-match topics reach 0.68, match topics 0.52–0.83, overlapping in 0.59–0.68 (step-4 check, dev only) | ✅ Tuned on dev: `tau_topic = 0.70`, `tau_min = 0.06` (plateau 0.68–0.71 / 0–0.14). Remaining dev trade-off: c20 ("kmeans", best 0.65) is lost | Grid search (7.3) |
| 13 | Summary rows act as "hubs": long, generic, same opening, so off-topic queries (quantum, SQL, Grover) match them at 0.62–0.68 | Keep summaries in the index | Experiment: TOC-only vs TOC + summary, or strip the "This study note covers" boilerplate before embedding |
| 14 | Topics mentioned only inside the notes, with no TOC heading (cross-entropy, squared error), match weakly (≤ 0.52) | Accept | Only if eval shows these misses matter |
| 15 | `tau_min` is compared to a mean over topics, so specialist products fade as prompts get more topics: c11 (3 topics) drops `reinforcement_learning` (0.73 on "q-learning" → score 0.24 < 0.30) although it's the only product covering that topic | ✅ Tuning pushed `tau_min` to ~0, so eligibility ≈ "covers at least one topic"; c07 and c11 now pass | Grid search (7.3) |
