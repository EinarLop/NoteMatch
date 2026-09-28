
# NoteMatch

## Goal
Given a user's free-text prompt (for example, "I'm taking Calculus II and struggling with integration techniques and series"), recommend the most relevant notes PDFs from the catalog and explain why.

## Functional requirements

**FR1. Catalog ingestion**
- Read a folder of notes PDFs plus a metadata file (JSON or CSV) with title, subject, level, price and purchase link.
- For each PDF, extract only the table of contents and a few sample pages, never the full content.
- Generate a short summary of each product with the LLM once, at ingestion time.
- Embed the TOC entries and summary and store them in a vector index, linked to the product ID.

**FR2. Query understanding**
- The LLM converts the user's prompt into structured JSON: subject, level (if mentioned) and a list of topics.
- The output is validated against a schema, with a retry if it's invalid.

**FR3. Per-topic retrieval**
- Each extracted topic is searched separately against the catalog index, returning the top matches with similarity scores.

**FR4. Ranking**
- Products are scored by how many of the user's topics they cover and how strongly.
- The system returns up to 3 products above a minimum score threshold.

**FR5. Explained recommendations**
- For each recommendation, the response lists which of the user's topics it covers and which TOC sections match.
- It also lists any topics no product covers.
- Explanations may only use catalog data, never invented claims.

**FR6. No-match handling**
- If no product passes the threshold, the system says so clearly instead of forcing a recommendation.

**FR7. Content protection**
- Only metadata, TOCs, summaries and sample pages enter the index. Full paid content is never retrievable.

**FR8. Interface**
- A CLI for development, plus a FastAPI endpoint (`POST /recommend`) that returns the recommendations as JSON.

**FR9. Model configuration**
- The LLM and embedding models are set in a config file, so the same pipeline can run on a local model or an API model.

## Non-functional requirements

- Runs on your M1 within the 20 GB storage budget.
- Each recommendation takes under about 10 seconds with a local model.
- Each request logs latency and token counts.
- Unit tests cover parsing, ranking and no-match behavior.
- The project runs through Docker with one command.

## Evaluation

- A test set of 20–30 prompts, each with the products you'd expect to be recommended, including a few prompts that should return no match.
- Metrics: hit rate at 3 (is a correct product in the top 3?), MRR (how high the correct product ranks), and accuracy on the no-match cases.
- One experiment comparing two setups, for example two embedding models or per-topic vs. whole-prompt retrieval, with the results in a README table.

