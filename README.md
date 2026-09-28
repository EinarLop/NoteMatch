# NoteMatch

NoteMatch recommends study notes based on what a student says they're struggling with.

You type something like "I need help with Q-learning, but I think I have to understand MDPs first", and it suggests up to three notes from the catalog that cover those topics. For each one it tells you which sections match what you asked for. If nothing in the catalog fits, it says so instead of recommending something unrelated.

## How it works

The project is a small RAG (retrieval-augmented generation) system.

When the catalog is loaded, each set of notes is read only as far as its table of contents and summary page. Those pieces are turned into embeddings and stored in a vector index. The rest of the notes is never read, since it's the paid content.

When a student asks for something, an LLM reads the request and pulls out the individual topics. For the example above, that's "Q-learning" and "Markov decision processes". Each topic is then searched separately against the index. Notes are ranked by how many of the topics they cover and how closely they match.

The explanation is built directly from the matching table-of-contents entries rather than written by the LLM. That way it can only mention things that are actually in the notes.

Everything runs locally on an M1 Mac with 8 GB of RAM, using a small open model through Ollama. Switching to a hosted model (Claude Haiku) is a config file change and costs about $0.0004 per request.

## Evaluation

I wrote 30 test requests the way a student might phrase them, including some that should return no match, and split them into 20 for development and 10 held out for the final test. The similarity thresholds were tuned on the development set only. The held-out set was run once at the end.

The main question was whether it's worth using the LLM to split a request into topics, or whether it's enough to embed the whole request and search with that. I also ran the topic-based approach with a hosted model (Claude Haiku) in place of the local one, using the same thresholds. Results on the held-out set:

| Approach | Right notes in top 3 | MRR | No-match cases correct | Median latency |
|---|---|---|---|---|
| Local LLM splits the request into topics | 6 of 8 | 0.75 | 1 of 2 | 0.9 s |
| Whole request embedded directly | 7 of 8 | 0.88 | 2 of 2 | 0.015 s |
| Claude Haiku splits the request into topics | 5 of 8 | 0.63 | 2 of 2 | 0.9 s |

The simpler approach won, which I didn't expect. The main reason is that the small local LLM copies an example from its prompt whenever a message has nothing to do with studying, so "can you recommend a pizza place?" came back as a question about game-playing algorithms.

Swapping in Claude fixed that completely. It made no parsing mistakes on either set and scored perfectly on the 20 development requests. On the held-out set it did slightly worse, which surprised me until I looked at the individual cases. One of the local model's hits only happened because of the copying bug: for "wut is minimax anyway", the invented topic "alpha-beta pruning" happened to find the right notes. Claude parsed that request correctly as "minimax", but a single word like that scores just under the similarity threshold, and so do "boosting" and "resolution" in the other two misses. With parsing fixed, short one-word topics falling just below the threshold are now the main weakness.

The test set is small, so a difference of one or two requests shouldn't be read as a definitive result. The topic-based approach still has one clear benefit: it can tell the student which of their topics each note covers and which ones nothing covers, which the whole-request search can't do.

## Running it

The models run in Ollama on the host machine, so install and start it first. Then one command builds the container, downloads the models if they're missing, builds the index on first run and starts the API:

```
docker compose up --build
```

The API is then at `http://localhost:8000` (`POST /recommend`, with interactive docs at `/docs`). Ollama stays outside the container because Docker on a Mac can't use the M1's GPU.

Without Docker, the same things are available from the command line:

```
uv run notematch ingest
uv run notematch recommend "I need help with Q-learning and MDPs"
uv run notematch eval
uv run notematch serve
```

## Built with

Python, Ollama (Qwen3 4B and nomic-embed-text), the Claude API (Haiku 4.5), Pydantic, NumPy, FastAPI and Docker.

The design and the reasoning behind each decision are in [design.md](design.md).
