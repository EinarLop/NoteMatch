# NoteMatch

NoteMatch recommends study notes based on what a student says they're struggling with.

You type something like "I need help with Q-learning, but I think I have to understand MDPs first", and it suggests up to three notes from the catalog that cover those topics. For each one it tells you which sections match what you asked for. If nothing in the catalog fits, it says so instead of recommending something unrelated.

## How it works

The project is a small RAG (retrieval-augmented generation) system.

When the catalog is loaded, each set of notes is read only as far as its table of contents and summary page. Those pieces are turned into embeddings and stored in a vector index. The rest of the notes is never read, since it's the paid content.

When a student asks for something, an LLM reads the request and pulls out the individual topics. For the example above, that's "Q-learning" and "Markov decision processes". Each topic is then searched separately against the index. Notes are ranked by how many of the topics they cover and how closely they match.

The explanation is built directly from the matching table-of-contents entries rather than written by the LLM. That way it can only mention things that are actually in the notes.

Everything runs locally on an M1 Mac with 8 GB of RAM, using a small open model through Ollama. A setting in the config file switches it to a hosted model instead.

## Evaluation

I wrote 30 test requests the way a student might phrase them, some of which should return no match. The system is scored on whether the right notes appear in the top three, how high they rank, and whether it correctly says "no match" when it should.

I also compare two approaches: searching each topic separately versus searching the whole request at once. The results will be added here once the implementation is finished.

## Status

The design, the notes catalog and the test set are done. The implementation is in progress.

## Built with

Python, Ollama (Qwen3 4B and nomic-embed-text), Pydantic, NumPy, FastAPI and Docker.

The design and the reasoning behind each decision are in [design.md](design.md).
