"""HTTP API (FR8): POST /recommend. Run: uv run notematch serve"""

from contextlib import asynccontextmanager

import anthropic
from fastapi import FastAPI, HTTPException

from notematch.config import load_config
from notematch.llm import embed
from notematch.recommend import get_topics, load_index, recommend
from notematch.schemas import RecommendRequest, RecommendResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once at startup, not per request. Config comes from $NOTEMATCH_CONFIG or config.toml.
    app.state.cfg = load_config()
    app.state.index = load_index(app.state.cfg)   # fails fast on an embedding-model mismatch
    # Warm both models so the first user doesn't pay the cold start (~10 s LLM, ~1.4 s embedder).
    get_topics("warm-up: load the model", app.state.cfg)
    embed(["warm-up"], app.state.cfg.embedding, "query")
    yield


app = FastAPI(title="NoteMatch", lifespan=lifespan)


@app.post("/recommend", response_model=RecommendResponse)
def recommend_endpoint(req: RecommendRequest) -> RecommendResponse:
    # Plain `def`, not `async def`: recommend() blocks on the LLM, so FastAPI runs it in a thread pool.
    try:
        return recommend(req.prompt, app.state.index, app.state.cfg)
    except ConnectionError:
        raise HTTPException(503, "Model server (Ollama) is unreachable.") from None
    except anthropic.APIConnectionError:
        raise HTTPException(503, "Claude API is unreachable.") from None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "products": len(app.state.index.products)}
