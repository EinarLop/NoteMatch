"""The only module that talks to models (design.md section 6.3).

Ollama address comes from the OLLAMA_HOST env var (default http://localhost:11434).
"""

from typing import Literal, NamedTuple, TypeVar

import numpy as np
import ollama
from pydantic import BaseModel

from notematch.config import EmbeddingConfig, LLMConfig

T = TypeVar("T", bound=BaseModel)


class Usage(NamedTuple):
    prompt_tokens: int
    completion_tokens: int


def complete_json(prompt: str, schema: type[T], cfg: LLMConfig) -> tuple[T, Usage]:
    """Ask the LLM to fill `schema`. Raises pydantic.ValidationError on bad output;
    retrying is the caller's job, so it is shared across providers."""
    if cfg.provider == "ollama":
        r = ollama.chat(
            model=cfg.model,
            messages=[{"role": "user", "content": prompt}],
            format=schema.model_json_schema(),  # constrained decoding to the schema
            think=cfg.think,
            options={"temperature": cfg.temperature},
        )
        usage = Usage(r.prompt_eval_count or 0, r.eval_count or 0)
        return schema.model_validate_json(r.message.content), usage

    # ponytail: Anthropic branch (forced tool call with the schema) lands with the
    # FR9 work, together with `uv add anthropic`.
    raise NotImplementedError(f"LLM provider {cfg.provider!r} not implemented yet")


def embed(texts: list[str], cfg: EmbeddingConfig, kind: Literal["query", "document"]) -> np.ndarray:
    """Embed a batch of texts -> float32 [len(texts), dim], L2-normalized so dot = cosine."""
    prefix = cfg.query_prefix if kind == "query" else cfg.document_prefix
    r = ollama.embed(model=cfg.model, input=[prefix + t for t in texts])
    vecs = np.asarray(r.embeddings, dtype=np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
