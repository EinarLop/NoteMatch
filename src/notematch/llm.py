"""The only module that talks to models (design.md section 6.3).

Ollama address comes from the OLLAMA_HOST env var (default http://localhost:11434).
The Claude API key comes from ANTHROPIC_API_KEY. Embeddings always run on Ollama.
"""

from functools import cache
from typing import Literal, NamedTuple, TypeVar

import anthropic
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

    if cfg.provider == "anthropic":
        r = _anthropic().messages.parse(
            model=cfg.model,
            max_tokens=1024,                     # outputs here are a short summary or a topic list
            # SDK 1.x dropped `temperature` from its signatures; Haiku 4.5 still honours it via extra_body.
            # ponytail: newer Claude models (Opus 4.7+, Sonnet 5) reject it; add a switch when moving to one.
            extra_body={"temperature": cfg.temperature},
            messages=[{"role": "user", "content": prompt}],
            output_format=schema,                # structured outputs: JSON constrained to the schema
        )
        if r.parsed_output is None:              # refusal or truncation: no JSON to validate
            raise RuntimeError(f"Claude returned no structured output (stop_reason={r.stop_reason})")
        usage = Usage(r.usage.input_tokens, r.usage.output_tokens)
        # Re-validate with our own model so bad output raises pydantic.ValidationError like the Ollama branch.
        return schema.model_validate(r.parsed_output.model_dump()), usage

    raise ValueError(f"Unknown LLM provider {cfg.provider!r}")


@cache
def _anthropic() -> anthropic.Anthropic:
    """One client per process. Reads ANTHROPIC_API_KEY from the environment, never from config."""
    return anthropic.Anthropic()


def embed(texts: list[str], cfg: EmbeddingConfig, kind: Literal["query", "document"]) -> np.ndarray:
    """Embed a batch of texts -> float32 [len(texts), dim], L2-normalized so dot = cosine."""
    prefix = cfg.query_prefix if kind == "query" else cfg.document_prefix
    r = ollama.embed(model=cfg.model, input=[prefix + t for t in texts])
    vecs = np.asarray(r.embeddings, dtype=np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
