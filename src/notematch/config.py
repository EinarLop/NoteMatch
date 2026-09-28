"""Config loading (design.md section 6): TOML file -> validated Pydantic model."""

import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    # A misspelled key is an error, not a silently ignored setting.
    model_config = ConfigDict(extra="forbid")


class LLMConfig(_Strict):
    provider: Literal["ollama", "anthropic"]
    model: str
    think: bool = False
    temperature: float = 0.0
    max_retries: int = Field(default=2, ge=0)


class EmbeddingConfig(_Strict):
    model: str
    query_prefix: str = ""
    document_prefix: str = ""
    # Specific to this embedding model; re-tune whenever the model changes.
    tau_topic: float
    tau_min: float


class RetrievalConfig(_Strict):
    mode: Literal["per_topic", "whole_prompt"] = "per_topic"
    max_results: int = Field(default=3, ge=1)


class IngestionConfig(_Strict):
    toc_stoplist: list[str] = []


class PathsConfig(_Strict):
    catalog: Path = Path("catalog")
    index: Path = Path("index")


class Config(_Strict):
    llm: LLMConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig = RetrievalConfig()
    ingestion: IngestionConfig = IngestionConfig()
    paths: PathsConfig = PathsConfig()


def load_config(path: str | Path | None = None) -> Config:
    """Load from `path`, else $NOTEMATCH_CONFIG, else ./config.toml."""
    path = Path(path or os.environ.get("NOTEMATCH_CONFIG", "config.toml"))
    with path.open("rb") as f:
        return Config.model_validate(tomllib.load(f))
