"""Data contracts shared by ingestion, query pipeline and API (design.md section 3)."""

from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


# --- Catalog input (metadata.csv) ---

class Product(BaseModel):
    product_id: str
    title: str
    subject: str
    price: float = Field(ge=0)
    purchase_url: HttpUrl


# --- Ingestion output ---

class TocEntry(BaseModel):
    text: str
    depth: int = Field(ge=1)  # 1 = section, 2 = subsection (as returned by get_toc)
    page: int | None = None


class ProductRecord(Product):
    summary: str  # LLM-generated at ingestion
    toc: list[TocEntry]


class IndexEntry(BaseModel):
    entry_id: str  # "adversarial_search:toc:3" / "adversarial_search:summary"
    product_id: str
    kind: Literal["toc", "summary"]
    text: str  # bare text, shown in explanations


# --- Query parser output (LLM -> JSON) ---

class ParsedQuery(BaseModel):
    # No min_length: structured output would force the model to invent a topic.
    # An empty list means "no study topics" and short-circuits to a no-match.
    topics: list[str] = Field(max_length=8)


# --- API ---

class RecommendRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=1000)


class SectionMatch(BaseModel):
    text: str
    similarity: float


class TopicMatch(BaseModel):
    topic: str
    sections: list[SectionMatch]


class Recommendation(BaseModel):
    product: Product  # catalog fields only; summary and TOC stay server-side
    score: float
    covered_topics: list[TopicMatch]


class RecommendResponse(BaseModel):
    parsed_query: ParsedQuery | None
    recommendations: list[Recommendation]
    uncovered_topics: list[str]
    message: str
