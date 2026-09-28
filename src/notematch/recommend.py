"""Online pipeline (design.md section 5): prompt -> parse -> retrieve -> rank -> explain."""

import json
import logging
import time
import uuid
from dataclasses import dataclass

import numpy as np
from pydantic import TypeAdapter, ValidationError

from notematch.config import Config
from notematch.llm import Usage, complete_json, embed
from notematch.schemas import (IndexEntry, ParsedQuery, ProductRecord, Recommendation,
                               RecommendResponse, SectionMatch, TopicMatch)

log = logging.getLogger("notematch")

# ponytail: inline few-shot; the 4B model sometimes copies an example (design 9, #11).
PARSE_PROMPT = """Extract the specific study topics the student needs help with.
Topics are short noun phrases. Split combined requests into separate topics.
Include every topic the student mentions needing, even ones they plan to study later.
If the message contains no study topic, return an empty list.

Student: "struggling with minimax and alpha-beta for my game AI assignment"
{{"topics": ["minimax", "alpha-beta pruning"]}}
Student: "hi, what's up?"
{{"topics": []}}
Student: {prompt}"""

NO_TOPICS = "Couldn't identify any study topics in your request."
NO_MATCH = "No product in the catalog covers these topics."


@dataclass
class Index:
    products: list[ProductRecord]
    entries: list[IndexEntry]
    vectors: np.ndarray            # [n_entries, dim], row i <-> entries[i]
    rows: list[np.ndarray]         # rows[p] = entry indices belonging to products[p]


def load_index(cfg: Config) -> Index:
    d = cfg.paths.index
    manifest = json.loads((d / "manifest.json").read_text())
    if manifest["embedding_model"] != cfg.embedding.model:
        raise RuntimeError(f"Index built with {manifest['embedding_model']!r} but config uses "
                           f"{cfg.embedding.model!r}: vectors aren't comparable. Re-run ingest.")
    products = TypeAdapter(list[ProductRecord]).validate_json((d / "products.json").read_bytes())
    entries = TypeAdapter(list[IndexEntry]).validate_json((d / "entries.json").read_bytes())
    pids = np.array([e.product_id for e in entries])
    rows = [np.flatnonzero(pids == p.product_id) for p in products]
    return Index(products, entries, np.load(d / "vectors.npy"), rows)


def parse_query(prompt: str, cfg: Config, complete=complete_json) -> tuple[ParsedQuery | None, Usage, int]:
    """LLM: prompt -> topics. Returns (parsed, or None if every attempt failed; usage; retries).
    `complete` is a parameter so tests can pass a fake LLM."""
    text = PARSE_PROMPT.format(prompt=json.dumps(prompt))
    for attempt in range(cfg.llm.max_retries + 1):
        try:
            parsed, usage = complete(text, ParsedQuery, cfg.llm)
        except ValidationError as e:
            # ponytail: tokens of failed attempts aren't counted (complete_json raises before returning them)
            text += f"\n\nYour previous answer was invalid ({e.errors()[0]['msg']}). Answer again."
            continue
        topics = list(dict.fromkeys(t.strip().lower() for t in parsed.topics if t.strip()))  # dedupe, keep order
        return ParsedQuery(topics=topics), usage, attempt
    return None, Usage(0, 0), cfg.llm.max_retries


def retrieve(topics: list[str], index: Index, cfg: Config):
    """-> S [n_products, T]: best similarity of each product for each topic;
    top[p][t]: product p's 3 best (entry index, similarity) for topic t."""
    sims = index.vectors @ embed(topics, cfg.embedding, "query").T      # [n_entries, T]
    S = np.array([sims[rows].max(axis=0) for rows in index.rows])       # max, not mean
    top = [[[(int(i), float(sims[i, t])) for i in rows[np.argsort(-sims[rows, t])[:3]]]
            for t in range(len(topics))] for rows in index.rows]
    return S, top


def rank(S: np.ndarray, tau_topic: float, tau_min: float, max_results: int):
    """-> (product indices best first, scores [P], covered [P, T]). Pure numpy, no models."""
    covered = S >= tau_topic
    scores = (S * covered).mean(axis=1)              # uncovered topics contribute 0
    eligible = (scores >= tau_min) & covered.any(axis=1)
    order = [int(p) for p in np.argsort(-scores, kind="stable") if eligible[p]][:max_results]
    return order, scores, covered


def explain(topics, index: Index, top, order, scores, covered, tau_topic: float) -> RecommendResponse:
    """Template only, no LLM: every claim comes from the index (FR5)."""
    recs = []
    for p in order:
        matches = []
        for t, topic in enumerate(topics):
            if covered[p, t]:
                sections = [SectionMatch(text=index.entries[i].text if index.entries[i].kind == "toc"
                                         else "(product summary)", similarity=round(sim, 3))
                            for i, sim in top[p][t] if sim >= tau_topic]
                matches.append(TopicMatch(topic=topic, sections=sections))
        prod = index.products[p]
        recs.append(Recommendation(product_id=prod.product_id, title=prod.title, price=prod.price,
                                   purchase_url=prod.purchase_url, score=round(float(scores[p]), 3),
                                   covered_topics=matches))
    covered_by_returned = covered[order].any(axis=0) if order else np.zeros(len(topics), bool)
    uncovered = [t for t, c in zip(topics, covered_by_returned) if not c]
    if not recs:
        msg = NO_MATCH
    else:
        msg = f"Found {len(recs)} product(s) covering {len(topics) - len(uncovered)} of {len(topics)} topics."
        if uncovered:
            msg += f" No product covers: {', '.join(uncovered)}."
    return RecommendResponse(parsed_query=ParsedQuery(topics=topics), recommendations=recs,
                             uncovered_topics=uncovered, message=msg)


def get_topics(prompt: str, cfg: Config, complete=complete_json) -> tuple[ParsedQuery | None, Usage, int]:
    """Topics for the configured retrieval mode (shared by recommend and evaluate)."""
    if cfg.retrieval.mode == "whole_prompt":
        return ParsedQuery(topics=[prompt]), Usage(0, 0), 0   # experiment B (design 7.4): no splitting
    return parse_query(prompt, cfg, complete)


def recommend(prompt: str, index: Index, cfg: Config, complete=complete_json) -> RecommendResponse:
    t0 = time.perf_counter()
    parsed, usage, retries = get_topics(prompt, cfg, complete)
    t1 = time.perf_counter()

    if not parsed or not parsed.topics:
        response = RecommendResponse(parsed_query=parsed, recommendations=[], uncovered_topics=[],
                                     message=NO_TOPICS)
        t2 = t3 = t1
    else:
        S, top = retrieve(parsed.topics, index, cfg)
        t2 = time.perf_counter()
        e = cfg.embedding
        order, scores, covered = rank(S, e.tau_topic, e.tau_min, cfg.retrieval.max_results)
        response = explain(parsed.topics, index, top, order, scores, covered, e.tau_topic)
        t3 = time.perf_counter()

    ms = lambda a, b: round((b - a) * 1000)
    log.info(json.dumps({
        "request_id": uuid.uuid4().hex[:8],
        "latency_ms": {"parse": ms(t0, t1), "embed_retrieve": ms(t1, t2), "rank_explain": ms(t2, t3),
                       "total": ms(t0, t3)},
        "prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens,
        "retries": retries, "n_results": len(response.recommendations),
    }))
    return response
