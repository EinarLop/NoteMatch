"""Evaluation harness (design.md section 7).

Two phases so threshold tuning stays cheap:
  run_cases: the slow part (LLM parse + embedding), once per case -> S matrices
  score:     the fast part (rank + metrics) for given thresholds, no model calls
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from notematch.config import Config
from notematch.recommend import Index, get_topics, load_index, rank, retrieve


@dataclass
class CaseRun:
    case: dict
    topics: list[str]          # [] when the parse found nothing or failed
    S: np.ndarray | None       # [n_products, T], None when there are no topics
    latency_ms: float
    tokens: int


def load_cases(path: Path, split: str) -> list[dict]:
    cases = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [c for c in cases if split == "all" or c["split"] == split]


def run_cases(cases: list[dict], index: Index, cfg: Config) -> list[CaseRun]:
    get_topics("warm-up: load the model", cfg)   # keep the cold start out of the latency numbers
    runs = []
    for n, c in enumerate(cases, 1):
        t0 = time.perf_counter()
        parsed, usage, _ = get_topics(c["prompt"], cfg)
        topics = parsed.topics if parsed else []
        S = retrieve(topics, index, cfg)[0] if topics else None
        runs.append(CaseRun(c, topics, S, (time.perf_counter() - t0) * 1000,
                            usage.prompt_tokens + usage.completion_tokens))
        print(f"  [{n}/{len(cases)}] {c['id']} {topics}", flush=True)
    return runs


def score(runs: list[CaseRun], index: Index, tau_topic: float, tau_min: float, max_results: int):
    """-> (metrics, per-case rows). Pure numpy: safe to call thousands of times for tuning."""
    ids = [p.product_id for p in index.products]
    rows = []
    for r in runs:
        expected = r.case["expected"]
        returned = [ids[p] for p in rank(r.S, tau_topic, tau_min, max_results)[0]] if r.topics else []
        ranks = [i + 1 for i, pid in enumerate(returned) if pid in expected]
        row = {"id": r.case["id"], "prompt": r.case["prompt"], "topics": r.topics,
               "expected": expected, "returned": returned}
        if expected:
            row |= {"hit": bool(ranks), "rr": 1 / ranks[0] if ranks else 0.0,
                    "recall": len(ranks) / len(expected)}
            row["failure"] = None if ranks else _diagnose(r, expected, ids, max_results)
        else:
            row["correct"] = not returned
            row["failure"] = None if not returned else "false positive"
        rows.append(row)

    match = [x for x in rows if x["expected"]]
    nomatch = [x for x in rows if not x["expected"]]
    lat = [r.latency_ms for r in runs]
    metrics = {
        "n_match": len(match), "n_nomatch": len(nomatch),
        "hit@3": _mean(x["hit"] for x in match),
        "mrr": _mean(x["rr"] for x in match),
        "recall@3": _mean(x["recall"] for x in match),
        "nomatch_acc": _mean(x["correct"] for x in nomatch),
        "latency_p50_ms": round(float(np.percentile(lat, 50))) if lat else None,
        "latency_p95_ms": round(float(np.percentile(lat, 95))) if lat else None,
        "mean_tokens": round(float(np.mean([r.tokens for r in runs]))) if runs else None,
    }
    return metrics, rows


def _mean(xs) -> float | None:
    xs = [float(x) for x in xs]
    return round(sum(xs) / len(xs), 3) if xs else None


def _diagnose(r: CaseRun, expected: list[str], ids: list[str], max_results: int) -> str:
    """Why a match case missed: which stage to fix first."""
    if not r.topics:
        return "parse: no topics"
    raw_top = [ids[p] for p in np.argsort(-r.S.mean(axis=1))[:max_results]]   # ranking with no thresholds
    if any(pid in expected for pid in raw_top):
        return "threshold: expected product ranks top-3 but was filtered out"
    return "retrieval/parse: expected product not in top-3 even without thresholds"


def evaluate(cfg: Config, cases_path: Path, split: str, config_name: str) -> dict:
    index = load_index(cfg)
    cases = load_cases(cases_path, split)
    print(f"Running {len(cases)} {split} cases ({cfg.retrieval.mode}, {cfg.llm.model})...")
    runs = run_cases(cases, index, cfg)
    e = cfg.embedding
    metrics, rows = score(runs, index, e.tau_topic, e.tau_min, cfg.retrieval.max_results)

    print(f"\nFailures (tau_topic={e.tau_topic}, tau_min={e.tau_min}):")
    for x in rows:
        if x["failure"]:
            print(f"  {x['id']} [{x['failure']}]\n      topics={x['topics']}\n"
                  f"      expected={x['expected'] or 'NO MATCH'} returned={x['returned']}")
    print(f"\nMatch cases ({metrics['n_match']}):  Hit@3={metrics['hit@3']}  MRR={metrics['mrr']}  "
          f"Recall@3={metrics['recall@3']}")
    correct = sum(x.get("correct", False) for x in rows)
    print(f"No-match cases:      accuracy={metrics['nomatch_acc']} ({correct}/{metrics['n_nomatch']})")
    print(f"Latency: p50={metrics['latency_p50_ms']} ms  p95={metrics['latency_p95_ms']} ms  "
          f"tokens/request={metrics['mean_tokens']}")

    out = Path("eval/results") / f"{config_name}-{split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": cfg.model_dump(mode="json"), "split": split,
                               "metrics": metrics, "cases": rows}, indent=2))
    print(f"\nWritten to {out}")
    return metrics
