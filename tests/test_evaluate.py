import numpy as np
import pytest

from notematch.evaluate import CaseRun, score
from notematch.recommend import Index
from notematch.schemas import ProductRecord


def product(pid):
    return ProductRecord(product_id=pid, title=pid, subject="ai", price=1, purchase_url="https://x.com",
                         summary="", toc=[])


INDEX = Index([product("a"), product("b"), product("c")], [], np.zeros((0, 1)), [])


def run(expected, S, topics=("t",)):
    return CaseRun({"id": "x", "prompt": "", "expected": expected}, list(topics),
                   None if S is None else np.array(S), latency_ms=100, tokens=10)


def test_metrics_by_hand():
    runs = [
        run(["a"], [[0.9], [0.7], [0.1]]),              # returns [a, b]: hit, rank 1
        run(["b", "c"], [[0.9], [0.7], [0.1]]),         # returns [a, b]: hit, rank 2, recall 1/2
        run(["b"], [[0.9], [0.6], [0.1]]),              # returns [a]: b is 2nd raw but 0.6 < tau 0.65
        run([], [[0.5], [0.4], [0.3]]),                 # nothing covered: correct no-match
        run([], [[0.8], [0.1], [0.1]]),                 # returns [a]: false positive
        run([], None, topics=()),                       # no topics: correct no-match
    ]
    m, rows = score(runs, INDEX, tau_topic=0.65, tau_min=0.0, max_results=2)
    assert m["hit@3"] == pytest.approx(2 / 3, abs=1e-3)
    assert m["mrr"] == pytest.approx((1 + 0.5 + 0) / 3, abs=1e-3)
    assert m["recall@3"] == pytest.approx((1 + 0.5 + 0) / 3, abs=1e-3)
    assert m["nomatch_acc"] == pytest.approx(2 / 3, abs=1e-3)
    assert rows[2]["failure"].startswith("threshold")
    assert rows[4]["failure"] == "false positive"

    # Expected product not even in the raw top-2 -> retrieval problem, not threshold.
    _, rows = score([run(["c"], [[0.9], [0.7], [0.6]])], INDEX, tau_topic=0.65, tau_min=0.0, max_results=2)
    assert rows[0]["failure"].startswith("retrieval")
