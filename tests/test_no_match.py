import numpy as np

from notematch.recommend import NO_MATCH, NO_TOPICS, explain, rank, recommend

NINE = '{"topics": ["a","b","c","d","e","f","g","h","i"]}'


def test_all_below_threshold_gives_no_match():
    S = np.array([[0.50, 0.40], [0.54, 0.30]])
    order, scores, covered = rank(S, tau_topic=0.55, tau_min=0.0, max_results=3)
    assert order == []                    # even with tau_min=0, a product must cover a topic
    r = explain(["quantum computing", "sql"], None, None, order, scores, covered, 0.55)
    assert r.recommendations == [] and r.message == NO_MATCH
    assert r.uncovered_topics == ["quantum computing", "sql"]


def test_no_topics_gives_no_match_without_retrieval(cfg, fake_llm):
    # index=None: the pipeline must stop before touching the index or the embedder
    r = recommend("hey what's up", None, cfg, fake_llm('{"topics": []}'))
    assert r.recommendations == [] and r.message == NO_TOPICS


def test_parse_failure_gives_no_match(cfg, fake_llm):
    r = recommend("...", None, cfg, fake_llm(NINE, NINE, NINE))
    assert r.recommendations == [] and r.message == NO_TOPICS and r.parsed_query is None
