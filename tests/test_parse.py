from notematch.recommend import parse_query

NINE = '{"topics": ["a","b","c","d","e","f","g","h","i"]}'  # breaks max_length=8


def test_valid_output_is_normalized(cfg, fake_llm):
    llm = fake_llm('{"topics": [" Q-Learning ", "MDPs", "q-learning", ""]}')
    parsed, usage, retries = parse_query("...", cfg, llm)
    assert parsed.topics == ["q-learning", "mdps"]   # stripped, lowercased, deduped, blanks dropped
    assert retries == 0 and usage.prompt_tokens == 10


def test_invalid_then_valid_retries_once(cfg, fake_llm):
    llm = fake_llm(NINE, '{"topics": ["minimax"]}')
    parsed, _, retries = parse_query("...", cfg, llm)
    assert parsed.topics == ["minimax"] and retries == 1 and llm.calls == 2


def test_always_invalid_gives_none_after_max_retries(cfg, fake_llm):
    llm = fake_llm(NINE, NINE, NINE)
    parsed, _, _ = parse_query("...", cfg, llm)
    assert parsed is None and llm.calls == cfg.llm.max_retries + 1


def test_empty_topics_is_valid_and_not_retried(cfg, fake_llm):
    llm = fake_llm('{"topics": []}')
    parsed, _, _ = parse_query("hey what's up", cfg, llm)
    assert parsed.topics == [] and llm.calls == 1
