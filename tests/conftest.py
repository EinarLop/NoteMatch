import pytest

from notematch.config import Config
from notematch.llm import Usage


@pytest.fixture
def cfg():
    return Config.model_validate({
        "llm": {"provider": "ollama", "model": "fake", "max_retries": 2},
        "embedding": {"model": "fake", "tau_topic": 0.55, "tau_min": 0.30},
    })


@pytest.fixture
def fake_llm():
    """Build a fake `complete_json` that returns the given raw JSON outputs in order,
    validating them exactly like the real one. `calls` counts invocations."""
    def make(*outputs):
        it = iter(outputs)
        def complete(prompt, schema, llm_cfg):
            complete.calls += 1
            return schema.model_validate_json(next(it)), Usage(10, 5)
        complete.calls = 0
        return complete
    return make
