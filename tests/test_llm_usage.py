"""Token accounting on the LLM client: offline, with a fake OpenAI client."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cv_screener.config import Settings
from cv_screener.llm import LLM, LLMError, estimate_tokens


def response(content: str, prompt: int | None = None, completion: int | None = None):
    usage = (
        None
        if prompt is None and completion is None
        else SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion)
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=None))],
        usage=usage,
    )


class ScriptedClient:
    """Returns the scripted responses in order; an Exception instance is raised instead."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_llm(script, **overrides) -> LLM:
    cfg = Settings(api_key="test", model="model-a", **overrides)
    return LLM(cfg, client=ScriptedClient(script))


def test_usage_starts_empty_and_sums_reported_tokens():
    llm = make_llm([response("hi", 100, 20), response("again", 40, 5)])
    assert llm.usage.requests == 0 and llm.usage.total_tokens == 0

    llm.chat([{"role": "user", "content": "a"}])
    llm.chat([{"role": "user", "content": "b"}])

    assert llm.usage.requests == 2 and llm.usage.calls == 2
    assert llm.usage.prompt_tokens == 140 and llm.usage.completion_tokens == 25
    assert llm.usage.total_tokens == 165
    assert llm.usage.estimated_calls == 0 and not llm.usage.estimated


def test_reset_usage_zeroes_the_counters():
    llm = make_llm([response("hi", 100, 20), response("hi", 7, 3)])
    llm.chat([{"role": "user", "content": "a"}])
    llm.reset_usage()
    assert llm.usage.requests == 0 and llm.usage.total_tokens == 0

    llm.chat([{"role": "user", "content": "b"}])
    assert llm.usage.prompt_tokens == 7 and llm.usage.completion_tokens == 3


def test_missing_usage_falls_back_to_a_marked_char_estimate():
    llm = make_llm([response("four chars here")])
    messages = [{"role": "user", "content": "x" * 400}]
    tools = [{"type": "function", "function": {"name": "t"}}]

    llm.chat(messages, tools=tools)

    assert llm.usage.estimated_calls == 1 and llm.usage.estimated
    # prompt is estimated from the serialized messages plus the tool schemas, so it exceeds
    # the estimate of the message text alone but stays in the same ballpark
    assert llm.usage.prompt_tokens > estimate_tokens("x" * 400)
    assert llm.usage.completion_tokens == estimate_tokens("four chars here")


def test_failed_attempts_count_as_requests_but_not_as_calls():
    llm = make_llm(
        [RuntimeError("429"), response("ok", 10, 2)],
        fallback_models=("model-b",),
    )
    llm.chat([{"role": "user", "content": "a"}])
    assert llm.usage.requests == 2 and llm.usage.calls == 1
    assert llm.usage.total_tokens == 12


def test_every_attempt_of_a_hopeless_call_is_counted():
    llm = make_llm([RuntimeError("boom")] * 4, fallback_models=("model-b",))
    with pytest.raises(LLMError):
        llm.chat([{"role": "user", "content": "a"}])
    assert llm.usage.requests == 4 and llm.usage.calls == 0
