"""Agent loop tests with a scripted fake LLM: no network, no key."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cv_screener.agent.loop import Agent
from cv_screener.agent.tools import TOOLS, dispatch
from cv_screener.config import Settings
from cv_screener.index.embeddings import HashEmbedder
from cv_screener.index.store import CandidateIndex
from cv_screener.llm import LLM


def tool_call(tool: str, **args):
    return SimpleNamespace(
        id=f"call_{tool}", function=SimpleNamespace(name=tool, arguments=json.dumps(args))
    )


def message(content: str | None = None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


class FakeClient:
    """Mimics openai.OpenAI just enough: returns scripted messages in order, records requests."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        msg = self.script.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


@pytest.fixture
def index(tmp_path: Path, sample_candidates) -> CandidateIndex:
    idx = CandidateIndex(path=tmp_path / "chroma", embedder=HashEmbedder())
    idx.rebuild(sample_candidates)
    return idx


def make_agent(index, script):
    client = FakeClient(script)
    llm = LLM(Settings(api_key="test", model="fake"), client=client)
    return Agent(llm, index), client


def test_agent_executes_tool_and_grounds_answer(index):
    agent, client = make_agent(
        index,
        [
            message(tool_calls=[tool_call("search_candidates", languages=["Spanish"], k=10)]),
            message("Ana Ruiz (Spanish native) and Bob Lee (Spanish B2) speak Spanish."),
        ],
    )
    answer = agent.ask("Which candidates speak Spanish?")
    assert [t.name for t in answer.trace] == ["search_candidates"]
    assert answer.trace[0].count == 2 and set(answer.trace[0].ids) == {"p01-ana-ml", "p02-bob-qa"}
    assert answer.grounded and answer.ungrounded_names == []
    # the second request carries the tool result back to the model
    roles = [m["role"] for m in client.requests[1]["messages"]]
    assert roles[-2:] == ["assistant", "tool"]
    assert client.requests[0]["tools"] == TOOLS


def test_agent_flags_names_that_no_tool_returned(index):
    agent, _ = make_agent(
        index,
        [
            message(tool_calls=[tool_call("search_candidates", languages=["Hebrew"])]),
            message("Cy Nowak speaks Hebrew."),  # hallucination: tool returned nothing
        ],
    )
    answer = agent.ask("Who speaks Hebrew?")
    assert answer.trace[0].count == 0
    assert not answer.grounded and answer.ungrounded_names == ["Cy Nowak"]


def test_agent_chains_find_by_name_and_get_candidate(index):
    agent, client = make_agent(
        index,
        [
            message(tool_calls=[tool_call("find_by_name", name="Nowack")]),
            message(tool_calls=[tool_call("get_candidate", candidate_id="p03-cy-fe")]),
            message("Cy Nowak is a junior frontend developer working with React and TypeScript."),
        ],
    )
    answer = agent.ask("Summarize the profile of Nowack")
    assert [t.name for t in answer.trace] == ["find_by_name", "get_candidate"]
    profile = json.loads(client.requests[2]["messages"][-1]["content"])
    assert profile["full_name"] == "Cy Nowak" and "photo_prompt" not in profile
    assert answer.grounded


def test_agent_stops_after_max_steps(index):
    looping = [message(tool_calls=[tool_call("find_by_name", name="Ana")])] * 3
    agent, client = make_agent(index, [*looping, message("Ana Ruiz.")])
    agent.max_steps = 3
    answer = agent.ask("Loop forever")
    assert len(answer.trace) == 3 and answer.text == "Ana Ruiz."
    assert "Answer now" in client.requests[-1]["messages"][-1]["content"]


def test_dispatch_unknown_tool_and_missing_candidate(index):
    result, n = dispatch(index, "nope", {})
    assert n == 0 and "unknown tool" in result
    result, n = dispatch(index, "get_candidate", {"candidate_id": "missing"})
    assert n == 0 and "no candidate" in result
