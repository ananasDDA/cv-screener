"""The research harness decides pass/fail and prices every run, so it is tested like library
code: offline, with a fake LLM client. Nothing here touches the network or the JobResQA
download, which is git-ignored and may be absent."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cv_screener.config import Settings  # noqa: E402
from cv_screener.index.embeddings import HashEmbedder  # noqa: E402
from cv_screener.llm import LLM  # noqa: E402
from research import harness, history, jobresqa, scaling, stats  # noqa: E402


def response(content: str, prompt: int = 900, completion: int = 40):
    message = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion),
    )


class OneShotClient:
    def __init__(self, content: str):
        self.content = content
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return response(self.content)


@pytest.fixture
def names() -> dict[str, str]:
    return {"p01-ana-ml": "Ana Ruiz", "p02-bob-qa": "Bob Lee", "p03-cy-fe": "Cy Nowak"}


def make_llm(content: str) -> tuple[LLM, OneShotClient]:
    client = OneShotClient(content)
    return LLM(Settings(api_key="test", model="fake"), client=client), client


def test_corpus_holds_every_candidate(sample_candidates):
    corpus = harness.corpus_text(sample_candidates)
    for candidate in sample_candidates:
        assert candidate.full_name in corpus
    prompt = harness.baseline_prompt(corpus)
    assert corpus in prompt and "never guess" in prompt


def test_prompt_arm_sends_one_call_with_no_tools(sample_candidates, names):
    llm, client = make_llm("Ana Ruiz speaks Spanish.")
    case = {"id": "spanish", "question": "Who speaks Spanish?", "must_include": ["p01-ana-ml"]}

    rec = harness.run_prompt_arm(llm, harness.corpus_text(sample_candidates), case, 1, names)

    assert rec.passed and rec.reasons == []
    assert rec.llm_calls == 1 and rec.tool_calls == 0
    assert rec.total_tokens == 940 and not rec.estimated
    assert len(client.requests) == 1 and "tools" not in client.requests[0]
    assert client.requests[0]["temperature"] == harness.TEMPERATURE


def test_prompt_arm_is_not_failed_for_having_no_tool_calls(sample_candidates, names):
    """`check()` demands at least one tool call; arm B has none by construction, so that one
    reason is dropped and only the content checks decide."""
    llm, _ = make_llm("Nobody in the dataset speaks Hebrew.")
    case = {"id": "hebrew", "question": "Who speaks Hebrew?", "expect_no_match": True}

    rec = harness.run_prompt_arm(llm, harness.corpus_text(sample_candidates), case, 1, names)

    assert rec.passed and rec.reasons == []


def test_prompt_arm_reports_a_missing_candidate(sample_candidates, names):
    llm, _ = make_llm("Only Bob Lee matches.")
    case = {"id": "spanish", "question": "Who speaks Spanish?", "must_include": ["p01-ana-ml"]}

    rec = harness.run_prompt_arm(llm, harness.corpus_text(sample_candidates), case, 1, names)

    assert not rec.passed and rec.reasons == ["missing Ana Ruiz"]


def test_prompt_arm_records_a_crash_as_a_failed_case(sample_candidates, names):
    llm = LLM(Settings(api_key="test", model="fake"), client=OneShotClient(""))
    llm.client.chat.completions.create = _boom
    case = {"id": "x", "question": "q", "must_include": []}

    rec = harness.run_prompt_arm(llm, "corpus", case, 1, names)

    assert not rec.passed and rec.error and "RuntimeError" in rec.error


def _boom(**kwargs):
    raise RuntimeError("provider down")


def test_p95_is_nearest_rank_and_median_rounds():
    assert stats.p95(list(range(1, 21))) == 19
    assert stats.p95([5.0]) == 5.0
    assert stats.median([1, 2, 3, 4]) == 2.5


def test_arm_summary_counts_pass_rate_and_cost_per_correct_answer():
    records = [
        _run("tools", "a", True, 100, 2.0),
        _run("tools", "b", False, 300, 4.0),
        _run("prompt", "a", True, 1000, 1.0),
    ]
    rows = {r["arm"]: r for r in stats.arm_summary(records)}
    tools = rows["A: tools (agent)"]
    assert tools["pass rate"] == "1/2 = 50%"
    assert tools["tok / correct answer"] == 400  # both runs are paid for, one answer is right
    assert rows["B: all CVs in the prompt"]["tok / correct answer"] == 1000


def test_per_question_puts_both_arms_on_one_row():
    records = [_run("tools", "a", True, 100, 2.0), _run("prompt", "a", False, 900, 5.0)]
    (row,) = stats.per_question(records)
    assert row["case"] == "a"
    assert row["A pass"] == "1/1" and row["B pass"] == "0/1"
    assert row["A tok"] == 100 and row["B tok"] == 900
    assert "prompt: missing someone" in row["failure reasons"]


def _run(arm: str, case: str, passed: bool, tokens: int, seconds: float) -> dict:
    return {
        "arm": arm,
        "case_id": case,
        "rep": 1,
        "passed": passed,
        "reasons": [] if passed else ["missing someone"],
        "llm_calls": 1,
        "tool_calls": 1,
        "prompt_tokens": tokens,
        "completion_tokens": 0,
        "total_tokens": tokens,
        "estimated": False,
        "requests": 1,
        "seconds": seconds,
        "model": "fake",
        "answer": "",
        "error": None,
        "extra": {},
    }


def test_grow_keeps_the_real_candidates_first_and_perturbs_the_copies(sample_candidates):
    grown = scaling.grow(sample_candidates, 8)
    assert len(grown) == 8
    assert [c.id for c in grown[:3]] == [c.id for c in sample_candidates]
    assert len({c.id for c in grown}) == 8
    copy = grown[3]
    assert copy.full_name != sample_candidates[0].full_name
    assert sorted(copy.skills) == sorted(sample_candidates[0].skills)  # rotated, not invented


def test_prompt_size_curve_grows_and_is_deterministic(sample_candidates):
    curve = scaling.prompt_size_curve(sample_candidates, (3, 9))
    assert [r["N documents"] for r in curve] == [3, 9]
    assert curve[1]["est. prompt tokens"] > curve[0]["est. prompt tokens"]
    assert scaling.prompt_size_curve(sample_candidates, (3, 9)) == curve


class StubResumeIndex:
    """Enough of `jobresqa.ResumeIndex` for the loop: no Chroma, no embedding model."""

    def __init__(self, order: list[str]):
        self.order = order
        self.queries: list[tuple[str, int]] = []

    def search(self, query: str, k: int = 2) -> list[dict[str, str]]:
        self.queries.append((query, k))
        return [{"resume_id": i, "name": i, "resume": f"CV of {i}"} for i in self.order[:k]]


def tool_call(**args):
    return SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="search_resumes", arguments=json.dumps(args)),
    )


class ToolThenAnswerClient:
    """First reply asks for the tool, the second answers."""

    def __init__(self, answer: str):
        self.script = [
            SimpleNamespace(content=None, tool_calls=[tool_call(query="Ana Ruiz", k=2)]),
            SimpleNamespace(content=answer, tool_calls=None),
        ]
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        message = self.script.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(prompt_tokens=500, completion_tokens=20),
        )


def test_external_tools_arm_records_retrieval_and_sums_both_calls():
    client = ToolThenAnswerClient("Senior Associate")
    llm = LLM(Settings(api_key="test", model="fake"), client=client)
    index = StubResumeIndex(["r1", "r2", "r3"])
    item = jobresqa.Item("ex1", "r2", "Ana Ruiz", "What is Ana Ruiz's title?", "Senior", "Basic")

    row = jobresqa.run_tools_arm(llm, index, item)

    assert row["answer"] == "Senior Associate"
    assert row["tool_calls"] == 1 and row["retrieved"] == ["r1", "r2"]
    assert row["retrieval_hit"] is True  # r2 is the gold résumé and it came back
    assert row["llm_calls"] == 2 and row["total_tokens"] == 1040
    assert client.requests[0]["tools"] == jobresqa.SEARCH_TOOL


def test_external_tools_arm_flags_a_miss():
    client = ToolThenAnswerClient("Unknown")
    llm = LLM(Settings(api_key="test", model="fake"), client=client)
    item = jobresqa.Item("ex2", "r9", "Ana Ruiz", "q", "Senior", "Basic")

    row = jobresqa.run_tools_arm(llm, StubResumeIndex(["r1", "r2"]), item)

    assert row["retrieval_hit"] is False and row["retrieved"] == ["r1", "r2"]


def test_resume_index_rebuilds_when_the_selection_changed(tmp_path, monkeypatch):
    """Regression: the store survives between runs. Upserting a smaller selection on top of a
    bigger one left orphan documents, and a search then returned an id with no name behind it."""
    monkeypatch.setattr(jobresqa, "CHROMA_DIR", tmp_path / "chroma")
    monkeypatch.setattr(jobresqa, "FastEmbedder", lambda: HashEmbedder())

    first = {f"r{i}": f"resume number {i} about databases" for i in range(5)}
    jobresqa.ResumeIndex(first, dict.fromkeys(first, "Someone"))

    second = {"r0": "resume number 0 about databases", "r1": "resume number 1 about databases"}
    index = jobresqa.ResumeIndex(second, dict.fromkeys(second, "Someone"))

    assert sorted(index.collection.get(include=[])["ids"]) == ["r0", "r1"]
    hits = index.search("databases", k=5)
    assert {h["resume_id"] for h in hits} <= set(second)  # never an id we cannot resolve


def test_judge_parses_the_verdict_line():
    class JudgeClient:
        def __init__(self, text):
            self.text = text
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            message = SimpleNamespace(content=self.text, tool_calls=None)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message)],
                usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10),
            )

    item = jobresqa.Item("ex", "r1", "Ana", "q", "Senior Associate", "Basic")
    good = LLM(Settings(api_key="t", model="f"), client=JudgeClient("CORRECT|same role stated"))
    bad = LLM(Settings(api_key="t", model="f"), client=JudgeClient("INCORRECT|names a different"))

    assert jobresqa.judge(good, item, "Senior Associate") == {
        "correct": True,
        "reason": "same role stated",
        "requests": 1,
        "judge_model": "f",
    }
    assert jobresqa.judge(bad, item, "Intern")["correct"] is False


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What is the candidate's title?", "What is Ana Ruiz's title?"),
        ("Does the candidate speak Spanish?", "Does Ana Ruiz speak Spanish?"),
        ("What challenges did his/her internship bring?", "About Ana Ruiz: "),
        ("How long has Ana Ruiz worked there?", "How long has Ana Ruiz worked there?"),
    ],
)
def test_questions_are_rewritten_to_name_one_person(question, expected):
    out = jobresqa._personalise(question, "Ana Ruiz")
    assert "Ana Ruiz" in out
    assert out.startswith(expected) or out == expected


def test_history_marks_unrecorded_tool_counts_rather_than_zero():
    rows = {r["case"]: r for r in history.summary()}
    assert rows["python-skill"]["historical runs"] == 5
    assert rows["python-skill"]["passed"] == 3  # failed in runs 3 and 4
    assert rows["portuguese-language"]["tool-call samples"] == 2  # only runs 1 and 2 are itemised
    assert rows["portuguese-language"]["tool calls (recorded)"] == "1–1"
    # runs 3-5 are summarised in NOTES.md: unknown tool counts are None, never a made-up 0
    unrecorded = [r for r in history.records() if r["run"] >= 3 and r["case_id"] != "python-skill"]
    assert unrecorded and all(r["tool_calls"] is None for r in unrecorded)
