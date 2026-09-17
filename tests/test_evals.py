"""The eval checker itself is pure and must be right before we trust its verdicts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
from run import check, mentions  # noqa: E402

from cv_screener.agent.loop import Answer, ToolTrace  # noqa: E402

NAMES = {"p01": "Ana Ruiz", "p02": "Bob Lee", "p03": "Cy Nowak"}


def answer(text: str, tools: int = 1, ungrounded: list[str] | None = None) -> Answer:
    trace = [ToolTrace("search_candidates", {}, 1, ["p01"])] * tools
    return Answer(text, trace, "fake", not ungrounded, ungrounded or [])


def test_mentions_accepts_last_name():
    assert mentions("Ruiz is a strong fit", "Ana Ruiz")
    assert not mentions("Nobody matches", "Ana Ruiz")


def test_include_exclude_checks():
    case = {"must_include": ["p01"], "must_exclude": ["p02"]}
    assert check(case, answer("Ana Ruiz fits."), NAMES) == []
    assert check(case, answer("Bob Lee fits."), NAMES) == [
        "missing Ana Ruiz",
        "should not mention Bob Lee",
    ]


def test_no_match_case_rejects_any_name_and_requires_tools():
    case = {"expect_no_match": True}
    assert check(case, answer("No candidate speaks Hebrew."), NAMES) == []
    assert check(case, answer("Cy Nowak speaks Hebrew."), NAMES) == [
        "named candidates on a no-match question: Cy Nowak"
    ]
    assert check(case, answer("Nobody.", tools=0), NAMES) == ["no tool calls"]


def test_ungrounded_answer_fails():
    reasons = check(
        {"must_include": ["p01"]}, answer("Ana Ruiz and Zed.", ungrounded=["Zed"]), NAMES
    )
    assert reasons == ["ungrounded names: Zed"]


def test_cases_file_is_well_formed():
    cases = json.loads((Path(__file__).resolve().parents[1] / "evals" / "cases.json").read_text())
    assert len(cases) >= 5
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids))
    for c in cases:
        assert c["question"]
        assert c.get("expect_no_match") or c.get("must_include")
