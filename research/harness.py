"""The two arms under test and the record they produce.

Arm A is the shipped `Agent`: the model sees three tools and never the dataset.
Arm B is a baseline with no tools at all: every candidate's `search_text()` is concatenated
into one system prompt and the model answers in a single call.

Both arms are judged by `check()` from `evals/run.py`, so the verdicts are comparable with the
runs already recorded in NOTES.md.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from cv_screener.agent.loop import Agent, Answer
from cv_screener.generate.schema import Candidate
from cv_screener.llm import LLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
from run import check  # noqa: E402  - evals/ is a script directory, not a package

CASES_FILE = ROOT / "evals" / "cases.json"

#: Temperature the agent uses for every answering call (`agent/loop.py`), mirrored in arm B.
TEMPERATURE = 0.2

#: Same rules as the agent's system prompt minus everything about tools, plus the corpus.
BASELINE_SYSTEM = """You are a recruiting assistant answering questions about a dataset of \
candidate CVs.

The complete dataset is included below; it is the only information you have.

Rules:
1. Answer only from the CVs below; never guess or recall candidates from memory.
2. Name only candidates whose CV appears below, and cite what in their profile supports the
   answer (skills, role, years, languages with level).
3. If no CV below matches, say clearly that no one in the dataset matches. Do not invent or stretch.
4. "Senior" in a question means senior or above.
5. Be concise: a short answer with a bullet per candidate. Refer to people by name; never paste
   ids, JSON or citation markers into the answer.

=== CVs ===
{corpus}
=== end of CVs ==="""


@dataclass
class RunRecord:
    """One (arm, case, repetition). Tokens are summed over every LLM call the arm made."""

    arm: str
    case_id: str
    rep: int
    passed: bool
    reasons: list[str] = field(default_factory=list)
    llm_calls: int = 0
    tool_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated: bool = False  # provider omitted `usage` on at least one call -> chars/4 fallback
    requests: int = 0  # HTTP attempts incl. retries, for the budget
    seconds: float = 0.0
    model: str | None = None
    answer: str = ""
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def cases() -> list[dict[str, Any]]:
    return json.loads(CASES_FILE.read_text())


def corpus_text(candidates: list[Candidate]) -> str:
    """Every CV in one block, the way arm B has to send it on every question."""
    return "\n\n---\n\n".join(c.search_text() for c in candidates)


def baseline_prompt(corpus: str) -> str:
    return BASELINE_SYSTEM.format(corpus=corpus)


def _record_from(llm: LLM, base: RunRecord) -> RunRecord:
    u = llm.usage
    base.llm_calls = u.calls
    base.prompt_tokens = u.prompt_tokens
    base.completion_tokens = u.completion_tokens
    base.total_tokens = u.total_tokens
    base.estimated = u.estimated
    base.requests = u.requests
    return base


def run_tools_arm(agent: Agent, case: dict[str, Any], rep: int, names: dict[str, str]) -> RunRecord:
    """Arm A: the shipped agent. Cost is the sum over every call in its tool loop."""
    agent.llm.reset_usage()
    rec = RunRecord(arm="tools", case_id=case["id"], rep=rep, passed=False)
    t0 = time.time()
    try:
        answer = agent.ask(case["question"])
        rec.reasons = check(case, answer, names)
        rec.passed = not rec.reasons
        rec.tool_calls = len(answer.trace)
        rec.answer = answer.text
        rec.model = answer.model
        rec.extra = {"tools": [t.name for t in answer.trace], "grounded": answer.grounded}
    except Exception as exc:  # noqa: BLE001 - a crash is a failed case, not a crashed experiment
        rec.error = f"{type(exc).__name__}: {str(exc)[:300]}"
        rec.reasons = [f"error: {rec.error}"]
        rec.model = agent.llm.last_model
    rec.seconds = round(time.time() - t0, 2)
    return _record_from(agent.llm, rec)


def run_prompt_arm(
    llm: LLM,
    corpus: str,
    case: dict[str, Any],
    rep: int,
    names: dict[str, str],
) -> RunRecord:
    """Arm B: one call, whole dataset in the system prompt, no tools."""
    llm.reset_usage()
    rec = RunRecord(arm="prompt", case_id=case["id"], rep=rep, passed=False)
    t0 = time.time()
    try:
        msg = llm.chat(
            [
                {"role": "system", "content": baseline_prompt(corpus)},
                {"role": "user", "content": case["question"]},
            ],
            temperature=TEMPERATURE,
        )
        text = (msg.content or "").strip()
        # Arm B sees every candidate, so nothing it can name is un-retrieved: grounding is
        # true by construction and only the include/exclude/no-match checks carry information.
        answer = Answer(text=text, trace=[], model=llm.last_model, grounded=True)
        rec.reasons = [r for r in check(case, answer, names) if r != "no tool calls"]
        rec.passed = not rec.reasons
        rec.answer = text
        rec.model = llm.last_model
    except Exception as exc:  # noqa: BLE001
        rec.error = f"{type(exc).__name__}: {str(exc)[:300]}"
        rec.reasons = [f"error: {rec.error}"]
        rec.model = llm.last_model
    rec.seconds = round(time.time() - t0, 2)
    return _record_from(llm, rec)
