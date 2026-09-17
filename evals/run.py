"""Run the agent on evals/cases.json and print pass/fail per case plus a summary.

Needs OPENROUTER_API_KEY and a built index (`cv index`). Usage:
    uv run python evals/run.py [--only case-id] [--json evals/last_run.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from cv_screener.agent.loop import Agent, Answer
from cv_screener.index.store import CandidateIndex
from cv_screener.llm import LLM

CASES = Path(__file__).with_name("cases.json")


@dataclass
class Result:
    id: str
    passed: bool
    reasons: list[str]
    tool_calls: int
    model: str | None
    seconds: float
    answer: str


def mentions(text: str, full_name: str) -> bool:
    """Full name or the last name is enough; last names are unique in this dataset."""
    return full_name in text or full_name.split()[-1] in text


def check(case: dict, answer: Answer, names: dict[str, str]) -> list[str]:
    reasons: list[str] = []
    if not answer.trace:
        reasons.append("no tool calls")
    if not answer.grounded:
        reasons.append(f"ungrounded names: {', '.join(answer.ungrounded_names)}")
    if case.get("expect_no_match"):
        named = [n for n in names.values() if mentions(answer.text, n)]
        if named:
            reasons.append(f"named candidates on a no-match question: {', '.join(named)}")
        return reasons
    for cid in case.get("must_include", []):
        if not mentions(answer.text, names[cid]):
            reasons.append(f"missing {names[cid]}")
    for cid in case.get("must_exclude", []):
        if mentions(answer.text, names[cid]):
            reasons.append(f"should not mention {names[cid]}")
    return reasons


def run(cases: list[dict], agent: Agent, names: dict[str, str]) -> list[Result]:
    results = []
    for case in cases:
        t0 = time.time()
        try:
            answer = agent.ask(case["question"])
            reasons = check(case, answer, names)
            res = Result(
                case["id"],
                not reasons,
                reasons,
                len(answer.trace),
                answer.model,
                round(time.time() - t0, 1),
                answer.text,
            )
        except Exception as exc:  # noqa: BLE001 - a crash is a failed case, not a crashed run
            res = Result(
                case["id"],
                False,
                [f"error: {type(exc).__name__}: {exc}"],
                0,
                None,
                round(time.time() - t0, 1),
                "",
            )
        results.append(res)
        mark = "PASS" if res.passed else "FAIL"
        print(
            f"[{mark}] {res.id:28s} tools={res.tool_calls} {res.seconds:5.1f}s  {res.model or '-'}"
        )
        for r in res.reasons:
            print(f"       - {r}")
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="run a single case id")
    ap.add_argument("--json", type=Path, help="write results to this file")
    args = ap.parse_args()

    cases = json.loads(CASES.read_text())
    if args.only:
        cases = [c for c in cases if c["id"] == args.only]
    index = CandidateIndex()
    if index.count() == 0:
        print("index is empty; run `uv run cv index` first", file=sys.stderr)
        return 2
    agent = Agent(LLM(), index)
    print(f"model chain: {', '.join(agent.llm.cfg.model_chain)}\n")
    results = run(cases, agent, index.names())
    passed = sum(r.passed for r in results)
    print(f"\n{passed}/{len(results)} passed")
    if args.json:
        args.json.write_text(json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False))
        print(f"details written to {args.json}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
