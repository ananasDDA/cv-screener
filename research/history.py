"""The five agent runs already recorded in NOTES.md, transcribed as data.

NOTES.md is the source of truth and is not modified. Runs 1 and 2 are printed in full, so
pass/fail and the tool-call count are known per case. Runs 3–5 are summarised there: run 3 and
run 4 give the failing case with its tool-call count and an 8/9 total, run 5 only "9/9 passed".
`tool_calls = None` therefore means "not recorded", never zero.
"""

from __future__ import annotations

from typing import Any

CASE_IDS = [
    "python-skill",
    "spanish-language",
    "senior-ml-fit",
    "summarize-katie",
    "portuguese-language",
    "years-filter",
    "no-match-hebrew",
    "no-match-cobol",
    "no-match-unknown-person",
]

RUN1 = {
    "python-skill": (True, 1, 24.9),
    "spanish-language": (True, 1, 11.2),
    "senior-ml-fit": (True, 3, 38.2),
    "summarize-katie": (True, 2, 13.9),
    "portuguese-language": (True, 1, 5.0),
    "years-filter": (True, 1, 4.6),
    "no-match-hebrew": (True, 1, 9.5),
    "no-match-cobol": (True, 1, 3.0),
    "no-match-unknown-person": (True, 1, 10.6),
}

RUN2 = {
    "python-skill": (True, 1, 12.5),
    "spanish-language": (True, 1, 6.0),
    "senior-ml-fit": (True, 1, 8.7),
    "summarize-katie": (True, 2, 24.0),
    "portuguese-language": (True, 1, 6.1),
    "years-filter": (True, 1, 3.7),
    "no-match-hebrew": (True, 1, 4.2),
    "no-match-cobol": (True, 1, 2.6),
    "no-match-unknown-person": (True, 2, 6.2),
}

#: Runs 3–5: only the failing case is itemised in NOTES.md; the rest passed (8/9, 8/9, 9/9).
PARTIAL = {
    3: {"python-skill": (False, 6, None)},
    4: {"python-skill": (False, 1, None)},
    5: {},
}


def records() -> list[dict[str, Any]]:
    """One record per (historical run, case), in the same shape as `harness.RunRecord`."""
    out: list[dict[str, Any]] = []
    for run_no, table in ((1, RUN1), (2, RUN2)):
        for case_id, (passed, tools, seconds) in table.items():
            out.append(_row(run_no, case_id, passed, tools, seconds))
    for run_no, failures in PARTIAL.items():
        for case_id in CASE_IDS:
            passed, tools, seconds = failures.get(case_id, (True, None, None))
            out.append(_row(run_no, case_id, passed, tools, seconds))
    return out


def _row(run_no: int, case_id: str, passed: bool, tools: int | None, seconds: float | None) -> dict:
    return {
        "source": f"NOTES.md run {run_no}",
        "run": run_no,
        "case_id": case_id,
        "passed": passed,
        "tool_calls": tools,
        "seconds": seconds,
    }


def summary() -> list[dict[str, Any]]:
    """Per-case pass count over the five historical runs plus the recorded tool-call range."""
    rows = []
    for case_id in CASE_IDS:
        recs = [r for r in records() if r["case_id"] == case_id]
        tools = [r["tool_calls"] for r in recs if r["tool_calls"] is not None]
        rows.append(
            {
                "case": case_id,
                "historical runs": len(recs),
                "passed": sum(r["passed"] for r in recs),
                "tool calls (recorded)": f"{min(tools)}–{max(tools)}" if tools else "not recorded",
                "tool-call samples": len(tools),
            }
        )
    return rows
