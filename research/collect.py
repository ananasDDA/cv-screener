"""Collectors: the only code in this package that spends API requests.

Run from the repo root, one part at a time:

    uv run python -m research.collect part1 --reps 3
    uv run python -m research.collect part4 --reps 2
    uv run python -m research.collect part3

Every finished run is appended to its JSON cache immediately, so an interrupted collection
resumes where it stopped and nothing is paid for twice. `--force` clears the cache first.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

from cv_screener.agent.loop import Agent
from cv_screener.config import Settings, settings
from cv_screener.generate.generator import load_all
from cv_screener.index.store import CandidateIndex
from cv_screener.llm import LLM

from . import cache, harness

#: Free models with tool-calling support (openrouter.ai/api/v1/models, supported_parameters).
#: The first four are the project's own chain from `.env`; `qwen3.8-27b` is the strongest
#: remaining free tool-caller. A sixth, `thinkingmachines/inkling:free`, was dropped when the
#: account's free-model day ran out — see the notebook.
LEADERBOARD_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "qwen/qwen3.8-27b:free",
]
DROPPED_MODELS = ["thinkingmachines/inkling:free"]

PAUSE_SECONDS = 1.5  # be gentle with the shared free-tier key


def _guard(label: str, planned: int) -> None:
    left = cache.remaining()
    print(
        f"[budget] spent {cache.spent()} / {cache.BUDGET_LIMIT}, {left} left; {label}: "
        f"~{planned} runs planned"
    )
    if left <= 0:
        raise SystemExit("budget exhausted; nothing collected")


def _append(name: str, records: list[dict[str, Any]], record: dict[str, Any], label: str) -> None:
    records.append(record)
    cache.save(name, records)
    cache.charge(record.get("requests", 0), label)


#: Three 429s in a row mean the account is throttled, not that the next call will work. Stop
#: rather than retry into a wall: the key is shared and tomorrow's quota has to survive.
QUOTA_STRIKES = 3


class QuotaWatch:
    """Aborts a collection once the provider is clearly refusing us."""

    def __init__(self) -> None:
        self.strikes = 0

    def hit(self, record: dict[str, Any]) -> bool:
        self.strikes = self.strikes + 1 if cache.is_rate_limited(record) else 0
        if self.strikes >= QUOTA_STRIKES:
            print(
                f"[quota] {self.strikes} consecutive 429s: stopping. The free-model day resets at "
                f"00:00 UTC; re-run this command afterwards rather than retrying now."
            )
            return True
        return False


# ---- part 1: tools vs everything in the prompt --------------------------------------------


def collect_part1(
    reps: int = 3, force: bool = False, arms: tuple[str, ...] = ("tools", "prompt")
) -> list[dict[str, Any]]:
    """Reps 1–3 of both arms are the A/B comparison. Extra reps of `tools` alone are for the
    stability section and are deliberately excluded from the arm comparison in the notebook."""
    name = "part1_runs"
    records = [] if force else cache.drop_quota_failures(name, cache.load(name))
    cases = harness.cases()
    specs = [(arm, c["id"], rep) for rep in range(1, reps + 1) for arm in arms for c in cases]
    todo = cache.missing(records, specs, ("arm", "case_id", "rep"))
    _guard("part1", len(todo))
    if not todo:
        return records

    index = CandidateIndex()
    if index.count() == 0:
        raise SystemExit("index is empty; run `uv run cv index` first")
    names = index.names()
    by_id = {c["id"]: c for c in cases}
    llm = LLM()
    agent = Agent(llm, index)
    corpus = harness.corpus_text(load_all())
    quota = QuotaWatch()

    for arm, case_id, rep in todo:
        if cache.remaining() <= 0:
            print("[budget] stopping: limit reached")
            break
        case = by_id[case_id]
        if arm == "tools":
            rec = harness.run_tools_arm(agent, case, rep, names)
        else:
            rec = harness.run_prompt_arm(llm, corpus, case, rep, names)
        print(
            f"  {arm:7s} {case_id:26s} rep{rep} "
            f"{'PASS' if rec.passed else 'FAIL'} {rec.total_tokens:7d} tok {rec.seconds:6.1f}s "
            f"{rec.model or '-'}"
        )
        _append(name, records, rec.as_dict(), "part1")
        if quota.hit(rec.as_dict()):
            break
        time.sleep(PAUSE_SECONDS)
    return cache.drop_quota_failures(name, records)


# ---- part 4: leaderboard of free models ---------------------------------------------------


def collect_part4(
    reps: int = 2, models: list[str] | None = None, force: bool = False
) -> list[dict[str, Any]]:
    name = "part4_runs"
    records = [] if force else cache.drop_quota_failures(name, cache.load(name))
    models = models or LEADERBOARD_MODELS
    cases = harness.cases()
    specs = [(m, c["id"], rep) for m in models for rep in range(1, reps + 1) for c in cases]
    todo = cache.missing(records, specs, ("requested_model", "case_id", "rep"))
    _guard("part4", len(todo))
    if not todo:
        return records

    index = CandidateIndex()
    names = index.names()
    by_id = {c["id"]: c for c in cases}
    key = settings().api_key
    quota = QuotaWatch()

    for model, case_id, rep in todo:
        if cache.remaining() <= 0:
            print("[budget] stopping: limit reached")
            break
        # One model, no fallback: a failure is attributed to this model, never hidden.
        llm = LLM(Settings(api_key=key, model=model, fallback_models=()))
        rec = harness.run_tools_arm(Agent(llm, index), by_id[case_id], rep, names)
        row = rec.as_dict() | {"requested_model": model}
        print(
            f"  {model:50s} {case_id:26s} rep{rep} "
            f"{'PASS' if rec.passed else 'FAIL'} {rec.seconds:6.1f}s {rec.error or ''}"[:160]
        )
        _append(name, records, row, "part4")
        if quota.hit(row):
            break
        time.sleep(PAUSE_SECONDS * (3 if rec.error else 1))
    return cache.drop_quota_failures(name, records)


# ---- part 3: external benchmark -----------------------------------------------------------


def collect_part3(force: bool = False) -> list[dict[str, Any]]:
    from . import jobresqa

    return jobresqa.collect(force=force)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("part", choices=["part1", "part3", "part4", "budget"])
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--arm", choices=["tools", "prompt"], help="part1 only: collect one arm")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.part == "budget":
        print(f"spent {cache.spent()} / {cache.BUDGET_LIMIT}")
        for label, n in sorted(cache.budget_breakdown().items()):
            print(f"  {label:10s} {n}")
        return 0
    if args.part == "part1":
        arms = (args.arm,) if args.arm else ("tools", "prompt")
        collect_part1(reps=args.reps or 3, force=args.force, arms=arms)
    elif args.part == "part4":
        collect_part4(reps=args.reps or 2, force=args.force)
    else:
        collect_part3(force=args.force)
    print(f"[budget] spent {cache.spent()} / {cache.BUDGET_LIMIT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
