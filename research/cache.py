"""JSON caches and the request budget.

Every experiment writes one JSON file of raw per-run records. A run is identified by a key
tuple; the collectors only execute the keys that are missing, so an interrupted collection
resumes and the notebook never re-spends requests.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

RESEARCH_DIR = Path(__file__).resolve().parent
RESULTS_DIR = RESEARCH_DIR / "results"
DATA_DIR = RESEARCH_DIR / "data"
BUDGET_FILE = RESULTS_DIR / "budget.json"

#: Hard ceiling agreed for the whole task, counted in HTTP requests to OpenRouter.
BUDGET_LIMIT = 750

#: The account's free-model day ran out mid-collection. Everything spent after the 00:00 UTC
#: reset is capped separately and much harder, so tomorrow's quota stays available for a demo.
POST_RESET_LIMIT = 150


def results_path(name: str) -> Path:
    return RESULTS_DIR / f"{name}.json"


def load(name: str) -> list[dict[str, Any]]:
    path = results_path(name)
    if not path.exists():
        return []
    return json.loads(path.read_text())


def save(name: str, records: Iterable[dict[str, Any]]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = results_path(name)
    path.write_text(json.dumps(list(records), indent=1, ensure_ascii=False))
    return path


def missing(records: list[dict[str, Any]], specs: list[tuple], key_fields: tuple[str, ...]) -> list:
    """The specs that have no record yet, comparing on `key_fields`."""
    have = {tuple(r[f] for f in key_fields) for r in records}
    return [s for s in specs if s not in have]


#: OpenRouter caps free-model requests per account per day and resets at 00:00 UTC. A run that
#: died on that cap says nothing about the model or the arm, so it is not a result: it is dropped
#: from the cache and re-run after the reset.
QUOTA_MARKER = "free-models-per-day"


def is_quota_failure(record: dict[str, Any]) -> bool:
    """Died on the account's daily cap: not a measurement, so it is dropped and re-run."""
    return QUOTA_MARKER in (record.get("error") or "")


def is_rate_limited(record: dict[str, Any]) -> bool:
    """Any 429. Three in a row end the collection instead of retrying into a wall."""
    error = record.get("error") or ""
    return "429" in error or "RateLimit" in error


def drop_quota_failures(name: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept = [r for r in records if not is_quota_failure(r)]
    if len(kept) != len(records):
        print(
            f"[cache] dropping {len(records) - len(kept)} run(s) killed by the daily free-tier "
            f"cap; they will be re-run"
        )
        save(name, kept)
    return kept


# ---- budget ------------------------------------------------------------------------------


def _state() -> dict[str, Any]:
    if not BUDGET_FILE.exists():
        return {"requests": 0, "by": {}}
    return json.loads(BUDGET_FILE.read_text())


def spent() -> int:
    return int(_state()["requests"])


def charge(requests: int, label: str) -> int:
    """Record `requests` HTTP attempts against the budget and return the new total."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    state = _state()
    state["requests"] += requests
    state["by"][label] = state["by"].get(label, 0) + requests
    BUDGET_FILE.write_text(json.dumps(state, indent=1))
    return state["requests"]


def budget_breakdown() -> dict[str, int]:
    return _state().get("by", {})


def mark_reset() -> int:
    """Freeze today's total as the baseline for the separate post-reset allowance."""
    state = _state()
    state.setdefault("reset_baseline", state["requests"])
    BUDGET_FILE.write_text(json.dumps(state, indent=1))
    return state["reset_baseline"]


def pre_reset_spent() -> int:
    return int(_state().get("reset_baseline", spent()))


def post_reset_spent() -> int:
    return spent() - pre_reset_spent()


def remaining() -> int:
    """Whatever is left of the overall budget, and never more than the post-reset allowance."""
    left = BUDGET_LIMIT - spent()
    if "reset_baseline" in _state():
        left = min(left, POST_RESET_LIMIT - post_reset_spent())
    return left
