"""The agent's three tools plus the two the web host adds, without touching `agent/tools.py`.

`TOOLS` and `dispatch` below wrap the originals: the base three are delegated unchanged, and
`list_candidates` / `add_candidate` mirror the MCP server's versions. Every dispatch also returns
the payload the matching widget expects, so the transcript can render it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..agent.tools import TOOLS as BASE_TOOLS
from ..agent.tools import dispatch as base_dispatch
from ..config import USER_CANDIDATES_DIR
from ..index.store import CandidateIndex
from ..library import NewCandidate, save_new
from . import data

# tool name → widget that visualises its result
WIDGET_FOR: dict[str, str] = {
    "search_candidates": "results",
    "find_by_name": "results",
    "get_candidate": "card",
    "add_candidate": "card",
    "list_candidates": "deck",
}

EXTRA_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_candidates",
            "description": (
                "Show the user the whole collection as a browsable deck. Use it when they ask to "
                "see, browse or review everything rather than search. Returns counts and names "
                "only, never skills or experience: search_candidates or get_candidate is still "
                "required to answer any question about the candidates."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_candidate",
            "description": (
                "Add one candidate to the local résumé library from a résumé the user shared. "
                "Extract facts only; leave unknown optional fields empty and never invent data. "
                "Stored on this machine, never uploaded."
            ),
            "parameters": NewCandidate.model_json_schema(),
        },
    },
]

TOOLS: list[dict[str, Any]] = [*BASE_TOOLS, *EXTRA_TOOLS]


def dispatch(
    index: CandidateIndex,
    name: str,
    args: dict[str, Any],
    user_dir: Path = USER_CANDIDATES_DIR,
) -> tuple[str, int, Any]:
    """Run one tool call. Returns (json for the model, candidate count, widget payload)."""
    if name == "list_candidates":
        summary = data.list_summary(index)
        return json.dumps(summary, ensure_ascii=False), summary["total"], summary
    if name == "add_candidate":
        return _add(index, args, user_dir)
    result, count = base_dispatch(index, name, args)
    return result, count, _widget_payload(index, name, args, result)


def _add(index: CandidateIndex, args: dict[str, Any], user_dir: Path) -> tuple[str, int, Any]:
    # Models occasionally wrap the fields in a "profile" object, like the MCP tool signature.
    payload = args.get("profile") if isinstance(args.get("profile"), dict) else args
    try:
        profile = NewCandidate.model_validate(payload)
    except Exception as exc:  # noqa: BLE001 - the model has to see why and retry
        return json.dumps({"error": f"invalid candidate: {str(exc)[:600]}"}), 0, None
    candidate = save_new(profile, user_dir)
    index.add(candidate)
    full = candidate.model_dump(exclude=data.PROFILE_EXCLUDE)
    confirmation = {"added": True, "id": candidate.id, "full_name": candidate.full_name}
    return json.dumps(confirmation, ensure_ascii=False), 1, full


def _widget_payload(
    index: CandidateIndex, name: str, args: dict[str, Any], result: str
) -> Any | None:
    """The widgets want the MCP shapes; `get_candidate` in agent/tools.py returns a narrower
    profile, so that one is re-read instead of reusing the model's copy."""
    if name == "get_candidate":
        profile = data.get_candidate(index, args)
        return None if "error" in profile else profile
    if name in {"search_candidates", "find_by_name"}:
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return None
    return None
