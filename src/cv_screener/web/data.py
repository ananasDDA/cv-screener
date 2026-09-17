"""Payloads for the widgets, in exactly the shapes the MCP server returns.

`mcp_server.build_server` defines its tools as closures, so they cannot be imported; these are
the same ten-line bodies. Keeping the shapes identical is what lets `widgets/*.html` run in the
web host without a fork.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..config import PHOTOS_DIR
from ..generate.photos import photo_data_uri
from ..generate.themes import theme_for
from ..index.store import CandidateIndex, Filters

PROFILE_EXCLUDE = {"photo_prompt", "template", "gender", "age"}

# Tools a widget may call on its own. Anything else is refused by /api/tool.
WIDGET_TOOLS = (
    "search_candidates",
    "find_by_name",
    "get_candidate",
    "get_candidate_photo",
    "get_deck",
)


def search_candidates(index: CandidateIndex, args: dict[str, Any]) -> list[dict[str, Any]]:
    filters = Filters(
        seniority=args.get("seniority") or [],
        role_family=args.get("role_family") or [],
        country=args.get("country") or [],
        languages=args.get("languages") or [],
        skills=args.get("skills") or [],
        min_years=args.get("min_years"),
    )
    k = min(max(int(args.get("k") or 10), 1), 20)
    return [asdict(h) for h in index.search(args.get("query") or None, filters, k=k)]


def find_by_name(index: CandidateIndex, args: dict[str, Any]) -> list[dict[str, Any]]:
    return [asdict(h) for h in index.find_by_name(str(args.get("name") or ""))]


def get_candidate(index: CandidateIndex, args: dict[str, Any]) -> dict[str, Any]:
    candidate_id = str(args.get("candidate_id") or "")
    c = index.get(candidate_id)
    if c is None:
        return {"error": f"no candidate with id {candidate_id!r}"}
    return c.model_dump(exclude=PROFILE_EXCLUDE)


def get_candidate_photo(
    index: CandidateIndex, args: dict[str, Any], photos_dir: Path = PHOTOS_DIR
) -> dict[str, str]:
    candidate_id = str(args.get("candidate_id") or "")
    c = index.get(candidate_id)
    if c is None:
        return {"error": f"no candidate with id {candidate_id!r}"}
    return {"data_uri": photo_data_uri(c.id, c.full_name, photos_dir)}


def get_deck(index: CandidateIndex, args: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    deck = []
    for h in index.all():
        c = index.get(h.id)
        deck.append(
            {
                **asdict(h),
                "accent": theme_for(h.id).accent,
                "top_skills": c.skills[:10] if c else [],
            }
        )
    return deck


def list_summary(index: CandidateIndex) -> dict[str, Any]:
    """Counts only. Dumping every profile into the model's context would let it answer later
    questions from memory instead of searching, and would not scale past a few dozen résumés."""
    hits = index.all()
    return {
        "total": len(hits),
        "user_added": sum(h.source == "user" for h in hits),
        "by_role": dict(Counter(h.role_family for h in hits).most_common()),
        "by_seniority": dict(Counter(h.seniority for h in hits).most_common()),
        "names": [h.name for h in hits[:40]],
        "note": (
            "The user is looking at the full deck in a widget. This summary has no skills, "
            "languages or experience: call search_candidates or get_candidate to answer "
            "any question about the candidates."
        ),
    }


def call_widget_tool(
    index: CandidateIndex, name: str, args: dict[str, Any], photos_dir: Path = PHOTOS_DIR
) -> Any:
    """Dispatch a widget-initiated `tools/call`. Raises KeyError for anything not allowed."""
    if name == "search_candidates":
        return search_candidates(index, args)
    if name == "find_by_name":
        return find_by_name(index, args)
    if name == "get_candidate":
        return get_candidate(index, args)
    if name == "get_candidate_photo":
        return get_candidate_photo(index, args, photos_dir)
    if name == "get_deck":
        return get_deck(index)
    raise KeyError(name)
