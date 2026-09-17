"""MCP server exposing the candidate index to any MCP client (Claude Desktop, Claude Code, ...).

Runs locally over stdio: nothing leaves the machine except what the client shows in its own
chat. No LLM key is needed here, the client's model is the agent. The three tools are the same
boundary the built-in agent uses (see agent/tools.py).
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mcp.server.apps import Apps
from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from . import library
from .config import CANDIDATES_DIR, CHROMA_DIR, PHOTOS_DIR, USER_CANDIDATES_DIR
from .generate.photos import photo_data_uri
from .generate.themes import theme_for
from .index.store import CandidateIndex, Filters
from .library import NewCandidate
from .widgets import URIS
from .widgets import load as load_widget

INSTRUCTIONS = """cv-screener is a database of job candidates (résumés / CVs) with hybrid search.

WHEN TO USE: any question about candidates, applicants, résumés, CVs, hiring, or "who has / who
speaks / who knows / who would fit" is about this database. Search immediately; do not ask the
user to confirm or to narrow the criteria first, and do not answer from memory or general knowledge.

HOW: search_candidates with `query` for meaning (role, domain, seniority in words) and the filters
for exact facts (languages, skills, seniority, years); filters combine with AND. Use k=10 or more
for "who / which candidates" questions and list everyone the search returns. Hits already carry
headline, years, languages with CEFR levels and skills; call get_candidate only for one person's
full profile, and find_by_name when a name is given.

LIBRARY: when the user shares a résumé (file or pasted text) and asks to add, save or import it,
extract the facts and call add_candidate once per person. Never invent data: leave unknown
optional fields empty. list_candidates shows the whole collection as a browsable deck;
remove_candidate deletes only user-added entries (ids starting with "u").

FRESHNESS: call the tools for every candidate question, even if similar results appeared earlier
in the conversation. The collection changes, and the tool widgets are how the user sees results.

GROUNDING: name only candidates returned by the tools. If a search returns nothing, say that no
candidate in the dataset matches."""


def ensure_index(
    index: CandidateIndex,
    candidates_dir: Path = CANDIDATES_DIR,
    user_dir: Path = USER_CANDIDATES_DIR,
) -> None:
    """Build the index on first run so `uvx cv-screener mcp` works without a setup step."""
    if index.is_current():
        return
    from .generate.generator import load_library

    cands = load_library(candidates_dir, user_dir)
    if not cands:
        raise RuntimeError(f"no candidates in {candidates_dir}; run `cv generate` first")
    print(f"cv-screener: building index for {len(cands)} candidates...", file=sys.stderr)
    index.rebuild(cands)


def build_server(
    index: CandidateIndex | None = None,
    photos_dir: Path = PHOTOS_DIR,
    user_dir: Path = USER_CANDIDATES_DIR,
) -> MCPServer:
    """Tools are bound to MCP Apps widgets (results table, profile card). Hosts that do not
    render apps just get the JSON; nothing else changes."""
    index = index or CandidateIndex(CHROMA_DIR)
    ensure_index(index, user_dir=user_dir)
    apps = Apps()
    plain_tools: list[tuple[Any, dict[str, Any]]] = []

    def server_tool(**kwargs: Any):
        def decorator(fn):
            plain_tools.append((fn, kwargs))
            return fn

        return decorator

    apps.add_html_resource(URIS["deck"], load_widget("deck"), name="Candidate deck")
    apps.add_html_resource(URIS["results"], load_widget("results"), name="Candidate results")
    apps.add_html_resource(URIS["card"], load_widget("card"), name="Candidate profile")

    read_only = ToolAnnotations(read_only_hint=True, open_world_hint=False)

    @apps.tool(
        resource_uri=URIS["results"],
        annotations=read_only,
        title="Search candidates",
        description=(
            "Search the candidate résumé database (cv-screener). Use it for any question like "
            "'who speaks X', 'who has experience with Y', 'best fit for role Z'. `query` ranks by "
            "meaning; filters are exact, combined with AND. Omit `query` (do not pass '*') when the "
            "question is only about exact facts; language filters then rank by proficiency. Hits include headline, seniority, years, "
            "languages with levels, skills and a score. Keywords: candidates, résumé, CV, hiring, "
            "кандидаты, резюме, кто из кандидатов."
        ),
    )
    def search_candidates(
        query: str | None = None,
        seniority: list[str] | None = None,
        role_family: list[str] | None = None,
        country: list[str] | None = None,
        languages: list[str] | None = None,
        skills: list[str] | None = None,
        min_years: int | None = None,
        k: int = 10,
    ) -> list[dict[str, Any]]:
        filters = Filters(
            seniority=seniority or [],
            role_family=role_family or [],
            country=country or [],
            languages=languages or [],
            skills=skills or [],
            min_years=min_years,
        )
        return [asdict(h) for h in index.search(query or None, filters, k=min(max(k, 1), 20))]

    @apps.tool(
        resource_uri=URIS["card"],
        annotations=read_only,
        title="Get candidate profile",
        description="Full profile of one candidate in the résumé database, by id from a search or find_by_name result.",
    )
    def get_candidate(candidate_id: str) -> dict[str, Any]:
        c = index.get(candidate_id)
        if c is None:
            return {"error": f"no candidate with id {candidate_id!r}"}
        return c.model_dump(exclude={"photo_prompt", "template", "gender", "age"})

    @apps.tool(
        resource_uri=URIS["results"],
        annotations=read_only,
        title="Find candidate by name",
        description="Find candidates in the résumé database by full or partial name; tolerant to misspellings.",
    )
    def find_by_name(name: str) -> list[dict[str, Any]]:
        return [asdict(h) for h in index.find_by_name(name)]

    @apps.tool(
        resource_uri=URIS["deck"],
        annotations=read_only,
        title="Browse the collection",
        description=(
            "List every candidate in the résumé database as a browsable deck. Use it when the user "
            "wants to see, browse or review the whole collection rather than search it. Keywords: "
            "show all résumés, all candidates, покажи все резюме, все кандидаты."
        ),
    )
    def list_candidates() -> dict[str, Any]:
        """The model gets a summary only; the deck widget pulls full data through get_deck.
        Dumping every profile into the context would let the model answer later questions
        from memory instead of searching, and would not scale past a few dozen résumés."""
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

    @apps.tool(
        resource_uri=URIS["deck"],
        visibility=["app"],  # widget-only: keeps the whole collection out of the model's context
        description="Full deck data for the collection widget.",
    )
    def get_deck() -> list[dict[str, Any]]:
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

    @apps.tool(
        resource_uri=URIS["card"],
        annotations=ToolAnnotations(
            read_only_hint=False, destructive_hint=False, open_world_hint=False
        ),
        title="Add a résumé to the collection",
        description=(
            "Add one candidate to the local résumé database from a résumé the user shared. Extract "
            "facts only; leave unknown optional fields empty. Stored on this machine, never uploaded."
        ),
    )
    def add_candidate(profile: NewCandidate) -> dict[str, Any]:
        candidate = library.save_new(profile, user_dir)
        index.add(candidate)
        return candidate.model_dump(exclude={"photo_prompt", "template", "gender", "age"})

    @server_tool(
        annotations=ToolAnnotations(
            read_only_hint=False, destructive_hint=True, open_world_hint=False
        ),
        title="Remove a user-added résumé",
        description="Remove a candidate the user added earlier (id starts with 'u'). The bundled dataset cannot be removed.",
    )
    def remove_candidate(candidate_id: str) -> dict[str, Any]:
        if not library.delete(candidate_id, user_dir):
            return {
                "removed": False,
                "error": "only user-added candidates (ids like u01-...) can be removed",
            }
        index.remove(candidate_id)
        return {"removed": True, "id": candidate_id}

    @apps.tool(
        resource_uri=URIS["card"],
        visibility=[
            "app"
        ],  # called by the widgets only; keeps ~100 KB of base64 out of the model's context
        description="Photo of a candidate as a data URI (for the widgets).",
    )
    def get_candidate_photo(candidate_id: str) -> dict[str, str]:
        c = index.get(candidate_id)
        if c is None:
            return {"error": f"no candidate with id {candidate_id!r}"}
        return {"data_uri": photo_data_uri(c.id, c.full_name, photos_dir)}

    server = MCPServer(
        "cv-screener",
        title="CV Screener",
        description=(
            "Local database of job candidates (résumés / CVs): search by skills, languages, "
            "seniority or meaning, browse the collection, add résumés. Use it for any question about "
            "candidates or hiring. Keywords: candidates, résumé, CV, applicants, hiring, recruiting, "
            "кандидаты, резюме, найм, вакансия."
        ),
        instructions=INSTRUCTIONS,
        version="0.1.0",
        extensions=[apps],
    )

    for fn, kwargs in plain_tools:
        server.tool(**kwargs)(fn)

    @server.prompt(
        name="screen_candidates",
        title="Screen candidates",
        description="Find and compare candidates for a role using the CV database.",
    )
    def screen_candidates(role: str, must_have: str = "") -> str:
        extra = f" Must-have facts (use exact filters): {must_have}." if must_have else ""
        return (
            f"Using the cv-screener tools, find the best candidates for this role: {role}.{extra} "
            "Search by meaning first, apply exact filters for hard requirements, then rank the "
            "results and explain each candidate's fit in two lines, citing their profile."
        )

    return server


def main() -> None:
    build_server().run(transport="stdio")
