"""MCP server exposing the candidate index to any MCP client (Claude Desktop, Claude Code, ...).

Runs locally over stdio: nothing leaves the machine except what the client shows in its own
chat. No LLM key is needed here, the client's model is the agent. The three tools are the same
boundary the built-in agent uses (see agent/tools.py).
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from .config import CANDIDATES_DIR, CHROMA_DIR
from .index.store import CandidateIndex, Filters

INSTRUCTIONS = """Candidate CV index with hybrid search. Use search_candidates with `query` for
meaning (role, domain) and the filters for exact facts (languages, skills, seniority, years);
they combine with AND. Results carry headline, years, languages with CEFR levels and skills, so
list everyone a search returns. Use get_candidate only for a full profile of one person.
If a search returns nothing, say that no candidate matches instead of guessing."""


def ensure_index(index: CandidateIndex, candidates_dir: Path = CANDIDATES_DIR) -> None:
    """Build the index on first run so `uvx cv-screener mcp` works without a setup step."""
    if index.count() > 0:
        return
    from .generate.generator import load_all

    cands = load_all(candidates_dir)
    if not cands:
        raise RuntimeError(f"no candidates in {candidates_dir}; run `cv generate` first")
    print(f"cv-screener: building index for {len(cands)} candidates...", file=sys.stderr)
    index.rebuild(cands)


def build_server(index: CandidateIndex | None = None) -> MCPServer:
    index = index or CandidateIndex(CHROMA_DIR)
    ensure_index(index)
    server = MCPServer("cv-screener", instructions=INSTRUCTIONS, version="0.1.0")

    @server.tool(
        description=(
            "Search the CV index. `query` ranks by meaning; filters are exact, combined with AND. "
            "Hits include headline, seniority, years, languages with levels, skills and a score."
        )
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

    @server.tool(
        description="Full profile of one candidate by id (from a search or find_by_name result)."
    )
    def get_candidate(candidate_id: str) -> dict[str, Any]:
        c = index.get(candidate_id)
        if c is None:
            return {"error": f"no candidate with id {candidate_id!r}"}
        return c.model_dump(exclude={"photo_prompt", "template", "gender", "age"})

    @server.tool(description="Find candidates by full or partial name; tolerant to misspellings.")
    def find_by_name(name: str) -> list[dict[str, Any]]:
        return [asdict(h) for h in index.find_by_name(name)]

    return server


def main() -> None:
    build_server().run(transport="stdio")
