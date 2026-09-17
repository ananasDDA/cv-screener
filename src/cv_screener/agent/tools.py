"""Tool definitions (OpenAI function-calling schema) and their dispatch onto the index.

The LLM never sees the dataset; it sees only what these three tools return.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from ..index.store import CandidateIndex, Filters, Hit

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_candidates",
            "description": (
                "Search the CV index. Use `query` for meaning (role, domain, seniority in words) and the "
                "filters for exact facts; omit `query` when only facts are asked (language filters then rank by "
                "proficiency). Filters combine with AND. Each result includes headline, "
                "seniority, years, languages with levels and the skills list, plus a similarity score in "
                "[0,1] when a query is given (scores below ~0.6 are weak matches)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Free-text description of who you are looking for.",
                    },
                    "seniority": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["junior", "middle", "senior", "lead", "principal"],
                        },
                    },
                    "role_family": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "e.g. backend, ml, frontend, data, devops, qa, mobile, security, product, data-science",
                    },
                    "country": {"type": "array", "items": {"type": "string"}},
                    "languages": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Spoken languages the candidate must have, e.g. ['Spanish']",
                    },
                    "skills": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Technologies the candidate must list, e.g. ['Python', 'Kubernetes']",
                    },
                    "min_years": {"type": "integer", "description": "Minimum years of experience"},
                    "k": {"type": "integer", "description": "Max results, default 5, max 20"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_candidate",
            "description": "Full profile of one candidate by id (from a previous search result).",
            "parameters": {
                "type": "object",
                "properties": {"candidate_id": {"type": "string"}},
                "required": ["candidate_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_by_name",
            "description": "Find candidates by (possibly partial or misspelled) name.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
]

PROFILE_FIELDS = (
    "id",
    "full_name",
    "headline",
    "seniority",
    "role_family",
    "years_experience",
    "city",
    "country",
    "summary",
    "skills",
    "languages",
    "experience",
    "education",
    "certifications",
)


def _hits(hits: list[Hit]) -> str:
    return json.dumps([asdict(h) for h in hits], ensure_ascii=False)


def dispatch(index: CandidateIndex, name: str, args: dict[str, Any]) -> tuple[str, int]:
    """Run one tool call. Returns (json_result, number_of_candidates_returned)."""
    if name == "search_candidates":
        filters = Filters(
            seniority=args.get("seniority") or [],
            role_family=args.get("role_family") or [],
            country=args.get("country") or [],
            languages=args.get("languages") or [],
            skills=args.get("skills") or [],
            min_years=args.get("min_years"),
        )
        hits = index.search(args.get("query") or None, filters, k=min(int(args.get("k") or 5), 20))
        return _hits(hits), len(hits)
    if name == "get_candidate":
        c = index.get(args["candidate_id"])
        if c is None:
            return json.dumps({"error": f"no candidate with id {args['candidate_id']!r}"}), 0
        return c.model_dump_json(include=set(PROFILE_FIELDS)), 1
    if name == "find_by_name":
        hits = index.find_by_name(args["name"])
        return _hits(hits), len(hits)
    return json.dumps({"error": f"unknown tool {name}"}), 0
