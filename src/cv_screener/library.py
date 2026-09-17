"""The user's private résumé library: candidates added through MCP, stored outside the repo."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field

from .config import USER_CANDIDATES_DIR
from .generate.personas import slugify
from .generate.schema import Candidate, Education, Experience, Language, Seniority


class NewCandidate(BaseModel):
    """What a client must extract from a résumé. Facts only: leave unknown optional fields empty
    rather than guessing."""

    full_name: str
    headline: str = Field(description="Professional title as written on the résumé")
    summary: str = Field(description="3-4 sentence summary, taken from or faithful to the résumé")
    seniority: Seniority
    role_family: str = Field(
        description=(
            "One of: backend, frontend, ml, data, data-science, devops, qa, mobile, security, "
            "product, design, other"
        )
    )
    years_experience: int = Field(ge=0, le=60)
    city: str = ""
    country: str = ""
    email: str = ""
    phone: str = ""
    linkedin: str = ""
    github: str | None = None
    skills: list[str] = Field(description="Technologies, tools and methods as listed")
    languages: list[Language] = Field(
        default_factory=list, description="Spoken languages with CEFR level or Native"
    )
    experience: list[Experience] = Field(default_factory=list, description="Most recent first")
    education: list[Education] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)


def next_id(full_name: str, user_dir: Path = USER_CANDIDATES_DIR) -> str:
    taken = [
        int(m.group(1)) for p in user_dir.glob("u*.json") if (m := re.match(r"u(\d+)-", p.name))
    ]
    return f"u{max(taken, default=0) + 1:02d}-{slugify(full_name) or 'candidate'}"


def save_new(new: NewCandidate, user_dir: Path = USER_CANDIDATES_DIR) -> Candidate:
    user_dir.mkdir(parents=True, exist_ok=True)
    candidate = Candidate(**new.model_dump(), id=next_id(new.full_name, user_dir), source="user")
    (user_dir / f"{candidate.id}.json").write_text(
        candidate.model_dump_json(indent=2), encoding="utf-8"
    )
    return candidate


def delete(candidate_id: str, user_dir: Path = USER_CANDIDATES_DIR) -> bool:
    """Only user-added candidates can be removed; the bundled dataset is read-only."""
    path = user_dir / f"{candidate_id}.json"
    if not re.fullmatch(r"u\d+-[a-z0-9-]+", candidate_id) or not path.exists():
        return False
    path.unlink()
    return True
