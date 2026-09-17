"""ChromaDB-backed candidate index: structured fields as metadata, CV text as the embedding.

`search()` is the one entry point for hybrid retrieval: metadata filters narrow the set,
cosine similarity ranks what is left. The agent, the CLI and a future MCP server all go
through the three public methods: search, get, find_by_name.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb

from ..config import CHROMA_DIR
from ..generate.schema import Candidate
from .embeddings import Embedder, FastEmbedder

COLLECTION = "candidates"


def flag(prefix: str, value: str) -> str:
    """Metadata key for a set-valued field: lang_spanish, skill_python."""
    return f"{prefix}_{re.sub(r'[^a-z0-9]+', '_', value.lower()).strip('_')}"


def skill_variants(skill: str) -> set[str]:
    """'Python 3.11' → {python 3.11, python}; 'Apache Airflow' → {apache airflow, apache, airflow}.
    Flags for every variant let a filter like skills=["Python"] match however the CV spelled it."""
    base = re.sub(r"\s*\(.*?\)", "", skill).strip()  # drop "(Django)"-style qualifiers
    no_version = re.sub(r"\s+v?\d[\w.]*$", "", base).strip()
    words = {w for w in re.split(r"[\s/,+()]+", skill) if len(w) > 2 and not w[0].isdigit()}
    return {v.lower() for v in {skill, base, no_version, *words} if v}


@dataclass
class Filters:
    seniority: list[str] = field(default_factory=list)
    role_family: list[str] = field(default_factory=list)
    country: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)  # all must be present
    skills: list[str] = field(default_factory=list)  # all must be present
    min_years: int | None = None

    def to_where(self) -> dict[str, Any] | None:
        clauses: list[dict[str, Any]] = []
        for key, values in (
            ("seniority", self.seniority),
            ("role_family", self.role_family),
            ("country", self.country),
        ):
            if values:
                clauses.append({key: {"$in": [v.lower() for v in values]}})
        clauses += [{flag("lang", v): True} for v in self.languages]
        clauses += [{flag("skill", v): True} for v in self.skills]
        if self.min_years is not None:
            clauses.append({"years_experience": {"$gte": self.min_years}})
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def is_empty(self) -> bool:
        return self.to_where() is None


@dataclass
class Hit:
    id: str
    name: str
    headline: str
    seniority: str
    role_family: str
    country: str
    years_experience: int
    languages: str
    score: float | None  # cosine similarity in [0, 1]; None for filter-only lookups

    @classmethod
    def from_meta(cls, meta: dict[str, Any], distance: float | None) -> Hit:
        return cls(
            id=meta["id"],
            name=meta["name"],
            headline=meta["headline"],
            seniority=meta["seniority"],
            role_family=meta["role_family"],
            country=meta["country_display"],
            years_experience=meta["years_experience"],
            languages=meta["languages"],
            score=None if distance is None else round(1.0 - distance, 3),
        )


def metadata_for(c: Candidate) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "id": c.id,
        "name": c.full_name,
        "headline": c.headline,
        "seniority": c.seniority,
        "role_family": c.role_family,
        "country": c.country.lower(),
        "country_display": c.country,
        "city": c.city,
        "years_experience": c.years_experience,
        "languages": ", ".join(f"{lang.name} ({lang.level})" for lang in c.languages),
        "skills": ", ".join(c.skills),
        "json": c.model_dump_json(),
    }
    meta.update({flag("lang", lang.name): True for lang in c.languages})
    meta.update({flag("skill", v): True for s in c.skills for v in skill_variants(s)})
    return meta


class CandidateIndex:
    def __init__(self, path: Path = CHROMA_DIR, embedder: Embedder | None = None):
        self.client = chromadb.PersistentClient(path=str(path))
        self._embedder = embedder
        self.collection = self.client.get_or_create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine"}, embedding_function=None
        )

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = FastEmbedder()
        return self._embedder

    # ---- build -------------------------------------------------------------------------

    def rebuild(self, candidates: list[Candidate]) -> int:
        self.client.delete_collection(COLLECTION)
        self.collection = self.client.create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine"}, embedding_function=None
        )
        docs = [c.search_text() for c in candidates]
        self.collection.add(
            ids=[c.id for c in candidates],
            documents=docs,
            embeddings=self.embedder.embed(docs),
            metadatas=[metadata_for(c) for c in candidates],
        )
        return len(candidates)

    def count(self) -> int:
        return self.collection.count()

    # ---- read --------------------------------------------------------------------------

    def search(self, query: str | None, filters: Filters | None = None, k: int = 5) -> list[Hit]:
        """Hybrid search. With a query: filter then rank by similarity. Without: filter only."""
        where = (filters or Filters()).to_where()
        if not query:
            res = self.collection.get(where=where, include=["metadatas"])
            hits = [Hit.from_meta(m, None) for m in res["metadatas"]]
            return sorted(hits, key=lambda h: (-h.years_experience, h.name))[:k]
        n = min(k, max(self.count(), 1))
        res = self.collection.query(
            query_embeddings=self.embedder.embed([query]),
            n_results=n,
            where=where,
            include=["metadatas", "distances"],
        )
        return [
            Hit.from_meta(m, d)
            for m, d in zip(res["metadatas"][0], res["distances"][0], strict=True)
        ]

    def get(self, candidate_id: str) -> Candidate | None:
        res = self.collection.get(ids=[candidate_id], include=["metadatas"])
        if not res["metadatas"]:
            return None
        return Candidate.model_validate_json(res["metadatas"][0]["json"])

    def find_by_name(self, name: str, cutoff: float = 0.5) -> list[Hit]:
        """Fuzzy name lookup; the collection is small enough to scan in memory."""
        res = self.collection.get(include=["metadatas"])
        needle = name.lower().strip()
        scored = []
        for meta in res["metadatas"]:
            full = meta["name"].lower()
            ratio = difflib.SequenceMatcher(None, needle, full).ratio()
            if needle in full or any(part == needle for part in full.split()):
                ratio = 1.0
            if ratio >= cutoff:
                scored.append((ratio, meta))
        scored.sort(key=lambda x: -x[0])
        return [Hit.from_meta(m, 1.0 - r) for r, m in scored[:5]]
