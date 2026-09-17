"""Index tests use HashEmbedder: deterministic, offline, no model download."""

from __future__ import annotations

from pathlib import Path

import pytest

from cv_screener.index.embeddings import HashEmbedder
from cv_screener.index.store import CandidateIndex, Filters, flag, metadata_for, skill_variants


@pytest.fixture
def index(tmp_path: Path, sample_candidates) -> CandidateIndex:
    idx = CandidateIndex(path=tmp_path / "chroma", embedder=HashEmbedder())
    assert idx.rebuild(sample_candidates) == 3
    return idx


def test_metadata_flags_for_languages_and_skills(sample_candidates):
    meta = metadata_for(sample_candidates[0])
    assert meta["lang_spanish"] is True and meta["skill_pytorch"] is True
    assert "lang_polish" not in meta
    assert flag("skill", "C++ / CUDA") == "skill_c_cuda"


def test_skill_variants_cover_versions_and_words():
    assert "python" in skill_variants("Python 3.11")
    assert {"apache airflow", "airflow"} <= skill_variants("Apache Airflow")
    assert "django" in skill_variants("Python (Django)")
    assert "" not in skill_variants("  ")


def test_filter_only_search_matches_fields(index: CandidateIndex):
    spanish = index.search(None, Filters(languages=["Spanish"]))
    assert {h.id for h in spanish} == {"p01-ana-ml", "p02-bob-qa"}
    assert all(h.score is None for h in spanish)
    seniors = index.search(None, Filters(seniority=["senior"], country=["spain"]))
    assert [h.id for h in seniors] == ["p01-ana-ml"]
    assert index.search(None, Filters(min_years=3, skills=["Playwright"]))[0].id == "p02-bob-qa"
    assert index.search(None, Filters(languages=["Hebrew"])) == []


def test_semantic_search_ranks_overlapping_text_first(index: CandidateIndex):
    hits = index.search("deep learning models PyTorch", k=3)
    assert hits[0].id == "p01-ana-ml"
    assert hits[0].score is not None and hits[0].score > hits[-1].score


def test_hybrid_search_applies_filter_before_ranking(index: CandidateIndex):
    hits = index.search("test automation", Filters(languages=["Spanish"]), k=5)
    assert {h.id for h in hits} == {"p01-ana-ml", "p02-bob-qa"}
    assert hits[0].id == "p02-bob-qa"


def test_get_returns_full_candidate(index: CandidateIndex):
    c = index.get("p03-cy-fe")
    assert c is not None and c.full_name == "Cy Nowak" and c.skills == ["React", "TypeScript"]
    assert index.get("nope") is None


def test_find_by_name_is_fuzzy(index: CandidateIndex):
    assert index.find_by_name("Ana")[0].id == "p01-ana-ml"
    assert index.find_by_name("bob lee")[0].id == "p02-bob-qa"
    assert index.find_by_name("Nowack")[0].id == "p03-cy-fe"
    assert index.find_by_name("Zzzz Qqqq") == []


def test_rebuild_replaces_previous_contents(index: CandidateIndex, sample_candidates):
    assert index.rebuild(sample_candidates[:1]) == 1
    assert index.count() == 1


def test_junk_query_is_ignored_and_language_filter_ranks_by_level(index: CandidateIndex):
    from cv_screener.index.store import meaningful

    assert meaningful("*") is None and meaningful(" all ") is None and meaningful("") is None
    assert meaningful("ML engineer") == "ML engineer"
    hits = index.search("*", Filters(languages=["Spanish"]))
    assert [h.id for h in hits] == ["p01-ana-ml", "p02-bob-qa"]  # Native before B2
    assert all(h.score is None for h in hits)


def test_outdated_store_is_detected(index: CandidateIndex, sample_candidates):
    assert index.is_current()
    index.client.delete_collection("candidates")
    index.collection = index.client.create_collection(
        "candidates", metadata={"hnsw:space": "cosine"}
    )
    assert not index.is_current()
