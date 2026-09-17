"""The MCP server is a thin layer over the index; exercise it in-process, no subprocess, no key."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cv_screener.index.embeddings import HashEmbedder
from cv_screener.index.store import CandidateIndex
from cv_screener.mcp_server import build_server, ensure_index


@pytest.fixture
def server(tmp_path: Path, sample_candidates):
    idx = CandidateIndex(path=tmp_path / "chroma", embedder=HashEmbedder())
    idx.rebuild(sample_candidates)
    return build_server(idx)


async def test_lists_the_three_tools(server):
    names = {t.name for t in await server.list_tools()}
    assert names == {"search_candidates", "get_candidate", "find_by_name"}


async def test_search_with_filter_returns_hits(server):
    result = await server.call_tool("search_candidates", {"languages": ["Spanish"], "k": 10})
    hits = _payload(result)
    assert {h["id"] for h in hits} == {"p01-ana-ml", "p02-bob-qa"}
    assert "skills" in hits[0]


async def test_get_and_find(server):
    profile = _payload(await server.call_tool("get_candidate", {"candidate_id": "p03-cy-fe"}))
    assert profile["full_name"] == "Cy Nowak" and "photo_prompt" not in profile
    assert "error" in _payload(await server.call_tool("get_candidate", {"candidate_id": "nope"}))
    found = _payload(await server.call_tool("find_by_name", {"name": "Nowack"}))
    assert found[0]["id"] == "p03-cy-fe"


def test_ensure_index_builds_from_json_when_empty(tmp_path: Path, sample_candidates):
    cdir = tmp_path / "candidates"
    cdir.mkdir()
    for c in sample_candidates:
        (cdir / f"{c.id}.json").write_text(c.model_dump_json())
    idx = CandidateIndex(path=tmp_path / "chroma", embedder=HashEmbedder())
    assert idx.count() == 0
    ensure_index(idx, cdir)
    assert idx.count() == 3
    with pytest.raises(RuntimeError):
        ensure_index(
            CandidateIndex(path=tmp_path / "empty", embedder=HashEmbedder()), tmp_path / "none"
        )


def _payload(result):
    """call_tool returns a CallToolResult; lists arrive as structured_content={"result": [...]}."""
    data = result.structured_content
    if data is None:
        return json.loads("".join(getattr(b, "text", "") for b in result.content))
    return data["result"] if set(data) == {"result"} else data
