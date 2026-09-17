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
    return build_server(idx, photos_dir=tmp_path, user_dir=tmp_path / "user")


async def test_lists_the_tools(server):
    names = {t.name for t in await server.list_tools()}
    assert names == {
        "search_candidates",
        "get_candidate",
        "find_by_name",
        "get_candidate_photo",
        "list_candidates",
        "add_candidate",
        "remove_candidate",
    }


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


async def test_tools_are_bound_to_widgets_and_photo_tool_is_app_only(server):
    tools = {t.name: t for t in await server.list_tools()}
    assert tools["search_candidates"].meta["ui"]["resourceUri"] == "ui://cv-screener/results.html"
    assert tools["get_candidate"].meta["ui"]["resourceUri"] == "ui://cv-screener/card.html"
    assert tools["get_candidate_photo"].meta["ui"]["visibility"] == ["app"]
    resources = {str(r.uri): r for r in await server.list_resources()}
    assert set(resources) == {
        "ui://cv-screener/results.html",
        "ui://cv-screener/card.html",
        "ui://cv-screener/deck.html",
    }
    assert all(r.mime_type == "text/html;profile=mcp-app" for r in resources.values())


async def test_widget_html_is_self_contained(server):
    from cv_screener.widgets import load

    for name in ("results", "card", "deck"):
        html = load(name)
        assert "{{" not in html and "<script src" not in html and "<link" not in html
        assert "ui/initialize" in html and "ui/notifications/tool-result" in html


async def test_photo_tool_returns_data_uri(server):
    result = await server.call_tool("get_candidate_photo", {"candidate_id": "p01-ana-ml"})
    assert _payload(result)["data_uri"].startswith("data:image/png;base64,")


async def test_tools_are_read_only_and_prompt_exists(server):
    tools = {t.name: t for t in await server.list_tools()}
    assert tools["search_candidates"].annotations.read_only_hint is True
    prompts = {p.name for p in await server.list_prompts()}
    assert "screen_candidates" in prompts
    msgs = await server.get_prompt(
        "screen_candidates", {"role": "senior ML engineer", "must_have": "Spanish"}
    )
    text = msgs.messages[0].content.text
    assert "senior ML engineer" in text and "Spanish" in text


NEW = {
    "full_name": "Dana Okoye",
    "headline": "Staff Rust Engineer",
    "summary": "Builds embedded firmware in Rust.",
    "seniority": "lead",
    "role_family": "backend",
    "years_experience": 12,
    "country": "Nigeria",
    "skills": ["Rust", "Embedded C"],
    "languages": [{"name": "English", "level": "Native"}, {"name": "Yoruba", "level": "Native"}],
}


async def test_add_list_search_and_remove_a_user_candidate(server, tmp_path: Path):
    added = _payload(await server.call_tool("add_candidate", {"profile": NEW}))
    assert added["id"] == "u01-dana-okoye" and added["source"] == "user"
    assert (tmp_path / "user" / "u01-dana-okoye.json").exists()

    listed = _payload(await server.call_tool("list_candidates", {}))
    assert len(listed) == 4 and all(p["accent"].startswith("#") for p in listed)
    assert [p["source"] for p in listed if p["id"].startswith("u")] == ["user"]

    hits = _payload(await server.call_tool("search_candidates", {"skills": ["Rust"]}))
    assert [h["id"] for h in hits] == ["u01-dana-okoye"]

    second = _payload(
        await server.call_tool("add_candidate", {"profile": {**NEW, "full_name": "Eli Tan"}})
    )
    assert second["id"] == "u02-eli-tan"

    refused = _payload(await server.call_tool("remove_candidate", {"candidate_id": "p01-ana-ml"}))
    assert refused["removed"] is False
    removed = _payload(
        await server.call_tool("remove_candidate", {"candidate_id": "u01-dana-okoye"})
    )
    assert removed["removed"] is True
    assert (
        _payload(await server.call_tool("search_candidates", {"skills": ["Embedded C"], "k": 5}))[
            0
        ]["id"]
        == "u02-eli-tan"
    )


async def test_add_candidate_rejects_invalid_profiles(server):
    bad = {**NEW, "seniority": "wizard"}
    with pytest.raises(Exception):  # noqa: B017 - the SDK wraps the pydantic error in its own type
        await server.call_tool("add_candidate", {"profile": bad})


async def test_mutating_tools_are_not_marked_read_only(server):
    tools = {t.name: t for t in await server.list_tools()}
    assert tools["add_candidate"].annotations.read_only_hint is False
    assert tools["remove_candidate"].annotations.destructive_hint is True
    assert tools["list_candidates"].meta["ui"]["resourceUri"] == "ui://cv-screener/deck.html"
