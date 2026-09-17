"""Web chat tests: offline, no key, fake LLM, HashEmbedder index in tmp_path."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient
from test_agent import FakeClient, message, tool_call

from cv_screener.config import Settings
from cv_screener.index.embeddings import HashEmbedder
from cv_screener.index.store import CandidateIndex
from cv_screener.llm import LLM
from cv_screener.web import models as model_catalogue
from cv_screener.web.app import build_llm, create_app, extract_text

CATALOGUE = [
    {
        "id": "vendor/paid-one",
        "name": "Paid One",
        "supported_parameters": ["tools"],
        "pricing": {"prompt": "0.000001", "completion": "0.000003"},
        "context_length": 128000,
    },
    {
        "id": "vendor/zebra:free",
        "name": "Zebra Free",
        "supported_parameters": ["tools"],
        "pricing": {"prompt": "0", "completion": "0"},
        "context_length": 64000,
    },
    {
        "id": "vendor/alpha:free",
        "name": "Alpha Free",
        "supported_parameters": ["tools"],
        "pricing": {"prompt": "0", "completion": "0"},
        "context_length": 32000,
    },
    {
        "id": "vendor/no-tools",
        "name": "No Tools",
        "supported_parameters": ["temperature"],
        "pricing": {"prompt": "0", "completion": "0"},
    },
]


@pytest.fixture(autouse=True)
def _clear_model_cache():
    model_catalogue.reset_cache()
    yield
    model_catalogue.reset_cache()


@pytest.fixture
def index(tmp_path: Path, sample_candidates) -> CandidateIndex:
    idx = CandidateIndex(path=tmp_path / "chroma", embedder=HashEmbedder())
    idx.rebuild(sample_candidates)
    return idx


def make_client(index, script, tmp_path: Path, fetcher=None) -> TestClient:
    llm = LLM(Settings(api_key="test", model="fake", fallback_models=()), client=FakeClient(script))
    app = create_app(
        llm=llm,
        index=index,
        photos_dir=tmp_path / "photos",
        user_dir=tmp_path / "user",
        models_fetcher=fetcher or (lambda: CATALOGUE),
    )
    return TestClient(app)


def sse(response) -> list[tuple[str, dict]]:
    """Parse an SSE body into (event, payload) pairs."""
    events, name, data = [], None, []
    for line in response.text.splitlines():
        if not line:
            if name and data:
                events.append((name, json.loads("\n".join(data))))
            name, data = None, []
        elif line.startswith("event:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data.append(line.split(":", 1)[1].lstrip())
    if name and data:
        events.append((name, json.loads("\n".join(data))))
    return events


# ---- page and widgets -------------------------------------------------------------------


def test_page_and_static_assets_are_served(index, tmp_path: Path):
    client = make_client(index, [], tmp_path)
    page = client.get("/")
    assert page.status_code == 200 and "CV Screener" in page.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/app.css").status_code == 200


def test_widgets_are_served_unchanged(index, tmp_path: Path):
    client = make_client(index, [], tmp_path)
    for name in ("results", "card", "deck"):
        html = client.get(f"/widgets/{name}.html")
        assert html.status_code == 200
        assert "{{" not in html.text and "ui/initialize" in html.text
    assert client.get("/widgets/nope.html").status_code == 404


# ---- /api/tool: same shapes as the MCP server -------------------------------------------


def test_widget_tools_match_the_mcp_shapes(index, tmp_path: Path):
    client = make_client(index, [], tmp_path)

    def call(name, args):
        r = client.post("/api/tool", json={"name": name, "arguments": args})
        assert r.status_code == 200, r.text
        return r.json()

    hits = call("search_candidates", {"languages": ["Spanish"], "k": 10})
    assert {h["id"] for h in hits} == {"p01-ana-ml", "p02-bob-qa"}
    assert "skills" in hits[0] and "headline" in hits[0]

    assert call("find_by_name", {"name": "Nowack"})[0]["id"] == "p03-cy-fe"

    profile = call("get_candidate", {"candidate_id": "p03-cy-fe"})
    assert profile["full_name"] == "Cy Nowak" and "photo_prompt" not in profile
    assert "error" in call("get_candidate", {"candidate_id": "nope"})

    assert call("get_candidate_photo", {"candidate_id": "p01-ana-ml"})["data_uri"].startswith(
        "data:image/png;base64,"
    )

    deck = call("get_deck", {})
    assert len(deck) == 3 and all(p["accent"].startswith("#") for p in deck)
    assert all("top_skills" in p for p in deck)


def test_unknown_tool_is_refused(index, tmp_path: Path):
    client = make_client(index, [], tmp_path)
    r = client.post("/api/tool", json={"name": "remove_candidate", "arguments": {}})
    assert r.status_code == 400 and "unknown tool" in r.json()["error"]


# ---- /api/chat --------------------------------------------------------------------------


def test_chat_streams_tool_widget_and_answer(index, tmp_path: Path):
    client = make_client(
        index,
        [
            message(tool_calls=[tool_call("search_candidates", languages=["Spanish"], k=10)]),
            message("Ana Ruiz and Bob Lee speak Spanish."),
        ],
        tmp_path,
    )
    events = sse(client.post("/api/chat", json={"message": "Which candidates speak Spanish?"}))
    assert [name for name, _ in events] == ["tool", "widget", "answer"]

    tool = dict(events)["tool"]
    assert tool["name"] == "search_candidates" and tool["count"] == 2
    assert set(tool["ids"]) == {"p01-ana-ml", "p02-bob-qa"}

    widget = dict(events)["widget"]
    assert widget["name"] == "results" and widget["args"]["languages"] == ["Spanish"]
    assert {h["id"] for h in widget["data"]} == {"p01-ana-ml", "p02-bob-qa"}

    answer = dict(events)["answer"]
    assert answer["grounded"] is True and "Ana Ruiz" in answer["text"]


def test_chat_reports_a_failing_model_as_an_error_event(index, tmp_path: Path, monkeypatch):
    monkeypatch.setattr("cv_screener.llm.time.sleep", lambda *_: None)
    client = make_client(index, [], tmp_path)  # empty script: the fake client raises
    events = sse(client.post("/api/chat", json={"message": "hello"}))
    assert [name for name, _ in events] == ["error"]
    assert "all models failed" in events[0][1]["message"]
    assert client.post("/api/chat", json={"message": "  "}).status_code == 400


def test_get_candidate_widget_payload_carries_contacts(index, tmp_path: Path):
    client = make_client(
        index,
        [
            message(tool_calls=[tool_call("get_candidate", candidate_id="p03-cy-fe")]),
            message("Cy Nowak builds React interfaces."),
        ],
        tmp_path,
    )
    widget = dict(sse(client.post("/api/chat", json={"message": "profile of Cy"})))["widget"]
    assert widget["name"] == "card" and widget["data"]["email"] == "p03-cy-fe@example.com"


def test_list_candidates_keeps_skills_out_of_the_model_context(index, tmp_path: Path):
    client = make_client(
        index,
        [
            message(tool_calls=[tool_call("list_candidates")]),
            message("The collection has 3 candidates."),
        ],
        tmp_path,
    )
    events = dict(sse(client.post("/api/chat", json={"message": "Show me the whole collection"})))
    assert events["widget"]["name"] == "deck"
    summary = events["widget"]["data"]
    assert summary["total"] == 3 and summary["user_added"] == 0
    blob = json.dumps({k: v for k, v in summary.items() if k != "note"})
    assert "PyTorch" not in blob and "skills" not in blob and "Native" not in blob
    assert sorted(summary["names"]) == ["Ana Ruiz", "Bob Lee", "Cy Nowak"]


NEW_RESUME = {
    "full_name": "Dana Okoye",
    "headline": "Staff Rust Engineer",
    "summary": "Builds embedded firmware in Rust.",
    "seniority": "lead",
    "role_family": "backend",
    "years_experience": 12,
    "country": "Nigeria",
    "skills": ["Rust", "Embedded C"],
    "languages": [{"name": "English", "level": "Native"}],
}


def test_add_candidate_through_the_agent_saves_and_indexes(index, tmp_path: Path):
    client = make_client(
        index,
        [
            message(tool_calls=[tool_call("add_candidate", **NEW_RESUME)]),
            message("Added Dana Okoye to the collection."),
        ],
        tmp_path,
    )
    events = dict(sse(client.post("/api/chat", json={"message": "add this resume"})))
    assert events["widget"]["name"] == "card"
    assert events["widget"]["data"]["id"] == "u01-dana-okoye"
    assert (tmp_path / "user" / "u01-dana-okoye.json").exists()

    hits = client.post(
        "/api/tool", json={"name": "search_candidates", "arguments": {"skills": ["Rust"]}}
    ).json()
    assert [h["id"] for h in hits] == ["u01-dana-okoye"]


def test_add_candidate_reports_invalid_profiles_to_the_model(index, tmp_path: Path):
    client = make_client(
        index,
        [
            message(
                tool_calls=[tool_call("add_candidate", **{**NEW_RESUME, "seniority": "wizard"})]
            ),
            message("I could not add that résumé."),
        ],
        tmp_path,
    )
    events = dict(sse(client.post("/api/chat", json={"message": "add this"})))
    assert "widget" not in events and events["tool"]["count"] == 0
    assert list((tmp_path / "user").glob("*.json")) == []


# ---- uploads ----------------------------------------------------------------------------


def test_upload_accepts_text_and_rejects_big_or_unsupported_files(index, tmp_path: Path):
    client = make_client(index, [], tmp_path)

    ok = client.post("/api/upload", files={"file": ("cv.txt", b"Dana Okoye\nRust engineer")})
    assert ok.status_code == 200
    body = ok.json()
    assert body["filename"] == "cv.txt" and "Dana Okoye" in body["prompt"]
    assert "add_candidate" in body["prompt"] and "never invent" in body["prompt"]

    big = client.post("/api/upload", files={"file": ("cv.txt", b"x" * (2 * 1024 * 1024 + 1))})
    assert big.status_code == 413 and "larger than" in big.json()["error"]

    bad = client.post("/api/upload", files={"file": ("cv.docx", b"PK\x03\x04junk")})
    assert bad.status_code == 415 and "unsupported file type" in bad.json()["error"]

    empty = client.post("/api/upload", files={"file": ("cv.md", b"   ")})
    assert empty.status_code == 422


def test_extract_text_rejects_unreadable_pdfs():
    with pytest.raises(ValueError, match="could not read"):
        extract_text("cv.pdf", b"not really a pdf")
    assert extract_text("cv.md", "héllo".encode()) == "héllo"


# ---- model picker -----------------------------------------------------------------------


def test_models_endpoint_lists_free_first_and_drops_non_tool_models(index, tmp_path: Path):
    client = make_client(index, [], tmp_path)
    payload = client.get("/api/models").json()
    ids = [m["id"] for m in payload["models"]]
    assert "vendor/no-tools" not in ids
    assert ids == ["vendor/alpha:free", "vendor/zebra:free", "vendor/paid-one"]
    paid = payload["models"][-1]
    assert paid["free"] is False and paid["prompt_price"] == 1.0 and paid["completion_price"] == 3.0
    assert payload["models"][0]["free"] is True


def test_models_endpoint_falls_back_to_the_configured_chain(index, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_MODEL", "primary/model:free")
    monkeypatch.setenv("OPENROUTER_FALLBACK_MODELS", "second/model:free,third/model")

    def boom():
        raise RuntimeError("offline")

    client = make_client(index, [], tmp_path, fetcher=boom)
    payload = client.get("/api/models").json()
    ids = [m["id"] for m in payload["models"]]
    assert ids[0] == "primary/model:free" and payload["default"] == "primary/model:free"
    assert set(ids) == {"primary/model:free", "second/model:free", "third/model"}
    assert ids[-1] == "third/model"  # the only paid one sorts last


def test_chat_rejects_a_model_outside_the_catalogue(index, tmp_path: Path):
    client = make_client(index, [], tmp_path)
    r = client.post("/api/chat", json={"message": "hi", "model": "evil/model"})
    assert r.status_code == 400 and "unknown model" in r.json()["error"]


def test_chat_uses_the_picked_model_with_the_rest_of_the_chain_behind_it(index, tmp_path: Path):
    used: list[str] = []
    llm = LLM(
        Settings(api_key="test", model="fake"),
        client=FakeClient([message("Nobody in the dataset matches.")]),
    )

    def factory(model: str | None) -> LLM:
        used.append(model or "")
        return build_llm(
            model,
            llm,
            Settings(api_key="test", model="default/model", fallback_models=("b/model", "c/model")),
        )

    app = create_app(
        llm=llm,
        index=index,
        photos_dir=tmp_path / "photos",
        user_dir=tmp_path / "user",
        llm_factory=factory,
        models_fetcher=lambda: CATALOGUE,
    )
    client = TestClient(app)
    events = dict(
        sse(client.post("/api/chat", json={"message": "hi", "model": "vendor/zebra:free"}))
    )
    assert used == ["vendor/zebra:free"]
    assert events["answer"]["model"] == "vendor/zebra:free"


def test_build_llm_puts_the_choice_first_and_keeps_the_others_as_fallbacks():
    base = LLM(Settings(api_key="test", model="fake"), client=FakeClient([]))
    cfg = Settings(api_key="k", model="a/model", fallback_models=("b/model", "c/model"))
    picked = build_llm("b/model", base, cfg)
    assert picked.cfg.model == "b/model"
    assert picked.cfg.model_chain == ("b/model", "a/model", "c/model")
    assert picked.client is base.client
    assert build_llm(None, base, cfg) is base
