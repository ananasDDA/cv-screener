"""Starlette app behind `cv web`: one chat page, an SSE turn endpoint and the widget bridge.

Boring on purpose — ASGI + Server-Sent Events + a static page with no build step. The agent is
synchronous, so a turn runs in a worker thread and pushes events into a queue that the SSE
generator drains.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import queue
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from sse_starlette.sse import EventSourceResponse
from starlette.applications import Starlette
from starlette.formparsers import MultiPartException
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from ..config import CHROMA_DIR, PHOTOS_DIR, USER_CANDIDATES_DIR, Settings, settings
from ..index.store import CandidateIndex
from ..llm import LLM
from ..widgets import URIS
from ..widgets import load as load_widget
from . import models as model_catalogue
from .agent import WebAgent
from .data import call_widget_tool

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD = 2 * 1024 * 1024  # 2 MB is plenty for a résumé; anything bigger is a mistake
TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".text"}
HISTORY_TURNS = 6

UPLOAD_PROMPT = (
    "I uploaded a résumé ({filename}). Add this person to the collection with add_candidate. "
    "Use only facts written below; leave unknown optional fields empty and never invent anything.\n"
    "--- résumé text ---\n{text}\n--- end of résumé ---"
)


def build_llm(model: str | None, base: LLM | None = None, cfg: Settings | None = None) -> LLM:
    """LLM for one request. A picked model becomes the head of the chain and the configured
    models stay behind it as fallbacks, because free models drop requests all the time."""
    if not model:
        return base or LLM()
    cfg = cfg or settings()
    chain = tuple(m for m in cfg.model_chain if m and m != model)
    picked = Settings(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        model=model,
        fallback_models=chain,
        embedding_model=cfg.embedding_model,
    )
    # Tests inject a fake transport; keep it so a picked model stays offline there too.
    return LLM(picked, client=base.client) if base is not None else LLM(picked)


def create_app(
    llm: LLM | None = None,
    index: CandidateIndex | None = None,
    photos_dir: Path = PHOTOS_DIR,
    user_dir: Path = USER_CANDIDATES_DIR,
    llm_factory: Callable[[str | None], LLM] | None = None,
    models_fetcher: Callable[[], list[dict[str, Any]]] | None = None,
) -> Starlette:
    """Build the ASGI app. Tests inject a fake `llm`, a tmp-path `index` and a models fetcher."""
    if index is None:
        from ..mcp_server import ensure_index

        index = CandidateIndex(CHROMA_DIR)
        ensure_index(index, user_dir=user_dir)
    make_llm = llm_factory or (lambda model: build_llm(model, llm))
    default_agent = WebAgent(llm or LLM(), index, user_dir=user_dir)

    def agent_for(model: str | None) -> WebAgent:
        if not model or model == settings().model:
            return default_agent
        return WebAgent(make_llm(model), index, user_dir=user_dir)

    def known_models() -> list[dict[str, Any]]:
        return model_catalogue.catalogue(models_fetcher)

    async def home(_: Request) -> Response:
        return FileResponse(STATIC / "index.html")

    async def widget(request: Request) -> Response:
        name = request.path_params["name"]
        if name not in URIS:
            return JSONResponse({"error": f"unknown widget {name!r}"}, status_code=404)
        return HTMLResponse(load_widget(name))

    async def api_models(_: Request) -> Response:
        entries = await asyncio.to_thread(known_models)
        return JSONResponse({"default": settings().model, "models": entries})

    async def api_tool(request: Request) -> Response:
        body = await _json_body(request)
        name = str(body.get("name") or "")
        args = body.get("arguments") or {}
        if not isinstance(args, dict):
            return JSONResponse({"error": "arguments must be an object"}, status_code=400)
        try:
            return JSONResponse(call_widget_tool(index, name, args, photos_dir))
        except KeyError:
            return JSONResponse({"error": f"unknown tool {name!r}"}, status_code=400)

    async def api_chat(request: Request) -> Response:
        body = await _json_body(request)
        message = str(body.get("message") or "").strip()
        if not message:
            return JSONResponse({"error": "message is empty"}, status_code=400)
        model = str(body.get("model") or "").strip() or None
        if model:
            entries = await asyncio.to_thread(known_models)
            if model not in model_catalogue.known_ids(entries):
                return JSONResponse({"error": f"unknown model {model!r}"}, status_code=400)
        history = _clean_history(body.get("history"))
        try:
            agent = agent_for(model)
        except Exception as exc:  # noqa: BLE001 - a bad key surfaces here, not as a 500
            return JSONResponse({"error": str(exc)}, status_code=400)
        return EventSourceResponse(_turn_events(agent, message, history), ping=15)

    async def api_upload(request: Request) -> Response:
        too_big = JSONResponse(
            {"error": f"the file is larger than {MAX_UPLOAD // (1024 * 1024)} MB"}, status_code=413
        )
        try:
            # Starlette caps a part at 1 MB by default; raise it so the size check below is ours.
            form = await request.form(max_files=1, max_fields=4, max_part_size=MAX_UPLOAD + 65536)
        except MultiPartException:
            return too_big
        upload = form.get("file")
        if not hasattr(upload, "read"):
            return JSONResponse({"error": "no file in the request"}, status_code=400)
        name = Path(str(upload.filename or "resume")).name
        blob = await upload.read()
        if len(blob) > MAX_UPLOAD:
            return too_big
        try:
            text = extract_text(name, blob)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=415)
        if not text.strip():
            return JSONResponse({"error": f"no text found in {name}"}, status_code=422)
        return JSONResponse(
            {
                "filename": name,
                "chars": len(text),
                "prompt": UPLOAD_PROMPT.format(filename=name, text=text[:40000]),
            }
        )

    return Starlette(
        routes=[
            Route("/", home),
            Route("/widgets/{name}.html", widget),
            Route("/api/models", api_models),
            Route("/api/tool", api_tool, methods=["POST"]),
            Route("/api/chat", api_chat, methods=["POST"]),
            Route("/api/upload", api_upload, methods=["POST"]),
            Mount("/static", StaticFiles(directory=STATIC), name="static"),
        ]
    )


# ---- one turn, streamed ----------------------------------------------------------------


async def _turn_events(agent: WebAgent, message: str, history: list[dict[str, Any]]):
    """Run the synchronous agent in a thread; yield its tool calls, widget and answer as SSE."""
    events: queue.Queue[tuple[str, Any] | None] = queue.Queue()

    def run() -> None:
        try:
            turn = agent.ask_turn(
                message, history, on_tool=lambda t: events.put(("tool", asdict(t)))
            )
            if turn.widget is not None:
                events.put(("widget", asdict(turn.widget)))
            events.put(
                (
                    "answer",
                    {
                        "text": turn.answer.text,
                        "model": turn.answer.model,
                        "grounded": turn.answer.grounded,
                        "ungrounded_names": turn.answer.ungrounded_names,
                    },
                )
            )
        except Exception as exc:  # noqa: BLE001 - the browser is the only place this can surface
            events.put(("error", {"message": f"{type(exc).__name__}: {exc}"}))
        finally:
            events.put(None)

    worker = asyncio.create_task(asyncio.to_thread(run))
    try:
        while True:
            item = await asyncio.to_thread(events.get)
            if item is None:
                break
            name, payload = item
            yield {"event": name, "data": json.dumps(payload, ensure_ascii=False)}
    finally:
        with contextlib.suppress(Exception):
            await worker


# ---- helpers ---------------------------------------------------------------------------


def extract_text(filename: str, blob: bytes) -> str:
    """Résumé text from an upload. PDFs go through pypdf; text files are decoded as UTF-8."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        import io

        from pypdf import PdfReader

        try:
            reader = PdfReader(io.BytesIO(blob))
            return "\n".join((page.extract_text() or "") for page in reader.pages).strip()
        except Exception as exc:  # noqa: BLE001 - a broken PDF is a user error, not a crash
            raise ValueError(f"could not read {filename} as a PDF: {exc}") from None
    if suffix in TEXT_SUFFIXES:
        return blob.decode("utf-8", errors="replace")
    raise ValueError(f"unsupported file type {suffix or filename!r}; use PDF, .txt or .md")


def _clean_history(raw: Any) -> list[dict[str, Any]]:
    """Keep the last few user/assistant turns; ignore anything else the page might send."""
    if not isinstance(raw, list):
        return []
    kept = [
        {"role": m["role"], "content": str(m.get("content") or "")}
        for m in raw
        if isinstance(m, dict) and m.get("role") in {"user", "assistant"}
    ]
    return kept[-2 * HISTORY_TURNS :]


async def _json_body(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 - malformed body is just an empty request
        return {}
    return body if isinstance(body, dict) else {}


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Blocking entrypoint used by `cv web`."""
    import uvicorn

    uvicorn.run(create_app(), host=host, port=port, log_level="info")
