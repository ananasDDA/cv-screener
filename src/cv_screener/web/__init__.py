"""Local web chat: the project's own agent in a browser, hosting the MCP Apps widgets.

`cv web` serves a single page that talks to `app.py` over SSE and acts as an MCP Apps *host*:
the widgets from `cv_screener.widgets` run unchanged inside sandboxed iframes and speak
JSON-RPC over postMessage to the page, which forwards their `tools/call` to `/api/tool`.
"""

from __future__ import annotations

from .app import create_app, serve

__all__ = ["create_app", "serve"]
