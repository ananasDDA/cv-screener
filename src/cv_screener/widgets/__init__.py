"""Self-contained HTML widgets rendered inside the chat by MCP Apps-capable hosts.

Each page is assembled from the shared tokens, bridge and profile renderer so the pieces stay
in one place; the result is a single HTML document with everything inlined (the host's default
CSP allows no external resources).
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

HERE = Path(__file__).parent
URIS = {
    "results": "ui://cv-screener/results.html",
    "card": "ui://cv-screener/card.html",
    "deck": "ui://cv-screener/deck.html",
}


@cache
def load(name: str) -> str:
    card_js = (HERE / "_card.js").read_text()
    profile_css = card_js.split("const PROFILE_CSS = `", 1)[1].rsplit("`;", 1)[0]
    html = (HERE / f"{name}.html").read_text()
    return (
        html.replace("{{TOKENS}}", (HERE / "_tokens.css").read_text())
        .replace("{{PROFILE_CSS}}", profile_css)
        .replace("{{BRIDGE}}", (HERE / "_bridge.js").read_text())
        .replace("{{CARD}}", card_js)
    )
