"""Model catalogue for the picker: OpenRouter's public list, filtered to tool-calling models.

No key is needed for `/api/v1/models`. The result is cached in memory for an hour; if the
request fails (offline, rate limit) the project's own chain from `.env` is used instead, so the
picker always has something to show.
"""

from __future__ import annotations

import time
from typing import Any

from ..config import Settings, settings

CATALOGUE_URL = "https://openrouter.ai/api/v1/models"
CACHE_TTL = 3600.0
FALLBACK_TTL = 60.0  # a failed fetch should not pin the static list for an hour

_cache: tuple[float, list[dict[str, Any]]] | None = None


def reset_cache() -> None:
    global _cache
    _cache = None


def fetch_models() -> list[dict[str, Any]]:
    """Raw `data` array from OpenRouter. Blocking; callers run it in a thread."""
    import httpx

    response = httpx.get(CATALOGUE_URL, timeout=10.0)
    response.raise_for_status()
    payload = response.json()
    return payload.get("data") or []


def catalogue(fetcher: Any = None, cfg: Settings | None = None) -> list[dict[str, Any]]:
    """Cached list of `{id, name, free, context_length, prompt_price, completion_price}`."""
    global _cache
    now = time.monotonic()
    if _cache and now < _cache[0]:
        return _cache[1]
    ttl = CACHE_TTL
    try:
        entries = [_entry(m) for m in (fetcher or fetch_models)() if _supports_tools(m)]
    except Exception:  # noqa: BLE001 - offline is a normal state for a local tool
        entries = []
    if not entries:
        entries, ttl = _static(cfg), FALLBACK_TTL
    ordered = order(entries, cfg)
    _cache = (now + ttl, ordered)
    return ordered


def order(entries: list[dict[str, Any]], cfg: Settings | None = None) -> list[dict[str, Any]]:
    """Configured default first, then free models, then paid ones; alphabetical inside a group."""
    default = (cfg or settings()).model
    seen: dict[str, dict[str, Any]] = {}
    for e in entries:
        seen.setdefault(e["id"], e)
    ranked = sorted(
        seen.values(),
        key=lambda e: (e["id"] != default, not e["free"], (e["name"] or e["id"]).lower()),
    )
    return ranked


def known_ids(entries: list[dict[str, Any]]) -> set[str]:
    return {e["id"] for e in entries}


def _supports_tools(model: dict[str, Any]) -> bool:
    return "tools" in (model.get("supported_parameters") or [])


def _entry(model: dict[str, Any]) -> dict[str, Any]:
    pricing = model.get("pricing") or {}
    prompt = _per_million(pricing.get("prompt"))
    completion = _per_million(pricing.get("completion"))
    model_id = str(model.get("id") or "")
    return {
        "id": model_id,
        "name": str(model.get("name") or model_id),
        "free": model_id.endswith(":free") or (prompt == 0.0 and completion == 0.0),
        "context_length": model.get("context_length"),
        "prompt_price": prompt,
        "completion_price": completion,
    }


def _per_million(raw: Any) -> float:
    try:
        return round(float(raw) * 1_000_000, 4)
    except (TypeError, ValueError):
        return 0.0


def _static(cfg: Settings | None = None) -> list[dict[str, Any]]:
    chain = [m for m in (cfg or settings()).model_chain if m]
    return [
        {
            "id": m,
            "name": m.rsplit("/", 1)[-1].replace(":free", ""),
            "free": m.endswith(":free"),
            "context_length": None,
            "prompt_price": 0.0,
            "completion_price": 0.0,
        }
        for m in chain
    ]
