"""Thin OpenRouter client: model fallback chain, retries, and JSON-into-Pydantic helper.

Free-tier models on OpenRouter are flaky (429s, empty choices), so every call walks the
configured model chain and retries a few times before giving up.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from .config import Settings, settings

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


class LLM:
    def __init__(self, cfg: Settings | None = None, client: OpenAI | None = None):
        self.cfg = cfg or settings()
        if not self.cfg.api_key and client is None:
            raise LLMError("OPENROUTER_API_KEY is not set. Copy .env.example to .env and fill it in.")
        self.client = client or OpenAI(base_url=self.cfg.base_url, api_key=self.cfg.api_key)
        self.last_model: str | None = None

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        attempts_per_model: int = 2,
    ) -> Any:
        """Return the first choice message. Walks the model chain on any failure."""
        errors: list[str] = []
        for model in self.cfg.model_chain:
            for attempt in range(attempts_per_model):
                try:
                    kwargs: dict[str, Any] = dict(
                        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
                    )
                    if tools:
                        kwargs["tools"] = tools
                    resp = self.client.chat.completions.create(**kwargs)
                    if not resp.choices:
                        raise LLMError("empty choices")
                    self.last_model = model
                    return resp.choices[0].message
                except Exception as exc:  # noqa: BLE001 - we want to fall through to the next model
                    errors.append(f"{model}: {type(exc).__name__}: {str(exc)[:120]}")
                    time.sleep(1.5 * (attempt + 1))
        raise LLMError("all models failed:\n" + "\n".join(errors))

    def structured(self, system: str, user: str, schema: type[T], retries: int = 3) -> T:
        """Ask for JSON matching `schema`; validate; feed validation errors back on failure."""
        messages = [
            {"role": "system", "content": system + "\n\nRespond with a single JSON object only. "
             "No prose, no markdown fences.\nJSON schema:\n" + json.dumps(schema.model_json_schema())},
            {"role": "user", "content": user},
        ]
        last_err = ""
        for _ in range(retries):
            msg = self.chat(messages, temperature=0.8)
            text = msg.content or ""
            try:
                return schema.model_validate(extract_json(text))
            except (ValueError, ValidationError) as exc:
                last_err = str(exc)[:800]
                messages += [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": f"Invalid JSON for the schema: {last_err}\nReturn the corrected JSON object only."},
                ]
        raise LLMError(f"could not get valid {schema.__name__}: {last_err}")


def extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of model output that may include fences or chatter."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in response")
    return json.loads(text[start : end + 1])
