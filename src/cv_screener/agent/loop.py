"""Hand-written tool-use loop. No framework: the whole agent is this file plus tools.py."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..index.store import CandidateIndex
from ..llm import LLM
from .tools import TOOLS, dispatch

SYSTEM = """You are a recruiting assistant answering questions about a dataset of candidate CVs.

Rules:
1. You cannot see the dataset. Retrieve with the tools; never guess or recall candidates from memory.
2. Name only candidates that appeared in tool results in this conversation, and cite what in their
   profile supports the answer (skills, role, years, languages with level).
3. If the tools return no candidates, or only weak matches (score below ~0.6 and nothing in the profile
   fits), say clearly that no one in the dataset matches. Do not invent or stretch.
4. Prefer exact filters for facts (languages, skills, seniority, years) and `query` for fuzzy needs
   (best fit for a role). Combine them when the question has both. Use k=10 or more for "who has / which
   candidates" questions so nobody is cut off.
5. For "summarize the profile of X" use find_by_name, then get_candidate for the full profile.
6. "Senior" in a question means senior or above: pass seniority ["senior", "lead", "principal"].
7. Be concise: a short answer with a bullet per candidate. Refer to people by name; never paste ids,
   JSON or citation markers into the answer."""


@dataclass
class ToolTrace:
    name: str
    args: dict[str, Any]
    count: int
    ids: list[str] = field(default_factory=list)


@dataclass
class Answer:
    text: str
    trace: list[ToolTrace]
    model: str | None
    grounded: bool  # every dataset name mentioned in the text came back from a tool
    ungrounded_names: list[str] = field(default_factory=list)


class Agent:
    def __init__(self, llm: LLM, index: CandidateIndex, max_steps: int = 6):
        self.llm = llm
        self.index = index
        self.max_steps = max_steps

    def ask(
        self, question: str, history: list[dict[str, Any]] | None = None, on_tool=None
    ) -> Answer:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM},
            *(history or []),
            {"role": "user", "content": question},
        ]
        trace: list[ToolTrace] = []
        seen_ids: set[str] = set()
        for _ in range(self.max_steps):
            msg = self.llm.chat(messages, tools=TOOLS, temperature=0.2)
            if not msg.tool_calls:
                return self._final(msg.content or "", trace, seen_ids)
            messages.append(_assistant_message(msg))
            for call in msg.tool_calls:
                args = _parse_args(call.function.arguments)
                result, count = dispatch(self.index, call.function.name, args)
                ids = re.findall(r'"id":\s*"([^"]+)"', result)
                seen_ids.update(ids)
                t = ToolTrace(call.function.name, args, count, ids)
                trace.append(t)
                if on_tool:
                    on_tool(t)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
        # Out of steps: force a final answer from what was retrieved.
        messages.append({"role": "user", "content": "Answer now using only the results above."})
        msg = self.llm.chat(messages, temperature=0.2)
        return self._final(msg.content or "", trace, seen_ids)

    def _final(self, text: str, trace: list[ToolTrace], seen_ids: set[str]) -> Answer:
        seen_names = {n for i, n in self.index.names().items() if i in seen_ids}
        mentioned = [n for n in self.index.names().values() if n.split()[0] in text or n in text]
        ungrounded = sorted(set(mentioned) - seen_names)
        return Answer(text.strip(), trace, self.llm.last_model, not ungrounded, ungrounded)


def _assistant_message(msg: Any) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": msg.content or "",
        "tool_calls": [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.function.name, "arguments": c.function.arguments},
            }
            for c in msg.tool_calls
        ],
    }


def _parse_args(raw: str | None) -> dict[str, Any]:
    try:
        return json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {}
