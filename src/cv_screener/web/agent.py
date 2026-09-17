"""The agent from `agent/loop.py`, extended with the web-only tools.

`Agent.ask` reads `agent.tools.TOOLS`/`dispatch` from module scope, so there is no seam to
subclass through: the loop body (~30 lines) is re-implemented here against an injectable tool
table. Everything else — system prompt, message shaping, the grounding check — is imported from
`agent/loop.py`, which stays untouched.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..agent.loop import SYSTEM, Agent, Answer, ToolTrace, _assistant_message, _parse_args
from ..config import USER_CANDIDATES_DIR
from ..index.store import CandidateIndex
from ..llm import LLM
from .agent_tools import TOOLS, WIDGET_FOR, dispatch

WEB_SYSTEM = (
    SYSTEM
    + """
8. `list_candidates` puts the whole collection in front of the user as a browsable deck; use it when
   they ask to see or browse everything. It returns counts only, so any question about the people in
   it still needs search_candidates or get_candidate.
9. When the user pastes or uploads a résumé and asks to add it, call `add_candidate` once with the
   facts you can read. Never invent: leave unknown optional fields empty. Then confirm in one line.
10. Results, profiles and the deck are already shown to the user as interactive widgets above your
   answer. Do not repeat every field in prose; summarize."""
)


@dataclass
class Widget:
    """Which widget the transcript should render, and the tool result to feed it."""

    name: str  # results | card | deck
    tool: str
    args: dict[str, Any]
    data: Any


@dataclass
class Turn:
    answer: Answer
    widget: Widget | None


class WebAgent(Agent):
    def __init__(
        self,
        llm: LLM,
        index: CandidateIndex,
        user_dir: Path = USER_CANDIDATES_DIR,
        max_steps: int = 6,
    ):
        super().__init__(llm, index, max_steps)
        self.user_dir = user_dir

    def ask_turn(
        self,
        question: str,
        history: list[dict[str, Any]] | None = None,
        on_tool: Any = None,
    ) -> Turn:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": WEB_SYSTEM},
            *(history or []),
            {"role": "user", "content": question},
        ]
        trace: list[ToolTrace] = []
        seen_ids: set[str] = set()
        widget: Widget | None = None
        for _ in range(self.max_steps):
            msg = self.llm.chat(messages, tools=TOOLS, temperature=0.2)
            if not msg.tool_calls:
                return Turn(self._final(msg.content or "", trace, seen_ids), widget)
            messages.append(_assistant_message(msg))
            for call in msg.tool_calls:
                name = call.function.name
                args = _parse_args(call.function.arguments)
                result, count, payload = dispatch(self.index, name, args, self.user_dir)
                ids = _ids(result)
                seen_ids.update(ids)
                t = ToolTrace(name, args, count, ids)
                trace.append(t)
                if on_tool:
                    on_tool(t)
                if payload is not None and name in WIDGET_FOR:
                    widget = Widget(WIDGET_FOR[name], name, args, payload)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
        messages.append({"role": "user", "content": "Answer now using only the results above."})
        msg = self.llm.chat(messages, temperature=0.2)
        return Turn(self._final(msg.content or "", trace, seen_ids), widget)


def _ids(result: str) -> list[str]:
    return re.findall(r'"id":\s*"([^"]+)"', result)
