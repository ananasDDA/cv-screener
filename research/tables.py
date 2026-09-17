"""Small HTML tables for the notebook, same style as the helper in `demo.ipynb`."""

from __future__ import annotations

from typing import Any

from IPython.display import HTML, display

_CELL = "padding:2px 8px;vertical-align:top;border-bottom:1px solid #8884"


def table(rows: list[dict[str, Any]], cols: list[str] | None = None, note: str = "") -> None:
    if not rows:
        display(HTML("<i>no rows</i>"))
        return
    cols = cols or list(rows[0])
    head = "".join(f"<th style='text-align:left;{_CELL}'>{c}</th>" for c in cols)
    body = "".join(
        "<tr>" + "".join(f"<td style='{_CELL}'>{_fmt(r.get(c, ''))}</td>" for c in cols) + "</tr>"
        for r in rows
    )
    tail = f"<div style='font-size:11px;opacity:.7;padding-top:4px'>{note}</div>" if note else ""
    display(
        HTML(
            f"<table style='font-size:12px;border-collapse:collapse'>"
            f"<tr>{head}</tr>{body}</table>{tail}"
        )
    )


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".")
    if value is None:
        return "—"
    return str(value)
