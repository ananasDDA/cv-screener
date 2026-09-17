"""Per-candidate visual theme, derived deterministically from the candidate id.

Three hand-written layouts stay as the reliable skeleton (WeasyPrint is picky about CSS);
variety comes from parameters layered on top: colours, fonts, photo shape, skills style,
date format. Same id → same theme, so re-rendering is reproducible.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime

ACCENTS = [
    "#16324f",
    "#6b5b3a",
    "#2f5d50",
    "#7a3e48",
    "#3d4a6b",
    "#8a5a1e",
    "#1f4e5f",
    "#5c4b7a",
    "#333333",
]
FONT_PAIRS = [
    ("Georgia, 'Times New Roman', serif", "Georgia, 'Times New Roman', serif"),
    (
        "'Helvetica Neue', Helvetica, Arial, sans-serif",
        "'Helvetica Neue', Helvetica, Arial, sans-serif",
    ),
    ("Georgia, serif", "'Helvetica Neue', Arial, sans-serif"),
    (
        "'Avenir Next', 'Segoe UI', Arial, sans-serif",
        "'Avenir Next', 'Segoe UI', Arial, sans-serif",
    ),
    ("'Palatino', 'Book Antiqua', serif", "'Palatino', 'Book Antiqua', serif"),
    ("'Gill Sans', 'Trebuchet MS', sans-serif", "'Gill Sans', 'Trebuchet MS', sans-serif"),
]
PHOTO_SHAPES = ["square", "circle", "rounded", "portrait", "portrait-bw"]
SKILL_STYLES = ["list", "inline", "tags"]
DATE_FORMATS = ["%b %Y", "%m/%Y", "%Y"]


@dataclass(frozen=True)
class Theme:
    accent: str
    heading_font: str
    body_font: str
    photo_shape: str
    skill_style: str
    date_format: str
    tint: str  # light background derived from the accent, used for sidebars and tags

    def fmt_date(self, value: str | None) -> str:
        if not value:
            return "Present"
        return datetime.strptime(value, "%Y-%m").strftime(self.date_format)


def theme_for(candidate_id: str) -> Theme:
    """Ids like `p07-...` carry the persona number; walking the option lists with different
    strides spreads themes evenly over a small dataset. Anything else falls back to a hash."""
    m = re.match(r"p(\d+)-", candidate_id)
    if m:
        n = int(m.group(1))
        picks = (n, n * 5, n * 2, n, n * 2)
    else:
        picks = tuple(hashlib.sha256(candidate_id.encode()).digest()[:5])
    accent = ACCENTS[picks[0] % len(ACCENTS)]
    heading, body = FONT_PAIRS[picks[1] % len(FONT_PAIRS)]
    return Theme(
        accent=accent,
        heading_font=heading,
        body_font=body,
        photo_shape=PHOTO_SHAPES[picks[2] % len(PHOTO_SHAPES)],
        skill_style=SKILL_STYLES[picks[3] % len(SKILL_STYLES)],
        date_format=DATE_FORMATS[picks[4] % len(DATE_FORMATS)],
        tint=_tint(accent),
    )


def _tint(hex_color: str, mix: float = 0.9) -> str:
    """Blend the accent towards white to get a paper-safe background."""
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    r, g, b = (round(c + (255 - c) * mix) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"
