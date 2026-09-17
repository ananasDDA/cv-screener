"""Render Candidate records to PDF through Jinja2 HTML templates and WeasyPrint."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..config import PDF_DIR, PHOTOS_DIR, TEMPLATES_DIR
from .photos import photo_data_uri
from .schema import Candidate
from .themes import Theme, theme_for

TEMPLATES = ("classic", "modern", "minimal")


def _env(theme: Theme) -> Environment:
    env = Environment(
        loader=FileSystemLoader(TEMPLATES_DIR), autoescape=select_autoescape(["html"])
    )
    env.filters["date"] = theme.fmt_date
    return env


def render_html(c: Candidate, photos_dir: Path = PHOTOS_DIR, theme: Theme | None = None) -> str:
    template = c.template if c.template in TEMPLATES else TEMPLATES[0]
    theme = theme or theme_for(c.id)
    photo = photo_data_uri(c.id, c.full_name, photos_dir)
    return _env(theme).get_template(f"{template}.html").render(c=c, t=theme, photo=photo)


def render_pdf(c: Candidate, out_dir: Path = PDF_DIR, photos_dir: Path = PHOTOS_DIR) -> Path:
    from weasyprint import HTML  # imported lazily: needs system libs, tests may not have them

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{c.id}.pdf"
    HTML(string=render_html(c, photos_dir), base_url=str(TEMPLATES_DIR)).write_pdf(path)
    return path
