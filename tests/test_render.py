"""Rendering is pure HTML generation plus a placeholder image; no network, no LLM."""

from __future__ import annotations

from pathlib import Path

import pytest

from cv_screener.generate.photos import find_photo, photo_data_uri, placeholder
from cv_screener.generate.render import TEMPLATES, render_html
from cv_screener.generate.schema import Candidate, Education, Experience, Language
from cv_screener.generate.themes import PHOTO_SHAPES, SKILL_STYLES, Theme, theme_for


@pytest.fixture
def candidate() -> Candidate:
    return Candidate(
        id="p99-test-person",
        full_name="Test Person",
        gender="female",
        age=30,
        city="Lisbon",
        country="Portugal",
        email="test.person@example.com",
        phone="+351 910 000 000",
        linkedin="linkedin.com/in/test-person",
        github=None,
        languages=[
            Language(name="Portuguese", level="Native"),
            Language(name="English", level="C1"),
        ],
        role_family="qa",
        seniority="middle",
        years_experience=4,
        template="classic",
        photo_prompt="headshot",
        headline="QA Automation Engineer",
        summary="Writes tests.",
        skills=["Playwright", "Java"],
        experience=[
            Experience(
                company="Acme",
                title="QA Engineer",
                location="Lisbon",
                start="2022-03",
                end=None,
                highlights=["Cut flaky tests by 40%"],
            )
        ],
        education=[
            Education(institution="IST", degree="BSc", field="CS", start_year=2015, end_year=2018)
        ],
        certifications=["ISTQB Foundation"],
    )


@pytest.mark.parametrize("template", TEMPLATES)
def test_every_template_renders_all_sections(candidate: Candidate, template: str, tmp_path: Path):
    html = render_html(candidate.model_copy(update={"template": template}), photos_dir=tmp_path)
    for needle in (
        "Test Person",
        "QA Automation Engineer",
        "Playwright",
        "Acme",
        "Mar 2022",
        "Present",
        "Portuguese",
        "ISTQB Foundation",
        "data:image/png;base64,",
    ):
        assert needle in html, f"{template}: missing {needle}"


def test_unknown_template_falls_back_to_classic(candidate: Candidate, tmp_path: Path):
    html = render_html(candidate.model_copy(update={"template": "nope"}), photos_dir=tmp_path)
    assert "Test Person" in html


def test_theme_is_deterministic_and_formats_dates():
    t = theme_for("p99-test-person")
    assert t == theme_for("p99-test-person")
    assert t.photo_shape in PHOTO_SHAPES and t.skill_style in SKILL_STYLES
    assert t.fmt_date(None) == "Present"
    assert (
        Theme("#000000", "serif", "serif", "square", "list", "%b %Y", "#eeeeee").fmt_date("2024-09")
        == "Sep 2024"
    )


@pytest.mark.parametrize("style", SKILL_STYLES)
def test_skill_styles_render(candidate: Candidate, style: str, tmp_path: Path):
    theme = Theme("#123456", "serif", "serif", "circle", style, "%Y", "#eeeeee")
    html = render_html(candidate, photos_dir=tmp_path, theme=theme)
    assert "Playwright" in html and "2022 – Present" in html


def test_placeholder_is_deterministic_and_uses_initials():
    a = placeholder("Test Person", "p99-test-person")
    b = placeholder("Test Person", "p99-test-person")
    assert a.tobytes() == b.tobytes()
    assert a.size == (512, 512)


def test_real_photo_wins_over_placeholder(candidate: Candidate, tmp_path: Path):
    assert find_photo(candidate.id, tmp_path) is None
    photo = tmp_path / f"{candidate.id}.jpg"
    photo.write_bytes(b"\xff\xd8\xff fake jpeg")
    assert find_photo(candidate.id, tmp_path) == photo
    assert photo_data_uri(candidate.id, candidate.full_name, tmp_path).startswith(
        "data:image/jpeg;base64,"
    )
