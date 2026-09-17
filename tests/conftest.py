from __future__ import annotations

import pytest

from cv_screener.generate.schema import Candidate, Education, Experience, Language


def make_candidate(
    id: str,
    name: str,
    *,
    seniority: str,
    role: str,
    country: str,
    years: int,
    languages: list[tuple[str, str]],
    skills: list[str],
    summary: str,
) -> Candidate:
    return Candidate(
        id=id,
        full_name=name,
        gender="female",
        age=30,
        city="X",
        country=country,
        email=f"{id}@example.com",
        phone="+1 000",
        linkedin=f"linkedin.com/in/{id}",
        github=None,
        languages=[Language(name=n, level=lv) for n, lv in languages],
        role_family=role,
        seniority=seniority,
        years_experience=years,
        template="classic",
        photo_prompt="",
        headline=f"{seniority} {role}",
        summary=summary,
        skills=skills,
        experience=[
            Experience(
                company="Co",
                title=role,
                location="X",
                start="2020-01",
                end=None,
                highlights=[summary],
            )
        ],
        education=[
            Education(institution="U", degree="BSc", field="CS", start_year=2010, end_year=2014)
        ],
    )


@pytest.fixture
def sample_candidates() -> list[Candidate]:
    return [
        make_candidate(
            "p01-ana-ml",
            "Ana Ruiz",
            seniority="senior",
            role="ml",
            country="Spain",
            years=8,
            languages=[("Spanish", "Native"), ("English", "C1")],
            skills=["Python", "PyTorch", "MLflow"],
            summary="Trains and deploys deep learning models with PyTorch on GCP",
        ),
        make_candidate(
            "p02-bob-qa",
            "Bob Lee",
            seniority="middle",
            role="qa",
            country="Portugal",
            years=4,
            languages=[("Portuguese", "Native"), ("Spanish", "B2"), ("English", "C1")],
            skills=["Java", "Selenium", "Playwright"],
            summary="Automates regression test suites with Selenium and Playwright",
        ),
        make_candidate(
            "p03-cy-fe",
            "Cy Nowak",
            seniority="junior",
            role="frontend",
            country="Poland",
            years=2,
            languages=[("Polish", "Native"), ("English", "B2")],
            skills=["React", "TypeScript"],
            summary="Builds React interfaces with TypeScript",
        ),
    ]
