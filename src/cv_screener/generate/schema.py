"""Candidate data model. This is the single source of truth for JSON files, PDFs and the index."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Seniority = Literal["junior", "middle", "senior", "lead", "principal"]
LanguageLevel = Literal["Native", "C2", "C1", "B2", "B1", "A2", "A1"]


class Experience(BaseModel):
    company: str
    title: str
    location: str
    start: str = Field(description="YYYY-MM")
    end: str | None = Field(default=None, description="YYYY-MM, null if current")
    highlights: list[str] = Field(description="3-5 concrete, quantified bullet points")


class Education(BaseModel):
    institution: str
    degree: str = Field(description="e.g. BSc, MSc, PhD, Diploma")
    field: str
    start_year: int
    end_year: int


class Language(BaseModel):
    name: str
    level: LanguageLevel


class GeneratedProfile(BaseModel):
    """The part of a CV the LLM writes. Identity and languages come from the persona spec."""

    headline: str = Field(
        description="One-line professional title as it would appear under the name"
    )
    summary: str = Field(description="3-4 sentence professional summary in first or third person")
    skills: list[str] = Field(description="12-20 concrete technologies, tools and methods")
    experience: list[Experience] = Field(description="2-5 positions, most recent first")
    education: list[Education]
    certifications: list[str] = Field(default_factory=list, description="0-3 real certifications")


class Candidate(GeneratedProfile):
    id: str
    full_name: str
    gender: Literal["male", "female"]
    age: int
    city: str
    country: str
    email: str
    phone: str
    linkedin: str
    github: str | None = None
    languages: list[Language]
    role_family: str
    seniority: Seniority
    years_experience: int
    template: str
    photo_prompt: str

    def search_text(self) -> str:
        """Plain-text rendering used for embeddings."""
        exp = "\n".join(
            f"{e.title} at {e.company} ({e.start} – {e.end or 'present'}): "
            + " ".join(e.highlights)
            for e in self.experience
        )
        edu = "; ".join(f"{e.degree} in {e.field}, {e.institution}" for e in self.education)
        langs = ", ".join(f"{lang.name} ({lang.level})" for lang in self.languages)
        return (
            f"{self.full_name}. {self.headline}. {self.seniority} {self.role_family}, "
            f"{self.years_experience} years of experience. {self.city}, {self.country}.\n"
            f"{self.summary}\nSkills: {', '.join(self.skills)}\nLanguages: {langs}\n"
            f"Experience:\n{exp}\nEducation: {edu}\n"
            f"Certifications: {', '.join(self.certifications)}"
        )
