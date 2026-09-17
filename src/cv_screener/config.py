"""Project paths and environment-driven settings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CANDIDATES_DIR = DATA_DIR / "candidates"
PDF_DIR = DATA_DIR / "pdf"
PHOTOS_DIR = DATA_DIR / "photos"
CHROMA_DIR = DATA_DIR / "chroma"
PHOTO_PROMPTS_FILE = DATA_DIR / "photo_prompts.json"
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = field(default_factory=lambda: os.getenv("OPENROUTER_MODEL", ""))
    fallback_models: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            m.strip() for m in os.getenv("OPENROUTER_FALLBACK_MODELS", "").split(",") if m.strip()
        )
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    )

    @property
    def model_chain(self) -> tuple[str, ...]:
        return (self.model, *self.fallback_models)


def settings() -> Settings:
    return Settings()
