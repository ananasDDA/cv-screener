"""Project paths and environment-driven settings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = PACKAGE_DIR / "templates"

# Résumés added through MCP are real people's data: they live outside the repo, never in git.
USER_HOME = Path(os.getenv("CV_SCREENER_HOME", "~/.cv-screener")).expanduser()
USER_CANDIDATES_DIR = USER_HOME / "candidates"

# Two layouts. In a git checkout everything lives under ./data. When installed as a package
# (`uvx --from git+https://github.com/ananasDDA/cv-screener cv mcp`) the dataset ships inside the
# wheel, read-only, and everything writable goes to the user's home directory.
IS_CHECKOUT = (ROOT / "data" / "candidates").is_dir() and (ROOT / "pyproject.toml").is_file()
if IS_CHECKOUT:
    DATA_DIR = ROOT / "data"
    CANDIDATES_DIR = DATA_DIR / "candidates"
    PHOTOS_DIR = DATA_DIR / "photos"
    WRITABLE_DIR = DATA_DIR
else:
    DATA_DIR = PACKAGE_DIR / "_data"
    CANDIDATES_DIR = DATA_DIR / "candidates"
    PHOTOS_DIR = DATA_DIR / "photos"
    WRITABLE_DIR = USER_HOME
PDF_DIR = WRITABLE_DIR / "pdf"
CHROMA_DIR = WRITABLE_DIR / "chroma"
PHOTO_PROMPTS_FILE = WRITABLE_DIR / "photo_prompts.json"

load_dotenv(ROOT / ".env")
# The HF tokenizers used by fastembed fork worker threads; without this they can abort at exit.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


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
