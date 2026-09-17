# CV Screener

Test task for a Python / AI developer role: generate a realistic synthetic CV dataset, index it into
a vector DB with structured fields, and answer questions about it through a tool-using CLI agent.
See `ROADMAP.md` for phases and `NOTES.md` for honest status.

## Layout
- `src/cv_screener/generate/` — persona grid (`personas.py`), Pydantic schema (`schema.py`),
  LLM profile generation (`generator.py`), HTML→PDF rendering (`render.py`, `templates/`).
- `src/cv_screener/index/` — ChromaDB collection, local fastembed embeddings, hybrid search
  (metadata filters + semantic ranking). No LLM here.
- `src/cv_screener/agent/` — hand-written tool-use loop over OpenRouter; tools call the index.
- `src/cv_screener/llm.py` — OpenRouter client with model fallback chain and JSON→Pydantic helper.
- `src/cv_screener/cli.py` — Typer entrypoints: `generate`, `index`, `search`, `chat`.
- `evals/` — JSON cases + runner printing pass/fail and a summary (needs an API key).
- `tests/` — pytest, must pass without any API key or network.
- `data/` — candidate JSON, PDFs, photos (committed); Chroma store (ignored).

## How to run
- Python 3.12 via `uv`. Always `uv run ...` or the Makefile targets; never a system Python.
- `make setup`, `make generate`, `make index`, `make chat`, `make test`, `make eval`.
- macOS: WeasyPrint needs `brew install pango`; the Makefile exports `DYLD_FALLBACK_LIBRARY_PATH`.
- `make check` (ruff + pytest) must be green before every commit.

## Architecture rules
- `Candidate` in `schema.py` is the single source of truth: JSON files, PDFs and index documents
  all derive from it. Never add a parallel representation.
- Diversity is designed, not hoped for: every candidate comes from an explicit persona spec.
  Identity (name, contacts) comes from seeded Faker; the LLM writes only the narrative.
- The index layer knows nothing about the LLM, and the agent knows nothing about Chroma.
  The three search functions are the only boundary, so an MCP server or web UI can reuse them.
- The agent retrieves through tools only; never put the whole dataset into a prompt.
- Answers cite candidate names/ids returned by tools. Empty results → say no one matches.
- LLM access only through OpenRouter; models come from `.env`, never hardcoded.
- Photos: rendering looks for `data/photos/<candidate_id>.png` and falls back to a deterministic
  placeholder avatar, so the pipeline never blocks on image generation.

## Python conventions
- `from __future__ import annotations`, type hints on every public function, `pathlib` over `os.path`.
- Pydantic models at every I/O boundary (LLM output, JSON files, tool arguments); plain dataclasses
  for internal config. No untyped dicts crossing module boundaries.
- Small pure functions; side effects (network, disk) at the edges so tests can mock them.
- Explicit errors: raise `LLMError`/`ValueError` with a message that says what to do, never
  swallow exceptions silently. Retries and fallbacks are logged, not hidden.
- Dependency injection for the LLM client (`LLM(client=...)`) so tests use a fake.
- Ruff for lint and format (config in `pyproject.toml`), line length 100. No commented-out code.
- No global mutable state; settings are read once via `config.settings()`.

## Git conventions
- English, imperative subject ≤ 72 chars, no trailing period: "Add hybrid search over Chroma".
- Body explains *why* and any non-obvious decision; wrap at 72. Skip the body for trivial changes.
- One logical change per commit, matching a roadmap phase or a clear sub-step of one.
  No "wip", "fix", "misc" commits; squash local noise before pushing.
- Data regeneration goes in its own commit ("Regenerate dataset after prompt change"), never mixed
  with code.
- Never commit `.env`, `data/chroma/`, or anything with a key. `.env.example` lists every variable.
- Every commit ends with the Co-Authored-By trailer for Claude.

## Testing and evals
- Tests are deterministic and offline. Mock the LLM, use the local embedding model, use `tmp_path`
  for Chroma.
- Evals hit the real model. Results are pasted into `NOTES.md` exactly as printed, failures included.
- After changing a prompt or the persona grid, re-run evals and update NOTES.md.
