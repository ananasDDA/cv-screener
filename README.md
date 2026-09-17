# CV Screener

[![CI](https://github.com/ananasDDA/cv-screener/actions/workflows/ci.yml/badge.svg)](https://github.com/ananasDDA/cv-screener/actions/workflows/ci.yml)

Synthetic CV dataset, hybrid vector search and a tool-using CLI agent that answers recruiter
questions about the candidates. Python 3.12, one command per stage.

```
personas → LLM (JSON by schema) → candidates/*.json → PDF (3 layouts × themes)
                                        └→ ChromaDB (vector + metadata) → agent tools → cv chat
```

## What it does

- **Generate** 14 realistic, diverse candidates (roles, levels, countries, stacks, languages) with a
  PDF résumé each. Diversity is designed in a persona grid, not left to the model.
- **Index** them into ChromaDB with local embeddings (no key needed) and structured metadata, so
  search works by exact fields (language, skill, seniority, years) and by meaning, or both at once.
- **Chat** with an agent that retrieves through tools only, names the candidates it actually found,
  and says so when nobody matches.
- **Evals** on the real model print pass/fail per case; **tests** run offline without a key.

## Setup from zero

Requirements: Python 3.12 (uv installs it if missing), [uv](https://docs.astral.sh/uv/), and
Pango/Cairo for PDF rendering:

- macOS: `brew install pango`
- Debian/Ubuntu: `sudo apt install libpango-1.0-0 libpangoft2-1.0-0`

```bash
git clone https://github.com/ananasDDA/cv-screener.git
cd cv-screener
make setup                      # uv sync: creates .venv, installs everything
cp .env.example .env            # then put your OpenRouter key into OPENROUTER_API_KEY
```

`.env` also selects the chat model and a fallback chain. Defaults are free OpenRouter models with
tool-calling support; any OpenAI-compatible model id works.

## Run

| Stage | Command | Needs key |
|---|---|---|
| Generate profiles + PDFs | `make generate` (or `uv run cv generate`) | yes |
| Re-render PDFs only | `uv run cv generate --pdf-only` | no |
| Build the index | `make index` | no |
| Search from the shell | `uv run cv search "senior ML" -s senior -s principal` | no |
| Chat with the agent | `make chat` / `uv run cv chat -q "Who speaks Spanish?"` | yes |
| Tests (offline) | `make test` | no |
| Evals (real model) | `make eval` | no |
| Lint + format + tests | `make check` | no |

The repository already contains a generated dataset (`data/candidates`, `data/pdf`), so
`make index` and `make chat` work right after setup. `make generate` skips candidates that
already exist; add `--force` to regenerate.

On macOS the Makefile exports `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib` so WeasyPrint finds
Homebrew's Pango. If you call `uv run cv generate` directly, export it yourself.

### Photos

The generator writes one prompt per candidate to `data/photo_prompts.json`. Generate headshots with
any image model and save them as `data/photos/<candidate_id>.png` (jpg/webp also work), then
`uv run cv generate --pdf-only`. Without a file the PDF gets a placeholder with initials, so the
pipeline never blocks on image generation.

## Example session

```
$ uv run cv chat -q "Who would be the best fit for a senior ML role?"
  → search_candidates(query='senior ML role', seniority=['senior', 'lead', 'principal'],
                      role_family=['ml', 'data-science'], k=10) → 2 result(s)
  → get_candidate(candidate_id='p02-rosalinda-barral') → 1 result(s)
  → get_candidate(candidate_id='p10-katie-brown') → 1 result(s)

Rosalinda Barral – Senior Machine Learning Engineer (8 years) ... strongest match.
Katie Brown – Principal NLP Researcher (15 years) ... research-leaning alternative.

$ uv run cv chat -q "Who speaks Hebrew?"
  → search_candidates(languages=['Hebrew'], k=20) → 0 result(s)

No candidate in the dataset lists Hebrew as a spoken language.
```

## How it works

- `generate/personas.py` — 14 hand-written persona specs; Faker (seeded, per-country locale) supplies
  names and contacts. `generate/generator.py` asks the LLM for the narrative as JSON matching the
  Pydantic schema in `generate/schema.py`, validating and retrying on schema errors.
- `generate/render.py` — Jinja2 templates (`classic`, `modern`, `minimal`) rendered by WeasyPrint;
  `generate/themes.py` derives colours, fonts, photo shape, skill style and date format per candidate.
- `index/store.py` — one Chroma document per candidate: the CV text embedded with `bge-small`
  (fastembed, ONNX, local) plus scalar metadata and boolean flags per language and skill variant.
  `search(query, filters, k)` covers filter-only, semantic and hybrid retrieval; `get` and
  `find_by_name` complete the boundary the agent uses.
- `agent/tools.py`, `agent/loop.py` — three tools in OpenAI function-calling format and a
  hand-written loop over the OpenRouter API (no framework). After each answer the names mentioned
  are checked against the ids the tools returned; anything else is flagged as ungrounded.
- `llm.py` — OpenRouter client with a model fallback chain and a JSON→Pydantic helper.
- `evals/` — `cases.json` (expected/forbidden candidates, no-match cases) and `run.py`.
- `demo.ipynb` — guided walkthrough of every stage with diagrams and live calls (`uv run jupyter lab demo.ipynb`).

## Layout

```
src/cv_screener/
  generate/   personas, schema, generator, render, themes, photos
  index/      embeddings, store
  agent/      tools, loop
  templates/  classic.html, modern.html, minimal.html, _theme.html, _skills.html
  cli.py      cv generate | index | search | chat
  llm.py      OpenRouter client
evals/        cases.json, run.py
tests/        offline pytest suite
data/         candidates/*.json, pdf/*.pdf, photos/, photo_prompts.json, chroma/ (ignored)
```

See `NOTES.md` for what is unfinished, known issues and eval results, and `ROADMAP.md` for the plan.
