# Roadmap

Scope of the test task first, extras after. Each phase ends with a green `make test` and a commit.

## Core (graded)

- [x] **0. Scaffold** — uv project, layout, `.env.example`, `CLAUDE.md`, Makefile.
- [x] **1. Candidate generation** — persona grid (14 specs), Faker identity, LLM narrative into a
      Pydantic schema, model fallback chain, `data/photo_prompts.json`.
- [x] **2. PDF rendering** — three Jinja2/WeasyPrint layouts (classic, modern, minimal), photo from
      `data/photos/<id>.png` or a generated placeholder, `cv generate` runs profiles + PDFs in one go.
- [x] **3. Index and search** — ChromaDB with local fastembed embeddings; structured fields as
      metadata (seniority, role, country, years, languages, skills); one `search(query, filters)`
      that combines metadata filters with semantic ranking; `cv index` and `cv search` commands.
- [ ] **4. Agent** — hand-written tool-use loop over OpenRouter; tools: `search_candidates`,
      `get_candidate`, `find_by_name`; grounding rules in the system prompt; `cv chat` CLI with a
      visible tool-call trace.
- [ ] **5. Evals** — `evals/cases.json` (8 cases incl. "no match" and "summarize <name>"),
      `evals/run.py` prints per-case pass/fail and a summary; results copied verbatim into NOTES.md.
- [ ] **6. Tests** — pytest without an API key: schema validation, persona uniqueness, index build +
      filtered search, agent loop against a fake LLM client.
- [ ] **7. Docs** — README (setup from zero, commands, architecture), NOTES.md (unfinished, broken,
      eval results, next steps), CLAUDE.md refresh.
- [ ] **8. Photos** — drop generated photos into `data/photos/`, re-render PDFs, review realism.
- [ ] **9. Demo notebook** — thin `demo.ipynb` that imports the package: open a PDF, run a filtered
      and a semantic search, show an agent trace, run the evals. This is the screen-share script.

## Extras (only after the core is complete)

- [ ] GitHub Actions: ruff + pytest on every push (no key needed).
- [ ] MCP server exposing the same three search tools.
- [ ] Web chat UI and a public deployment.

## Call prep

- Show `make generate` on one candidate, open the PDF.
- `cv search` with a metadata filter, then a pure semantic query.
- `cv chat` on the four sample questions with the tool trace visible.
- `make test` and `make eval`, compare with NOTES.md.
