# CV Screener

Test task: generate a realistic synthetic CV dataset, index it into a vector DB with structured
fields, and answer questions about it through a tool-using CLI agent.

## Layout
- `src/cv_screener/generate/` — persona grid, LLM profile generation (Pydantic schema), HTML→PDF rendering.
- `src/cv_screener/index/` — ChromaDB collection, local embeddings (fastembed), hybrid search (metadata filters + semantic).
- `src/cv_screener/agent/` — hand-written tool-use loop over the OpenAI-compatible OpenRouter API; tools call the index.
- `src/cv_screener/cli.py` — Typer entrypoints: `generate`, `index`, `chat`.
- `evals/` — JSON cases + runner that prints pass/fail and a summary.
- `tests/` — pytest, must pass without any API key.
- `data/` — generated candidates (JSON), PDFs, photos, Chroma store (ignored).

## Rules
- Python 3.12, managed with `uv`. Run things with `uv run ...`.
- LLM access only through OpenRouter (`OPENROUTER_API_KEY`); models are configured in `.env`, never hardcoded.
- No API keys in the repo. `.env.example` lists every variable.
- Tests never hit the network: mock the LLM client, use the local embedding model.
- The agent must retrieve through tools; never load the whole dataset into the prompt.
- Answers cite candidate names/ids that came back from tools. No match → say so.
- Photos: the generator looks for `data/photos/<candidate_id>.png`; if absent it draws a deterministic
  placeholder avatar so the pipeline never blocks on image generation.
- Commit messages in English, imperative mood, one logical change per commit.
- Keep NOTES.md honest: what is unfinished, what is broken, real eval results.
