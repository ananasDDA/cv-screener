# macOS: WeasyPrint needs Homebrew's pango/cairo visible to dyld.
export DYLD_FALLBACK_LIBRARY_PATH := /opt/homebrew/lib:$(DYLD_FALLBACK_LIBRARY_PATH)

# Optional local overrides (git-ignored), e.g. UV_PROJECT_ENVIRONMENT.
-include local.mk

.PHONY: setup generate index chat web test eval check

setup:
	uv sync

generate:
	uv run cv generate

index:
	uv run cv index

chat:
	uv run cv chat

web:
	uv run cv web

test:
	uv run pytest -q

eval:
	uv run python evals/run.py

check:
	uv run ruff check . && uv run ruff format --check . && uv run pytest -q
