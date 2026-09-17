# macOS: WeasyPrint needs Homebrew's pango/cairo visible to dyld.
export DYLD_FALLBACK_LIBRARY_PATH := /opt/homebrew/lib:$(DYLD_FALLBACK_LIBRARY_PATH)

.PHONY: setup generate index chat test eval

setup:
	uv sync

generate:
	uv run cv generate

index:
	uv run cv index

chat:
	uv run cv chat

test:
	uv run pytest -q

eval:
	uv run python evals/run.py
