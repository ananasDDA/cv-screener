# CV Screener

Synthetic CV dataset, hybrid vector search and a tool-using CLI agent. Python 3.12.

_Work in progress. Full setup instructions will land here._

## System requirements
- Python 3.12 and [uv](https://docs.astral.sh/uv/).
- Pango/Cairo for PDF rendering (WeasyPrint):
  - macOS: `brew install pango` (the Makefile exports `DYLD_FALLBACK_LIBRARY_PATH` so dyld finds it).
  - Debian/Ubuntu: `apt install libpango-1.0-0 libpangoft2-1.0-0`.
