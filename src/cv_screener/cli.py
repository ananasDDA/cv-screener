"""Command-line entrypoints. Each roadmap phase adds one command here."""

from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer(
    help="Synthetic CV dataset, hybrid search and a tool-using agent.", no_args_is_help=True
)
console = Console()


@app.callback()
def main() -> None:
    """Typer treats a single command as the root; this callback keeps subcommands."""


@app.command()
def generate(
    workers: int = typer.Option(3, help="Parallel LLM requests."),
    force: bool = typer.Option(False, "--force", help="Regenerate profiles that already exist."),
    pdf_only: bool = typer.Option(False, "--pdf-only", help="Skip the LLM, only re-render PDFs."),
) -> None:
    """Generate candidate profiles (LLM) and render one PDF per candidate."""
    from .generate.generator import generate_all, load_all
    from .generate.render import render_pdf

    cands = (
        load_all()
        if pdf_only
        else generate_all(workers=workers, only_missing=not force, log=console.print)
    )
    console.print(f"rendering {len(cands)} PDFs")
    for c in cands:
        path = render_pdf(c)
        console.print(f"  [green]pdf[/green]  {path.relative_to(path.parents[2])}  ({c.template})")


if __name__ == "__main__":
    app()
