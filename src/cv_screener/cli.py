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


@app.command()
def index() -> None:
    """Rebuild the vector index from data/candidates."""
    from .generate.generator import load_all
    from .index.store import CandidateIndex

    cands = load_all()
    n = CandidateIndex().rebuild(cands)
    console.print(f"indexed {n} candidates")


@app.command()
def search(
    query: str = typer.Argument("", help="Free-text query; empty for filter-only."),
    seniority: list[str] = typer.Option([], "--seniority", "-s"),
    role: list[str] = typer.Option([], "--role", "-r"),
    country: list[str] = typer.Option([], "--country", "-c"),
    language: list[str] = typer.Option([], "--language", "-l"),
    skill: list[str] = typer.Option([], "--skill"),
    min_years: int | None = typer.Option(None, "--min-years"),
    k: int = typer.Option(5, "-k"),
) -> None:
    """Hybrid search: metadata filters plus semantic ranking."""
    from rich.table import Table

    from .index.store import CandidateIndex, Filters

    filters = Filters(
        seniority=seniority,
        role_family=role,
        country=country,
        languages=language,
        skills=skill,
        min_years=min_years,
    )
    hits = CandidateIndex().search(query or None, filters, k=k)
    table = Table(title=f"query={query!r} filters={filters.to_where()}")
    for col in ("score", "id", "name", "headline", "level", "role", "country", "yrs", "languages"):
        table.add_column(col)
    for h in hits:
        table.add_row(
            "" if h.score is None else f"{h.score:.3f}",
            h.id,
            h.name,
            h.headline[:50],
            h.seniority,
            h.role_family,
            h.country,
            str(h.years_experience),
            h.languages,
        )
    console.print(table if hits else "[yellow]no candidates match[/yellow]")


@app.command()
def chat(
    question: str | None = typer.Option(None, "--once", "-q", help="Ask one question and exit."),
    trace: bool = typer.Option(True, "--trace/--no-trace", help="Show tool calls as they happen."),
) -> None:
    """Chat with the agent. It answers only from what the search tools return."""
    from .agent.loop import Agent, ToolTrace
    from .index.store import CandidateIndex
    from .llm import LLM, LLMError

    try:
        llm = LLM()
    except LLMError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None
    index = CandidateIndex()
    if index.count() == 0:
        console.print("[red]index is empty; run `uv run cv index` first[/red]")
        raise typer.Exit(1)
    agent = Agent(llm, index)

    def show_tool(t: ToolTrace) -> None:
        if trace:
            args = ", ".join(f"{k}={v!r}" for k, v in t.args.items())
            console.print(f"  [dim]→ {t.name}({args}) → {t.count} result(s)[/dim]")

    def answer(q: str, history: list) -> None:
        a = agent.ask(q, history, on_tool=show_tool)
        console.print(f"\n{a.text}\n")
        if not a.grounded:
            console.print(
                f"[red]⚠ names not returned by any tool: {', '.join(a.ungrounded_names)}[/red]"
            )
        if trace:
            console.print(f"[dim]model: {a.model}[/dim]")
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": a.text}]

    history: list = []
    if question:
        answer(question, history)
        return
    console.print("[bold]CV Screener[/bold] — ask about the candidates. Ctrl-D or 'exit' to quit.")
    while True:
        try:
            q = console.input("[bold cyan]you>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in {"exit", "quit"}:
            break
        answer(q, history)


if __name__ == "__main__":
    app()
