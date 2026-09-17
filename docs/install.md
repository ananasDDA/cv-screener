# cv-screener — installation runbook for an AI agent

This document is addressed to an AI coding agent installing the **cv-screener** MCP server on a
user's machine. Follow the steps in order. Do not skip the safety rules.

Source repository: <https://github.com/ananasDDA/cv-screener>

## What you are installing

A local, stdio Model Context Protocol server that exposes a résumé/CV database with hybrid search
(exact metadata filters + semantic ranking over ChromaDB, embeddings computed locally with
`bge-small` through fastembed). No API key is required: the host's model is the agent. Nothing is
uploaded anywhere.

Tools exposed: `search_candidates`, `find_by_name`, `get_candidate`, `list_candidates`,
`add_candidate`, `remove_candidate`.

## Safety rules

1. **Ask the user before editing any configuration file.** Show them the exact file path and the
   exact change you intend to make, and wait for a clear yes.
2. **Back the file up first**, e.g. copy `claude_desktop_config.json` to
   `claude_desktop_config.json.bak` before writing.
3. **Merge, never overwrite.** If `mcpServers` already exists, add the `cv-screener` key to it and
   leave every other server untouched. Preserve the rest of the JSON exactly as it was.
4. Do not install anything else, and do not change unrelated settings.

## Step 1 — check the prerequisite

`uv` (with its `uvx` runner) is the only prerequisite. Python 3.12 is fetched by uv automatically.

```bash
uv --version
```

If the command is not found, install uv (see <https://docs.astral.sh/uv/>), asking the user first:

- macOS / Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- macOS with Homebrew: `brew install uv`
- Windows (PowerShell): `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

Then open a new shell and re-check `uv --version`.

## Step 2a — install for Claude Code

Run exactly this command:

```bash
claude mcp add cv-screener -- uvx --from git+https://github.com/ananasDDA/cv-screener cv mcp
```

## Step 2b — install for Claude Desktop

Config file path:

| OS | Path |
|---|---|
| macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Windows | `%APPDATA%\Claude\claude_desktop_config.json` |

The file is also reachable from the app: Settings → Developer → Edit Config.

Back the file up, then **merge** this entry into the existing `mcpServers` object:

```json
{
  "mcpServers": {
    "cv-screener": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/ananasDDA/cv-screener", "cv", "mcp"]
    }
  }
}
```

If the file does not exist yet, create it with exactly the JSON above.

If Claude Desktop reports that it cannot find `uvx`, replace `"uvx"` with its absolute path. Find it
with `which uvx` (macOS/Linux) or `where uvx` (Windows); a common macOS value is
`/opt/homebrew/bin/uvx`.

After writing the file, tell the user to **quit and reopen Claude Desktop** — a restart is required,
reloading the window is not enough.

## Step 3 — verify

For Claude Code:

```bash
claude mcp list
```

Expect a line like `cv-screener: uvx --from git+https://github.com/ananasDDA/cv-screener cv mcp - ✓ Connected`.

Notes on the first start:

- It takes roughly **30–60 seconds**. `uvx` resolves and installs the dependencies, and the server
  builds the ChromaDB index for the bundled candidates on first run.
- A **~130 MB embedding model** (`bge-small`) is downloaded once and cached; later starts are fast.
- If the client times out on the first attempt, wait for the install to finish and try again.

## Step 4 — tell the user what they must do themselves

In **Claude Desktop** the tools of a newly added local server stay switched off until the user
enables them in the chat's tools menu, and the app asks again whenever a tool's definition changes
(for example after an update). Say this explicitly; the installation is not finished until they do
it.

## Step 5 — smoke test

Ask the model to call the server, for example:

- "Which candidates speak Spanish?" → `search_candidates` with `languages: ["Spanish"]`
- "Show me the whole collection" → `list_candidates`

If a tool call returns results, the installation works.

## Privacy

Everything runs locally over stdio; no network service is started and no candidate data leaves the
machine. Résumés the user adds through `add_candidate` are stored in `~/.cv-screener/candidates`
(override with the `CV_SCREENER_HOME` environment variable), outside the repository, and are never
committed. Installed this way, the ChromaDB index is also written under `~/.cv-screener`.

## Uninstall

- Claude Code: `claude mcp remove cv-screener`
- Claude Desktop: remove the `cv-screener` key from `mcpServers`, then restart the app.
