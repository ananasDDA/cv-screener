"""Aggregation over raw run records. No plotting, no I/O."""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

ARM_LABEL = {"tools": "A: tools (agent)", "prompt": "B: all CVs in the prompt"}


def median(values: Sequence[float]) -> float:
    return round(statistics.median(values), 1) if values else float("nan")


def p95(values: Sequence[float]) -> float:
    """Nearest-rank p95. With 27 samples per arm that is the 26th value; no interpolation."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    rank = max(1, int(-(-0.95 * len(ordered) // 1)))
    return round(float(ordered[rank - 1]), 1)


def group_by(records: Iterable[dict[str, Any]], *fields: str) -> dict[tuple, list[dict[str, Any]]]:
    out: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        out[tuple(r[f] for f in fields)].append(r)
    return dict(out)


def arm_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per arm: pass rate, token and latency percentiles, cost per correct answer."""
    rows = []
    for arm, recs in sorted(group_by(records, "arm").items()):
        tokens = [r["total_tokens"] for r in recs]
        prompts = [r["prompt_tokens"] for r in recs]
        seconds = [r["seconds"] for r in recs]
        passed = [r for r in recs if r["passed"]]
        rows.append(
            {
                "arm": ARM_LABEL.get(arm[0], arm[0]),
                "runs": len(recs),
                "pass rate": f"{len(passed)}/{len(recs)} = {len(passed) / len(recs):.0%}",
                "median prompt tok": median(prompts),
                "median total tok": median(tokens),
                "p95 total tok": p95(tokens),
                "median s": median(seconds),
                "p95 s": p95(seconds),
                "tok / correct answer": round(sum(tokens) / len(passed)) if passed else None,
                "LLM calls (median)": median([r["llm_calls"] for r in recs]),
                "errors": sum(1 for r in recs if r["error"]),
                "estimated rows": sum(1 for r in recs if r["estimated"]),
            }
        )
    return rows


def per_question(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per case with both arms side by side."""
    rows = []
    for (case_id,), recs in group_by(records, "case_id").items():
        row: dict[str, Any] = {"case": case_id}
        for arm in ("tools", "prompt"):
            sub = [r for r in recs if r["arm"] == arm]
            if not sub:
                continue
            tag = "A" if arm == "tools" else "B"
            row[f"{tag} pass"] = f"{sum(r['passed'] for r in sub)}/{len(sub)}"
            row[f"{tag} tok"] = median([r["total_tokens"] for r in sub])
            row[f"{tag} s"] = median([r["seconds"] for r in sub])
        row["models"] = ", ".join(
            sorted({(r["model"] or "-").split("/")[-1].replace(":free", "") for r in recs})
        )
        fails = [r for r in recs if not r["passed"]]
        row["failure reasons"] = "; ".join(
            sorted({f"{r['arm']}: {reason}" for r in fails for reason in r["reasons"]})
        )[:160]
        rows.append(row)
    return sorted(rows, key=lambda r: r["case"])


def case_stability(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-case pass rate and tool-call spread for one arm's repetitions."""
    rows = []
    for (case_id,), recs in group_by(records, "case_id").items():
        tools = [r["tool_calls"] for r in recs]
        rows.append(
            {
                "case": case_id,
                "runs": len(recs),
                "passed": sum(r["passed"] for r in recs),
                "pass rate": f"{sum(r['passed'] for r in recs) / len(recs):.0%}",
                "tool calls": f"{min(tools)}–{max(tools)}" if tools else "-",
                "median tool calls": median(tools),
                "s (median)": median([r["seconds"] for r in recs]),
                "models": ", ".join(sorted({r["model"] or "-" for r in recs})),
            }
        )
    return sorted(rows, key=lambda r: r["case"])


def leaderboard(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per model, sorted by pass rate then median latency."""
    rows = []
    for (model,), recs in group_by(records, "requested_model").items():
        throttled = [r for r in recs if _is_429(r)]
        # A 429 never reached the model, so it is not evidence about the model: it is excluded
        # from the denominator and reported in its own column.
        answered = [r for r in recs if not _is_429(r)]
        ok = [r for r in answered if not r["error"]]
        passed = sum(r["passed"] for r in answered)
        rate = passed / len(answered) if answered else None
        rows.append(
            {
                "model": model,
                "runs": len(recs),
                "answered": len(answered),
                "pass rate": rate,  # numeric, for the chart; None when nothing got through
                "pass %": f"{rate:.0%}" if rate is not None else "not measured",
                "passed": f"{passed}/{len(answered)}" if answered else "0/0",
                "rate limits": len(throttled),
                "errors": sum(1 for r in answered if r["error"]),
                "median tok": median([r["total_tokens"] for r in ok]),
                "median s": median([r["seconds"] for r in ok]),
                "median tool calls": median([r["tool_calls"] for r in ok]),
                "ungrounded": sum(1 for r in answered if r["extra"].get("grounded") is False),
            }
        )
    # Models that never answered sort last, whatever their (absent) pass rate.
    return sorted(
        rows, key=lambda r: (r["pass rate"] is None, -(r["pass rate"] or 0), r["median s"])
    )


def _is_429(record: dict[str, Any]) -> bool:
    error = record.get("error") or ""
    return "429" in error or "RateLimit" in error
