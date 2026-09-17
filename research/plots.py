"""Matplotlib figures for the notebook.

One measure per axis, no dual scales, a legend whenever two series share a panel and direct
labels on the bars so identity never rests on colour alone. Colours are the first slots of the
validated categorical palette; the figure surface is pinned to white so the charts read the
same in a light and a dark notebook theme.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

SERIES = {"tools": "#2a78d6", "prompt": "#eb6834"}
LABEL = {"tools": "A: tools (agent)", "prompt": "B: all CVs in the prompt"}
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#ffffff"


def _canvas(width: float, height: float) -> tuple[Figure, Any]:
    fig, ax = plt.subplots(figsize=(width, height), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#00000022")
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    return fig, ax


def _finish(ax: Any, title: str, subtitle: str = "") -> None:
    """Title above a muted subtitle, both flush left. The offset is in points, not axes
    fractions, so it does not shrink as a figure grows taller."""
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=26 if subtitle else 8)
    if subtitle:
        ax.annotate(
            subtitle,
            xy=(0, 1),
            xycoords="axes fraction",
            textcoords="offset points",
            xytext=(0, 8),
            color=MUTED,
            fontsize=9,
            va="bottom",
        )
    plt.tight_layout()


def _legend_below(ax: Any, ncol: int = 2) -> None:
    """Legends go under the axes: inside a dense panel every corner collides with a mark."""
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=ncol,
        fontsize=9,
        labelcolor=MUTED,
    )


def tokens_by_question(rows: list[dict[str, Any]]) -> Figure:
    """Median total tokens per question, one bar per arm. Measured from provider `usage`."""
    cases = [r["case"] for r in rows]
    y = range(len(cases))
    fig, ax = _canvas(8.6, 0.52 * len(cases) + 2.2)
    h = 0.36
    for offset, key, col in ((0.2, "A tok", "tools"), (-0.2, "B tok", "prompt")):
        values = [r.get(key) or 0 for r in rows]
        bars = ax.barh(
            [i + offset for i in y], values, height=h, color=SERIES[col], label=LABEL[col], zorder=3
        )
        ax.bar_label(bars, fmt="%.0f", padding=3, color=MUTED, fontsize=8)
    ax.set_yticks(list(y), cases)
    ax.set_xlabel(
        "median total tokens per question (prompt + completion, all LLM calls)",
        color=MUTED,
        fontsize=9,
    )
    ax.set_xlim(0, max(max(r.get("B tok") or 0, r.get("A tok") or 0) for r in rows) * 1.18)
    ax.grid(axis="x", color="#00000012", zorder=0)
    _legend_below(ax)
    _finish(ax, "Tokens per question", "measured; 3 repetitions per arm, median shown")
    return fig


def latency_by_arm(records: list[dict[str, Any]]) -> Figure:
    """Wall-clock distribution per arm: box plot plus every individual run."""
    fig, ax = _canvas(8.0, 3.2)
    arms = ["tools", "prompt"]
    data = [[r["seconds"] for r in records if r["arm"] == a] for a in arms]
    box = ax.boxplot(data, vert=False, widths=0.45, patch_artist=True, showfliers=False)
    for patch, arm in zip(box["boxes"], arms, strict=True):
        patch.set(facecolor=SERIES[arm] + "44", edgecolor=SERIES[arm], linewidth=2)
    for part in ("whiskers", "caps", "medians"):
        for artist in box[part]:
            artist.set(color=INK, linewidth=1.4)
    for i, (arm, values) in enumerate(zip(arms, data, strict=True), start=1):
        ax.scatter(
            values,
            [i] * len(values),
            s=18,
            color=SERIES[arm],
            alpha=0.75,
            zorder=4,
            edgecolors=SURFACE,
            linewidths=0.8,
        )
    ax.set_yticks([1, 2], [LABEL[a] for a in arms])
    ax.set_xlabel("seconds per question (wall clock)", color=MUTED, fontsize=9)
    ax.grid(axis="x", color="#00000012", zorder=0)
    _finish(ax, "Latency by arm", "box = quartiles, line = median, dots = all 27 runs")
    return fig


def pass_rate_by_arm(records: list[dict[str, Any]]) -> Figure:
    fig, ax = _canvas(7.0, 2.2)
    arms = ["tools", "prompt"]
    rates, labels = [], []
    for arm in arms:
        sub = [r for r in records if r["arm"] == arm]
        rates.append(sum(r["passed"] for r in sub) / len(sub) if sub else 0)
        labels.append(f"{sum(r['passed'] for r in sub)}/{len(sub)}")
    bars = ax.barh(
        [LABEL[a] for a in arms], rates, height=0.5, color=[SERIES[a] for a in arms], zorder=3
    )
    ax.bar_label(
        bars,
        labels=[f"{r:.0%}  ({lab})" for r, lab in zip(rates, labels, strict=True)],
        padding=6,
        color=INK,
        fontsize=10,
    )
    ax.set_xlim(0, 1.25)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="x", color="#00000012", zorder=0)
    _finish(ax, "Pass rate by arm", "same 9 eval cases, same checker, 3 repetitions")
    return fig


def scaling(
    curve: list[dict[str, Any]], measured_a_prompt: float, limits: dict[str, int]
) -> Figure:
    """Computed arm B prompt growth against arm A's measured, flat prompt size."""
    fig, ax = _canvas(8.4, 4.4)
    xs = [r["N documents"] for r in curve]
    ys = [r["est. prompt tokens"] for r in curve]
    ax.plot(
        xs,
        ys,
        marker="o",
        markersize=7,
        linewidth=2,
        color=SERIES["prompt"],
        label="B: all CVs in the prompt — computed (chars/4)",
        zorder=3,
    )
    ax.plot(
        xs,
        [measured_a_prompt] * len(xs),
        marker="o",
        markersize=7,
        linewidth=2,
        linestyle="--",
        color=SERIES["tools"],
        label="A: tools — measured median, independent of N",
        zorder=3,
    )
    for name, value in limits.items():
        ax.axhline(value, color="#52514e", linewidth=1, linestyle=":", zorder=1)
        ax.text(xs[0], value * 1.07, f"context limit {name}", color=MUTED, fontsize=8, ha="left")
    for x, y in zip(xs, ys, strict=True):
        ax.annotate(
            f"{y:,}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=8,
            color=MUTED,
        )
    ax.annotate(
        f"{measured_a_prompt:,.0f}",
        (xs[len(xs) // 2], measured_a_prompt),
        textcoords="offset points",
        xytext=(0, 9),
        ha="center",
        fontsize=8,
        color=MUTED,
    )
    ax.set_yscale("log")
    ax.set_ylim(measured_a_prompt * 0.45, max(ys) * 3.2)
    ax.set_xlabel("documents in the collection (N)", color=MUTED, fontsize=9)
    ax.set_ylabel("prompt tokens per question (log)", color=MUTED, fontsize=9)
    ax.grid(color="#00000012", zorder=0)
    _legend_below(ax)
    _finish(
        ax,
        "Prompt size against collection size",
        "dashed = measured on 14 real CVs; solid = computed from synthetic collections",
    )
    return fig


def leaderboard(rows: list[dict[str, Any]]) -> Figure:
    """Pass rate per model. One series, so no legend: the title names the measure."""
    rows = list(reversed(rows))
    names = [r["model"].replace(":free", "") for r in rows]
    fig, ax = _canvas(9.2, 0.5 * len(rows) + 1.8)
    bars = ax.barh(
        names, [r["pass rate"] for r in rows], height=0.55, color=SERIES["tools"], zorder=3
    )
    ax.bar_label(
        bars,
        labels=[
            f"{r['pass rate']:.0%}  ({r['passed']}"
            + (f", {r['errors']} err" if r["errors"] else "")
            + ")"
            for r in rows
        ],
        padding=6,
        color=INK,
        fontsize=9,
    )
    ax.set_xlim(0, 1.35)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="x", color="#00000012", zorder=0)
    _finish(
        ax,
        "Free models on the 9 eval cases (arm A, no fallback)",
        "one day, 2 repetitions per model — a snapshot, not a ranking",
    )
    return fig


def external_arms(rows: list[dict[str, Any]]) -> Figure:
    """Accuracy and median tokens per arm on the external benchmark, as two panels."""
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.0), facecolor=SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=9, length=0)
        ax.grid(axis="y", color="#00000012", zorder=0)
    arms = [
        r["arm"].split(": ")[0] + (": top-k" if r["arm"].startswith("A") else ": all") for r in rows
    ]
    colours = [SERIES["tools"] if a.startswith("A") else SERIES["prompt"] for a in arms]
    b1 = axes[0].bar(arms, [r["accuracy"] for r in rows], width=0.5, color=colours, zorder=3)
    axes[0].bar_label(
        b1,
        fmt="%.0f%%",
        labels=[f"{r['accuracy']:.0%}" for r in rows],
        padding=4,
        color=INK,
        fontsize=10,
    )
    axes[0].set_ylim(0, 1.15)
    axes[0].set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    axes[0].set_title("judge accuracy", color=INK, fontsize=11, loc="left")
    b2 = axes[1].bar(arms, [r["median tokens"] for r in rows], width=0.5, color=colours, zorder=3)
    axes[1].bar_label(b2, fmt="%.0f", padding=4, color=INK, fontsize=10)
    axes[1].set_title("median tokens per question", color=INK, fontsize=11, loc="left")
    plt.tight_layout()
    return fig
