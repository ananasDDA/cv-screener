"""How arm B's prompt grows with the size of the CV collection, and arm A's does not.

Nothing here calls a model. Synthetic collections are built by duplicating the 14 real
candidates and perturbing the copies (new identity, jittered years, rotated skills) so the
per-document length stays realistic; the prompt is then measured in characters and converted
with the same chars/4 estimator `llm.estimate_tokens` uses.

The estimator is calibrated against the *measured* arm B prompt tokens at N = 14, and the
calibration error is reported rather than hidden.
"""

from __future__ import annotations

from typing import Any

from faker import Faker

from cv_screener.generate.schema import Candidate
from cv_screener.llm import estimate_tokens

from .harness import baseline_prompt, corpus_text

SIZES = (14, 50, 100, 200)

#: Context windows worth marking on the plot. The models in `.env` sit at 262k and 1M, but a
#: prompt that large is priced and latency-bound long before it is refused.
CONTEXT_LIMITS = {"32k": 32_000, "128k": 128_000, "262k (models used here)": 262_144}

TYPICAL_QUESTION = "Which candidates speak Spanish?"


def grow(candidates: list[Candidate], n: int, seed: int = 7) -> list[Candidate]:
    """The real candidates first, then perturbed copies until the list has `n` entries."""
    faker = Faker()
    Faker.seed(seed)
    out = list(candidates[:n])
    while len(out) < n:
        base = candidates[len(out) % len(candidates)]
        shift = len(out) % max(len(base.skills), 1)
        out.append(
            base.model_copy(
                update={
                    "id": f"{base.id}-copy{len(out)}",
                    "full_name": faker.name(),
                    "years_experience": max(1, base.years_experience + (len(out) % 7) - 3),
                    "skills": base.skills[shift:] + base.skills[:shift],
                }
            )
        )
    return out


def prompt_size_curve(candidates: list[Candidate], sizes: tuple[int, ...] = SIZES) -> list[dict]:
    """Estimated arm B prompt size per question for each collection size."""
    rows = []
    for n in sizes:
        prompt = baseline_prompt(corpus_text(grow(candidates, n))) + TYPICAL_QUESTION
        rows.append(
            {
                "N documents": n,
                "prompt chars": len(prompt),
                "est. prompt tokens": estimate_tokens(prompt),
            }
        )
    return rows


def calibration(candidates: list[Candidate], measured_prompt_tokens: float) -> dict[str, Any]:
    """Compare the chars/4 estimate at N = 14 with what the provider actually billed."""
    estimated = prompt_size_curve(candidates, (len(candidates),))[0]["est. prompt tokens"]
    return {
        "N": len(candidates),
        "estimated (chars/4)": estimated,
        "measured (provider usage)": round(measured_prompt_tokens),
        "estimate / measured": round(estimated / measured_prompt_tokens, 3)
        if measured_prompt_tokens
        else None,
    }
