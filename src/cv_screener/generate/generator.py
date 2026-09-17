"""Turn persona specs into full Candidate records using the LLM, and write them to disk."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

from ..config import CANDIDATES_DIR, PHOTO_PROMPTS_FILE
from ..llm import LLM
from .personas import Persona, personas
from .schema import Candidate, GeneratedProfile, Language

SYSTEM = """You write realistic, specific CV content for synthetic software-industry candidates.
The output must be indistinguishable from a real CV: plausible real companies for the given city,
consistent dates, concrete responsibilities with numbers, tools named precisely, no placeholders,
no clichés like "passionate" or "results-driven". Write in English. Do not mention the person's name."""


def build_prompt(p: Persona) -> str:
    today = date.today().strftime("%Y-%m")
    return f"""Today is {today}. Write the CV content for this person:

- Title: {p.title} ({p.seniority}), {p.years_experience} years of total experience, currently employed or recently left.
- Role family: {p.role_family}. Core stack: {", ".join(p.stack)} (use these, plus a few natural extras).
- Location: {p.city}, {p.country}. Age {p.age}: education and career start must be consistent with that.
- Education: {p.education_hint}.
- Career shape: {p.company_hint}. Use real companies that exist in that market.
- Positions: most recent first; the most recent one should end at present (end = null).
- Bullet points: 3-5 per position, each with a concrete outcome (latency, revenue, users, scale, team size).
{p.extra}"""


def generate_one(llm: LLM, p: Persona) -> Candidate:
    profile = llm.structured(SYSTEM, build_prompt(p), GeneratedProfile)
    return Candidate(
        **profile.model_dump(),
        id=p.id,
        full_name=p.full_name,
        gender=p.gender,
        age=p.age,
        city=p.city,
        country=p.country,
        email=p.email,
        phone=p.phone,
        linkedin=p.linkedin,
        github=p.github,
        languages=[Language(name=n, level=lvl) for n, lvl in p.languages],
        role_family=p.role_family,
        seniority=p.seniority,
        years_experience=p.years_experience,
        template=p.template,
        photo_prompt=p.photo_prompt(),
    )


def save(c: Candidate, out_dir: Path = CANDIDATES_DIR) -> Path:
    path = out_dir / f"{c.id}.json"
    path.write_text(c.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_all(dir_: Path = CANDIDATES_DIR) -> list[Candidate]:
    return [Candidate.model_validate_json(p.read_text()) for p in sorted(dir_.glob("*.json"))]


def write_photo_prompts(cands: list[Candidate], path: Path = PHOTO_PROMPTS_FILE) -> None:
    path.write_text(json.dumps([{"id": c.id, "prompt": c.photo_prompt} for c in cands], indent=2))


def generate_all(workers: int = 3, only_missing: bool = True, log=print) -> list[Candidate]:
    llm = LLM()
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    todo = [
        p for p in personas() if not (only_missing and (CANDIDATES_DIR / f"{p.id}.json").exists())
    ]
    log(
        f"generating {len(todo)} candidates with {llm.cfg.model} (+{len(llm.cfg.fallback_models)} fallbacks)"
    )
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(generate_one, llm, p): p for p in todo}
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                c = fut.result()
                save(c)
                log(f"  ok   {c.id:40s} {c.headline} [{llm.last_model}]")
            except Exception as exc:  # noqa: BLE001
                log(f"  FAIL {p.id}: {exc}")
    cands = load_all()
    write_photo_prompts(cands)
    return cands
