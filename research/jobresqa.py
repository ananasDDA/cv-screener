"""Part 3: the same A/B comparison on an external benchmark.

JobResQA (Avature, CC BY-SA 2.0, arXiv:2601.23183) is 581 QA pairs over 105 synthetic
résumé / job-description pairs in five languages. We take the English split.

The résumés are free text with `[NAME]`-style placeholders and do not fit the project's
`Candidate` schema, so nothing is parsed into it. Instead a separate, temporary Chroma
collection holds the raw résumé text embedded with the project's own `FastEmbedder`, and a
minimal tool loop (one `search_resumes` tool) plays arm A against an arm B that puts every
selected résumé into a single prompt. Answers are graded by an LLM judge against the
benchmark's `short_answer`.
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
from dataclasses import dataclass
from typing import Any

import chromadb
from faker import Faker

from cv_screener.config import settings
from cv_screener.index.embeddings import FastEmbedder
from cv_screener.llm import LLM

from . import cache

csv.field_size_limit(sys.maxsize)

TSV = cache.DATA_DIR / "jobresqa.en.tsv"
CHROMA_DIR = cache.DATA_DIR / "chroma_jobresqa"
COLLECTION = "jobresqa_en"
DATASET_URL = "https://github.com/Avature/jobresqa-benchmark"
PAPER_URL = "https://arxiv.org/abs/2601.23183"
LICENCE = "CC BY-SA 2.0, © 2025 Avature"

# Cut down from 13 résumés / 21 questions after the account's free-model day ran out: the
# post-reset allowance is 150 requests for the whole task, and this part has to fit inside it.
N_RESUMES = 10
QUOTA = {"Basic": 5, "Intermediate": 4, "Complex": 3}  # 12 questions, mixed complexity
PER_RESUME = 2
MAX_UNKNOWN = 2
TEMPERATURE = 0.2

#: Every JobResQA row pairs a résumé with a job description. Questions that lean on the JD need
#: that second document, and we are comparing retrieval over résumés, so they are dropped.
JD_REFERENCE = re.compile(
    r"job description|\bthe job\b|\bjob\b|this role|the position|the posting|\bJD\b"
    r"|align|requirement|required|overqualif",
    re.I,
)

SEARCH_TOOL: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_resumes",
            "description": (
                "Semantic search over the résumé collection. Returns the full text of the k "
                "best-matching résumés, each preceded by the candidate's name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Who or what to look for."},
                    "k": {"type": "integer", "description": "How many résumés, default 2, max 5"},
                },
                "required": ["query"],
            },
        },
    }
]

TOOLS_SYSTEM = """You answer questions about a collection of résumés.

You cannot see the collection. Call search_resumes first and answer only from what it returns.
The question names the person it is about, so search for that name.
Answer with the single fact asked for, in at most one short sentence. If the résumé does not
contain the answer, reply exactly: Unknown."""

PROMPT_SYSTEM = """You answer questions about a collection of résumés.

The full text of every résumé is included below; it is the only information you have.
Answer with the single fact asked for, in at most one short sentence. If the résumés do not
contain the answer, reply exactly: Unknown.

=== RESUMES ===
{corpus}
=== end of resumes ==="""

JUDGE_SYSTEM = """You grade one answer to a question about a résumé.

Question: {question}
Reference answer: {gold}
Answer to grade: {answer}

The answer is CORRECT when it states the same fact as the reference, even if worded
differently, more verbosely, or with extra correct detail. It is INCORRECT when it contradicts
the reference, omits the fact the reference gives, or asserts a specific fact where the
reference is "Unknown".

Reply with exactly one line, no markdown:
CORRECT|<reason in under 15 words>
or
INCORRECT|<reason in under 15 words>"""


@dataclass
class Item:
    example_id: str
    resume_id: str
    person: str
    question: str
    gold: str
    complexity: str


def available() -> bool:
    return TSV.exists()


def load_rows() -> list[dict[str, str]]:
    with TSV.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _names(resume_ids: list[str]) -> dict[str, str]:
    """A stable synthetic name per résumé, so questions can be phrased per person."""
    faker = Faker("en_US")
    Faker.seed(11)
    return {rid: faker.name() for rid in resume_ids}


def select(
    rows: list[dict[str, str]] | None = None,
) -> tuple[list[Item], dict[str, str], dict[str, str]]:
    """Deterministic selection: résumé-only questions, a mix of complexity levels.

    Returns the questions, {resume_id: résumé text with the synthetic name substituted} and
    {resume_id: synthetic name}. All 13 résumés go into the corpus, including any that ended
    up without a question: they are the distractors retrieval has to get past.
    """
    rows = rows or load_rows()
    usable = [r for r in rows if not JD_REFERENCE.search(r["question"])]
    by_resume: dict[str, list[dict[str, str]]] = {}
    for r in usable:
        by_resume.setdefault(r["resume_id"], []).append(r)
    ranked = sorted(
        by_resume,
        key=lambda rid: (
            -len({q["complexity_level"] for q in by_resume[rid]}),
            -len(by_resume[rid]),
            rid,
        ),
    )[:N_RESUMES]
    names = _names(sorted(ranked))

    picked: list[Item] = []
    unknown = 0
    per_resume: dict[str, int] = dict.fromkeys(ranked, 0)
    # Level by level, round-robin over résumés, so neither a level nor one résumé dominates.
    for level, quota in QUOTA.items():
        taken_here = 0
        for rid in ranked:
            if taken_here >= quota or per_resume[rid] >= PER_RESUME:
                continue
            chosen = {p.example_id for p in picked}
            for q in sorted(by_resume[rid], key=lambda q: q["example_id"]):
                if q["complexity_level"] != level or q["example_id"] in chosen:
                    continue
                is_unknown = q["short_answer"].strip().lower() == "unknown"
                if is_unknown and unknown >= MAX_UNKNOWN:
                    continue
                unknown += is_unknown
                per_resume[rid] += 1
                taken_here += 1
                picked.append(
                    Item(
                        q["example_id"],
                        rid,
                        names[rid],
                        _personalise(q["question"], names[rid]),
                        q["short_answer"].strip(),
                        level,
                    )
                )
                break
    texts = {rid: by_resume[rid][0]["resume"].replace("[NAME]", names[rid]) for rid in ranked}
    return picked, texts, names


def _personalise(question: str, name: str) -> str:
    """Turn "the candidate" into a named person so the question identifies one résumé."""
    q = re.sub(r"\bthe candidate['’]s\b", f"{name}'s", question, flags=re.I)
    q = re.sub(r"\bcandidates\b(?=\s)", f"{name}'s", q)
    q = re.sub(r"\bthe candidate\b", name, q, flags=re.I)
    q = re.sub(r"\bhis/her\b", "their", q, flags=re.I)
    return q if name in q else f"About {name}: {question}"


# ---- the temporary index ------------------------------------------------------------------


class ResumeIndex:
    """Raw résumé text in its own Chroma collection, embedded with the project's embedder."""

    def __init__(self, texts: dict[str, str], names: dict[str, str], rebuild: bool = False):
        self.texts, self.names = texts, names
        self.embedder = FastEmbedder()  # loading the ONNX model is slow; do it once
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        existing = {c.name for c in client.list_collections()}
        if rebuild and COLLECTION in existing:
            client.delete_collection(COLLECTION)
            existing.discard(COLLECTION)
        self.collection = client.get_or_create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine"}, embedding_function=None
        )
        if self.collection.count() != len(texts):
            ids = sorted(texts)
            docs = [f"{names[i]}\n{texts[i]}" for i in ids]
            self.collection.upsert(ids=ids, documents=docs, embeddings=self.embedder.embed(docs))

    def search(self, query: str, k: int = 2) -> list[dict[str, str]]:
        k = max(1, min(int(k or 2), 5))
        vec = self.embedder.embed([query])
        res = self.collection.query(query_embeddings=vec, n_results=k, include=["documents"])
        return [
            {"resume_id": i, "name": self.names[i], "resume": self.texts[i]} for i in res["ids"][0]
        ]


# ---- the two arms -------------------------------------------------------------------------


def run_tools_arm(llm: LLM, index: ResumeIndex, item: Item, max_steps: int = 3) -> dict[str, Any]:
    """Minimal tool loop: the project's `Agent` is bound to `Candidate`, so this is its
    stripped-down twin over raw text — same shape, one tool."""
    llm.reset_usage()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": TOOLS_SYSTEM},
        {"role": "user", "content": item.question},
    ]
    retrieved: list[str] = []
    tool_calls = 0
    t0 = time.time()
    text = ""
    for _ in range(max_steps):
        msg = llm.chat(messages, tools=SEARCH_TOOL, temperature=TEMPERATURE)
        if not msg.tool_calls:
            text = (msg.content or "").strip()
            break
        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.function.name, "arguments": c.function.arguments},
                    }
                    for c in msg.tool_calls
                ],
            }
        )
        for call in msg.tool_calls:
            args = _args(call.function.arguments)
            hits = index.search(args.get("query") or item.person, args.get("k") or 2)
            retrieved += [h["resume_id"] for h in hits]
            tool_calls += 1
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(hits, ensure_ascii=False),
                }
            )
    else:
        messages.append({"role": "user", "content": "Answer now using only the results above."})
        text = (llm.chat(messages, temperature=TEMPERATURE).content or "").strip()
    return _usage(
        llm,
        {
            "answer": text,
            "tool_calls": tool_calls,
            "retrieved": retrieved,
            "retrieval_hit": item.resume_id in retrieved,
            "seconds": round(time.time() - t0, 2),
            "model": llm.last_model,
        },
    )


def run_prompt_arm(llm: LLM, corpus: str, item: Item) -> dict[str, Any]:
    llm.reset_usage()
    t0 = time.time()
    msg = llm.chat(
        [
            {"role": "system", "content": PROMPT_SYSTEM.format(corpus=corpus)},
            {"role": "user", "content": item.question},
        ],
        temperature=TEMPERATURE,
    )
    return _usage(
        llm,
        {
            "answer": (msg.content or "").strip(),
            "tool_calls": 0,
            "retrieved": [],
            "retrieval_hit": True,  # every résumé is in the prompt by construction
            "seconds": round(time.time() - t0, 2),
            "model": llm.last_model,
        },
    )


def judge(llm: LLM, item: Item, answer: str) -> dict[str, Any]:
    """Binary grade with a one-line reason. Costs are billed to the judge, not to the arms."""
    llm.reset_usage()
    prompt = JUDGE_SYSTEM.format(question=item.question, gold=item.gold, answer=answer or "(empty)")
    msg = llm.chat([{"role": "user", "content": prompt}], temperature=0.0)
    verdict = (msg.content or "").strip().splitlines()[0] if msg.content else ""
    correct = verdict.upper().startswith("CORRECT")
    reason = verdict.split("|", 1)[1].strip() if "|" in verdict else verdict
    return {
        "correct": correct,
        "reason": reason[:160],
        "requests": llm.usage.requests,
        "judge_model": llm.last_model,
    }


def _usage(llm: LLM, row: dict[str, Any]) -> dict[str, Any]:
    u = llm.usage
    return row | {
        "llm_calls": u.calls,
        "prompt_tokens": u.prompt_tokens,
        "completion_tokens": u.completion_tokens,
        "total_tokens": u.total_tokens,
        "estimated": u.estimated,
        "requests": u.requests,
    }


def _args(raw: str | None) -> dict[str, Any]:
    try:
        return json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {}


# ---- collection ---------------------------------------------------------------------------


def corpus(texts: dict[str, str], names: dict[str, str]) -> str:
    return "\n\n---\n\n".join(f"{names[i]}\n{texts[i]}" for i in sorted(texts))


def collect(force: bool = False) -> list[dict[str, Any]]:
    name = "part3_runs"
    records = [] if force else cache.drop_quota_failures(name, cache.load(name))
    strikes = 0
    if not available():
        raise SystemExit(f"{TSV} is missing; download the English split first")
    items, texts, names = select()
    specs = [(arm, it.example_id) for arm in ("tools", "prompt") for it in items]
    todo = cache.missing(records, specs, ("arm", "example_id"))
    print(f"[budget] spent {cache.spent()} / {cache.BUDGET_LIMIT}; part3: {len(todo)} runs planned")
    if not todo:
        return records

    llm = LLM()
    index = ResumeIndex(texts, names)
    flat = corpus(texts, names)
    by_id = {it.example_id: it for it in items}

    for arm, example_id in todo:
        if cache.remaining() <= 0:
            print("[budget] stopping: limit reached")
            break
        item = by_id[example_id]
        error: str | None = None
        try:
            row = (
                run_tools_arm(llm, index, item)
                if arm == "tools"
                else run_prompt_arm(llm, flat, item)
            )
            grade = judge(LLM(settings()), item, row["answer"])
        except Exception as exc:  # noqa: BLE001 - a crash is a failed question, not a crashed run
            error = f"{type(exc).__name__}: {str(exc)[:300]}"
            row = _usage(
                llm,
                {
                    "answer": "",
                    "tool_calls": 0,
                    "retrieved": [],
                    "retrieval_hit": False,
                    "seconds": 0.0,
                    "model": llm.last_model,
                },
            )
            grade = {
                "correct": False,
                "reason": error[:160],
                "requests": 0,
                "judge_model": None,
            }
        # The judge's own request count must not shadow the arm's, so it moves out of `grade`
        # before the merge and is billed on its own line.
        judge_requests = grade.pop("requests", 0)
        rec = (
            row
            | grade
            | {
                "arm": arm,
                "example_id": example_id,
                "resume_id": item.resume_id,
                "person": item.person,
                "complexity": item.complexity,
                "question": item.question,
                "gold": item.gold,
                "judge_requests": judge_requests,
                "error": error,
            }
        )
        records.append(rec)
        cache.save(name, records)
        cache.charge(rec["requests"] + rec["judge_requests"], "part3")
        print(
            f"  {arm:7s} {example_id} {item.complexity:12s} "
            f"{'OK ' if rec['correct'] else 'BAD'} {rec['total_tokens']:6d} tok "
            f"{rec['seconds']:5.1f}s  gold={item.gold[:28]!r} got={rec['answer'][:40]!r}"
        )
        strikes = strikes + 1 if cache.is_rate_limited(rec) else 0
        if strikes >= 3:
            print("[quota] three 429s in a row; stopping rather than retrying into a wall")
            break
        time.sleep(1.5)
    return cache.drop_quota_failures(name, records)


def accuracy_table(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for arm, label in (("tools", "A: retrieve top-k"), ("prompt", "B: all résumés in prompt")):
        sub = [r for r in records if r["arm"] == arm]
        if not sub:
            continue
        tokens = sorted(r["total_tokens"] for r in sub)
        correct = sum(r["correct"] for r in sub)
        rows.append(
            {
                "arm": label,
                "questions": len(sub),
                "correct": correct,
                "accuracy": correct / len(sub),
                "median tokens": tokens[len(tokens) // 2],
                "median s": sorted(r["seconds"] for r in sub)[len(sub) // 2],
                "retrieval hit": f"{sum(r['retrieval_hit'] for r in sub)}/{len(sub)}",
            }
        )
    return rows


def by_complexity(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for level in ("Basic", "Intermediate", "Complex"):
        row: dict[str, Any] = {"complexity": level}
        for arm, tag in (("tools", "A"), ("prompt", "B")):
            sub = [r for r in records if r["arm"] == arm and r["complexity"] == level]
            row[f"{tag} correct"] = f"{sum(r['correct'] for r in sub)}/{len(sub)}" if sub else "—"
        rows.append(row)
    return rows
