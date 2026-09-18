"""External check of the semantic layer on real résumés: Kaggle "Resume Dataset".

2 484 real résumés in 24 job categories (snehaanbhawal, CC0). No LLM is involved: the same local
embedder the product uses (bge-small via fastembed) is evaluated on what it is there for, putting
similar résumés close together. The data is real people's text, so it stays in the git-ignored
`research/data/`; only aggregate results are cached under `research/results/`.

Leakage control: a résumé starts with its job title, which often contains the category word
("HR ADMINISTRATOR" in category HR). Every metric is therefore computed twice, on the full text and
on text with the title line removed and the category words masked.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "data" / "kaggle-resume"
CSV_PATH = DATA / "Resume" / "Resume.csv"
RESULTS = Path(__file__).parent / "results" / "kaggle_resume.json"
URL = "https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset"
CHUNK_CHARS = 1600  # ~400 tokens, under bge-small's 512-token window
BATCH = 32  # long chunks: big batches blow up attention memory and run ~6x slower on CPU
MAX_CHUNKS = 3  # first ~4 800 characters: covers most of a median résumé (5 900 chars)


def load() -> tuple[list[str], list[str]]:
    csv.field_size_limit(sys.maxsize)
    with CSV_PATH.open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["Resume_str"].strip()]
    return [r["Resume_str"] for r in rows], [r["Category"] for r in rows]


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_leak(text: str, category: str) -> str:
    """Drop the title (the text before the first section heading) and mask the category words."""
    body = re.split(r"\s{3,}", text.strip(), maxsplit=1)
    body = body[1] if len(body) > 1 else body[0]
    for w in re.split(r"[-\s]+", category.lower()):
        # short labels ("HR", "BPO") are masked as whole words only, longer ones with their endings
        pattern = rf"\b{re.escape(w)}\b" if len(w) <= 3 else rf"\b{re.escape(w)}\w*"
        body = re.sub(pattern, " ", body, flags=re.I)
    return normalise(body)


def chunks(text: str) -> list[str]:
    text = normalise(text)
    parts = [text[i : i + CHUNK_CHARS] for i in range(0, len(text), CHUNK_CHARS)]
    return parts[:MAX_CHUNKS] or [""]


def embed(texts: list[str], cache: Path, embedder=None) -> np.ndarray:
    """Mean-pooled chunk embeddings, L2-normalised. Cached as .npy next to the data."""
    if cache.exists():
        return np.load(cache)
    if embedder is None:
        from cv_screener.index.embeddings import FastEmbedder

        embedder = FastEmbedder()
    spans, flat = [], []
    for t in texts:
        c = chunks(t)
        spans.append((len(flat), len(flat) + len(c)))
        flat.extend(c)
    vecs = np.zeros((len(flat), 0), dtype=np.float32)
    parts = []
    for start in range(0, len(flat), BATCH):
        parts.append(np.array(embedder.embed(flat[start : start + BATCH]), dtype=np.float32))
        print(
            f"  embedded {min(start + BATCH, len(flat))}/{len(flat)} chunks",
            file=sys.stderr,
            flush=True,
        )
    vecs = np.concatenate(parts)
    doc = np.stack([vecs[a:b].mean(axis=0) for a, b in spans])
    doc /= np.linalg.norm(doc, axis=1, keepdims=True) + 1e-9
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, doc)
    return doc


def knn_predict(train_x, train_y, test_x, k: int = 5) -> list[str]:
    sims = test_x @ train_x.T
    top = np.argsort(-sims, axis=1)[:, :k]
    out = []
    for row, idx in zip(sims, top, strict=True):
        votes: dict[str, float] = {}
        for j in idx:
            votes[train_y[j]] = votes.get(train_y[j], 0.0) + float(row[j])
        out.append(max(votes, key=votes.get))
    return out


def neighbour_precision(x: np.ndarray, y: list[str], k: int = 5) -> float:
    """'Find similar candidates': share of the k nearest résumés that are in the same category."""
    sims = x @ x.T
    np.fill_diagonal(sims, -1)
    top = np.argsort(-sims, axis=1)[:, :k]
    labels = np.array(y)
    return float((labels[top] == labels[:, None]).mean())


def query_precision(x: np.ndarray, y: list[str], embedder, k: int = 10) -> dict[str, float]:
    """The product's use case: a recruiter types a role, the index returns k résumés."""
    cats = sorted(set(y))
    queries = [f"{c.replace('-', ' ').lower()} professional résumé" for c in cats]
    q = np.array(embedder.embed(queries), dtype=np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    top = np.argsort(-(q @ x.T), axis=1)[:, :k]
    labels = np.array(y)
    return {c: float((labels[top[i]] == c).mean()) for i, c in enumerate(cats)}


def evaluate(x: np.ndarray, y: list[str], texts: list[str], seed: int = 7) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline

    ya = np.array(y)
    scores: dict[str, list[tuple[float, float]]] = {"knn": [], "embed_lr": [], "tfidf_lr": []}
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(x, ya):
        preds = {
            "knn": knn_predict(x[tr], list(ya[tr]), x[te]),
            "embed_lr": LogisticRegression(max_iter=2000, C=10).fit(x[tr], ya[tr]).predict(x[te]),
            "tfidf_lr": make_pipeline(
                TfidfVectorizer(
                    sublinear_tf=True, min_df=2, ngram_range=(1, 2), max_features=60000
                ),
                LogisticRegression(max_iter=2000, C=10),
            )
            .fit([texts[i] for i in tr], ya[tr])
            .predict([texts[i] for i in te]),
        }
        for name, p in preds.items():
            scores[name].append((accuracy_score(ya[te], p), f1_score(ya[te], p, average="macro")))
    return {
        name: {
            "accuracy": round(float(np.mean([a for a, _ in s])), 4),
            "macro_f1": round(float(np.mean([f for _, f in s])), 4),
            "accuracy_std": round(float(np.std([a for a, _ in s])), 4),
        }
        for name, s in scores.items()
    }


# What popular notebooks on this dataset report. Accuracy on their own held-out split, title kept.
COMPETITORS = [
    {
        "who": "Kaggle · RF on word counts (110 votes)",
        "accuracy": 0.53,
        "url": "https://www.kaggle.com/code/sanchukanirupama/rf-based-multiclass-resume-classifier",
    },
    {
        "who": "Kaggle · TF-IDF(800) + k-NN",
        "accuracy": 0.558,
        "url": "https://www.kaggle.com/code/raselmeya/resume-categorization-with-ml-and-dl",
    },
    {
        "who": "Kaggle · TF-IDF(800) + logistic regression",
        "accuracy": 0.634,
        "url": "https://www.kaggle.com/code/raselmeya/resume-categorization-with-ml-and-dl",
    },
    {
        "who": "Kaggle · TF-IDF(800) + random forest",
        "accuracy": 0.681,
        "url": "https://www.kaggle.com/code/raselmeya/resume-categorization-with-ml-and-dl",
    },
    {
        "who": "Blog · TF-IDF + linear SVM",
        "accuracy": 0.8715,
        "url": "https://marcocamilo.com/portfolio/resume-classifier.html",
    },
    {
        "who": "Blog · fine-tuned BERT",
        "accuracy": 0.9167,
        "url": "https://marcocamilo.com/portfolio/resume-classifier.html",
    },
]
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _rrf(*rankings: np.ndarray, k: int = 60) -> np.ndarray:
    """Reciprocal rank fusion of several score matrices (queries x documents)."""
    fused = np.zeros_like(rankings[0], dtype=np.float64)
    for scores in rankings:
        ranks = np.argsort(np.argsort(-scores, axis=1), axis=1)
        fused += 1.0 / (k + ranks + 1)
    return fused


def hypothesis(x: np.ndarray, y: list[str], texts: list[str], embedder, seed: int = 7) -> dict:
    """Meaning plus words. The product already pairs semantic ranking with exact fields; here the
    lexical side is TF-IDF. Classification: average the class probabilities of the two models.
    Role query: add the prefix bge was trained with, then fuse the two rankings."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import StratifiedKFold

    ya = np.array(y)
    accs, f1s = [], []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(x, ya):
        emb = LogisticRegression(max_iter=2000, C=10).fit(x[tr], ya[tr])
        vec = TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2), max_features=60000)
        lex = LogisticRegression(max_iter=2000, C=10).fit(
            vec.fit_transform([texts[i] for i in tr]), ya[tr]
        )
        proba = (
            emb.predict_proba(x[te]) + lex.predict_proba(vec.transform([texts[i] for i in te]))
        ) / 2
        pred = emb.classes_[proba.argmax(axis=1)]
        accs.append(accuracy_score(ya[te], pred))
        f1s.append(f1_score(ya[te], pred, average="macro"))

    cats = sorted(set(y))
    plain = [f"{c.replace('-', ' ').lower()} professional résumé" for c in cats]
    labels = np.array(y)

    def p_at_10(scores: np.ndarray) -> float:
        top = np.argsort(-scores, axis=1)[:, :10]
        return float(np.mean([(labels[top[i]] == c).mean() for i, c in enumerate(cats)]))

    def q(queries: list[str]) -> np.ndarray:
        v = np.array(embedder.embed(queries), dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    vec = TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2), max_features=60000)
    doc_tfidf = vec.fit_transform(texts)
    # The lexical side gets the role words only. Generic words ("professional", "résumé") match
    # everything and drown the ranking; with role words absent from the texts its score is zero
    # and the hybrid falls back to the semantic ranking on its own.
    lexical = (vec.transform([c.replace("-", " ").lower() for c in cats]) @ doc_tfidf.T).toarray()
    semantic_prefixed = q([BGE_QUERY_PREFIX + t for t in plain]) @ x.T
    return {
        "fusion_accuracy": round(float(np.mean(accs)), 4),
        "fusion_macro_f1": round(float(np.mean(f1s)), 4),
        "query_p10_prefix": round(p_at_10(semantic_prefixed), 4),
        "query_p10_lexical_only": round(p_at_10(lexical), 4),
        "query_p10_rank_fusion": round(p_at_10(_rrf(semantic_prefixed, lexical)), 4),
        # Adding the scores instead of the ranks: with no lexical signal the sum is the semantic
        # score, so the hybrid can never fall below the better of its parts by much.
        "query_p10_score_fusion": round(p_at_10(semantic_prefixed + lexical), 4),
    }


def run(force: bool = False) -> dict:
    if RESULTS.exists() and not force:
        return json.loads(RESULTS.read_text())
    from cv_screener.index.embeddings import FastEmbedder

    embedder = FastEmbedder()
    raw, y = load()
    variants = {
        "full text": [normalise(t) for t in raw],
        "title removed, category words masked": [
            strip_leak(t, c) for t, c in zip(raw, y, strict=True)
        ],
    }
    out: dict = {
        "source": URL,
        "n": len(y),
        "categories": len(set(y)),
        "majority_baseline": round(max(y.count(c) for c in set(y)) / len(y), 4),
        "category_sizes": {c: y.count(c) for c in sorted(set(y))},
        "competitors": COMPETITORS,
        "variants": {},
    }
    for i, (name, texts) in enumerate(variants.items()):
        x = embed(texts, DATA / f"emb_{i}.npy", embedder)
        per_cat = query_precision(x, y, embedder)
        out["variants"][name] = {
            "classification": evaluate(x, y, texts),
            "same_category_at_5": round(neighbour_precision(x, y), 4),
            "query_precision_at_10": round(float(np.mean(list(per_cat.values()))), 4),
            "query_precision_by_category": {k: round(v, 2) for k, v in per_cat.items()},
            "hypothesis": hypothesis(x, y, texts, embedder),
        }
    RESULTS.write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    print(json.dumps(run(force="--force" in sys.argv), indent=2))
