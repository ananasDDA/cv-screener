"""Local text embeddings. No API key, no network after the first model download."""

from __future__ import annotations

import hashlib
import math
from typing import Protocol


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class FastEmbedder:
    """bge-small via fastembed (ONNX). ~130 MB download on first use, cached afterwards."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding  # heavy import, keep it lazy

        self._model = TextEmbedding(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [vec.tolist() for vec in self._model.embed(texts)]


class HashEmbedder:
    """Deterministic bag-of-words embedding for tests: shared tokens → closer vectors.
    It has no notion of meaning, only overlap, so tests using it assert on filters and plumbing."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for tok in text.lower().split():
                vec[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out
