"""Query typo correction via embedding similarity to the health vocabulary."""

from __future__ import annotations

import logging
import string

import numpy as np

from health_rag.domain_gate import HEALTH_ANCHOR_LINES
from health_rag.embeddings import Embedder

logger = logging.getLogger(__name__)

_TRAILING_PUNCT = string.punctuation + string.whitespace


def _build_lexicon() -> list[str]:
    """Unique words (len > 2) from the shared health anchor vocabulary."""
    words: set[str] = set()
    for line in HEALTH_ANCHOR_LINES:
        for w in line.lower().split():
            if len(w) > 2:
                words.add(w)
    return sorted(words)


class SimilarityQueryNormalizer:
    """Correct typo tokens that mix letters and digits by finding the nearest
    health lexicon term via the same embedder used for retrieval.

    Only tokens containing both a letter and a digit are considered candidates
    (pure numbers like ``2024`` and regular words are left untouched).
    Trailing punctuation is stripped before embedding and reattached afterward.
    """

    def __init__(
        self,
        embedder: Embedder,
        *,
        min_replace_similarity: float = 0.50,
    ) -> None:
        self._embedder = embedder
        self._min_sim = min_replace_similarity
        self._lexicon = _build_lexicon()
        self._lexicon_set = frozenset(self._lexicon)
        self._matrix: np.ndarray | None = None

    def _ensure_matrix(self) -> None:
        if self._matrix is None:
            self._matrix = self._embedder.encode(self._lexicon)

    def normalize(self, text: str) -> str:
        q = " ".join(text.split())
        if not q:
            return q

        tokens = q.split()
        candidates: list[tuple[int, str, str]] = []
        for i, tok in enumerate(tokens):
            core = tok.rstrip(_TRAILING_PUNCT)
            if len(core) <= 2 or core.lower() in self._lexicon_set:
                continue
            has_digit = any(c.isdigit() for c in core)
            has_alpha = any(c.isalpha() for c in core)
            if has_digit and has_alpha:
                candidates.append((i, tok, core))

        if not candidates:
            return q

        self._ensure_matrix()
        assert self._matrix is not None
        vecs = self._embedder.encode([c for _, _, c in candidates])
        sims = self._matrix @ vecs.T

        out = list(tokens)
        for col, (idx, original, core) in enumerate(candidates):
            best = int(np.argmax(sims[:, col]))
            score = float(sims[best, col])
            if score >= self._min_sim:
                replacement = self._lexicon[best] + original[len(core):]
                if replacement.lower() != original.lower():
                    logger.debug(
                        "query_normalize: %r -> %r (sim=%.3f)",
                        original, replacement, score,
                    )
                out[idx] = replacement

        return " ".join(out)
