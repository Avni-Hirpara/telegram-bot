"""Lightweight health-topic gate: query embedding vs fixed anchor phrases (local, no LLM)."""

from __future__ import annotations

import logging

import numpy as np

from health_rag.embeddings import Embedder

logger = logging.getLogger(__name__)

# Short anchor lines covering the corpus / user intent; tune threshold via settings if needed.
HEALTH_ANCHOR_LINES: tuple[str, ...] = (
    "healthy eating nutrition diet protein vitamins meals snacks hydration",
    "physical activity exercise fitness walking running strength training yoga",
    "sleep stress mental wellness relaxation anxiety recovery burnout",
    "blood pressure diabetes heart disease chronic illness prevention screening",
    "weight management obesity lifestyle habits wellness self care symptoms",
    "doctor visit medication lifestyle medicine public health India context",
)


class HealthDomainGate:
    def __init__(self, embedder: Embedder, min_similarity: float) -> None:
        self._embedder = embedder
        self._min_similarity = min_similarity
        self._anchor_matrix = embedder.encode(list(HEALTH_ANCHOR_LINES))

    def health_similarity(self, query: str) -> float:
        q = query.strip()
        if not q:
            return 0.0
        qv = self._embedder.encode([q])
        # L2-normalized vectors → cosine similarity = inner product
        sims = (self._anchor_matrix @ qv.T).flatten()
        return float(np.max(sims))

    def is_health_related(self, query: str) -> bool:
        sim = self.health_similarity(query)
        ok = sim >= self._min_similarity
        logger.info(
            "domain_gate: query=%r max_anchor_sim=%.4f threshold=%.4f allowed=%s",
            query.strip()[:500],
            sim,
            self._min_similarity,
            ok,
        )
        return ok
