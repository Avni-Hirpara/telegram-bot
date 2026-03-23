from __future__ import annotations

from collections import OrderedDict
from typing import Protocol

import numpy as np

from health_rag.domain.models import RetrievedChunk
from health_rag.embeddings import Embedder


class ChunkRetriever(Protocol):
    """Pluggable retrieval; swap implementations as corpus size and modalities grow."""

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        ...


class QueryEmbedCache:
    """LRU-ish query embedding cache shared by retriever implementations."""

    def __init__(self, embedder: Embedder, max_size: int) -> None:
        self._embedder = embedder
        self._max = max_size
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, query: str) -> np.ndarray:
        if query in self._cache:
            self._cache.move_to_end(query)
            return self._cache[query]
        vec = self._embedder.encode([query])
        self._cache[query] = vec
        self._cache.move_to_end(query)
        while len(self._cache) > self._max:
            self._cache.popitem(last=False)
        return vec
