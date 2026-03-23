from __future__ import annotations

from health_rag.config import Settings
from health_rag.embeddings import Embedder
from health_rag.retrieval.base import ChunkRetriever
from health_rag.retrieval.dense import DenseRetriever
from health_rag.retrieval.hybrid import HybridRetriever
from health_rag.storage.store import ChunkStore


def build_chunk_retriever(settings: Settings, store: ChunkStore, embedder: Embedder) -> ChunkRetriever:
    """Construct the active retriever from settings (dense now; hybrid when corpus grows)."""
    if settings.retrieval_backend == "dense":
        return DenseRetriever(settings, store, embedder)
    if settings.retrieval_backend == "hybrid":
        return HybridRetriever(settings, store, embedder)
    raise ValueError(f"Unknown retrieval_backend: {settings.retrieval_backend!r}")
