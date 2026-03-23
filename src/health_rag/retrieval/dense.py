from __future__ import annotations

from health_rag.config import Settings
from health_rag.domain.models import RetrievedChunk
from health_rag.embeddings import Embedder
from health_rag.index.faiss_index import load_index
from health_rag.retrieval.base import QueryEmbedCache
from health_rag.storage.store import ChunkStore


class DenseRetriever:
    """FAISS inner product on L2-normalized vectors (cosine similarity)."""

    def __init__(self, settings: Settings, store: ChunkStore, embedder: Embedder) -> None:
        self.settings = settings
        self.store = store
        self.embedder = embedder
        self._rows = store.iter_chunks_ordered_by_id()
        self._ids_list = [r.id for r in self._rows]
        self._index = load_index(settings.faiss_path)
        self._query_cache = QueryEmbedCache(embedder, settings.query_embedding_cache_size)

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        q = query.strip()
        if not q or not self._ids_list:
            return []

        qv = self._query_cache.get(q)
        k = min(self.settings.retrieval_top_k_dense, len(self._ids_list))
        scores, indices = self._index.search(qv, k)

        ranked: list[tuple[str, float]] = []
        seen: set[str] = set()
        for rank in range(k):
            idx = int(indices[0][rank])
            if idx < 0:
                continue
            cid = self._ids_list[idx]
            if cid in seen:
                continue
            seen.add(cid)
            ranked.append((cid, float(scores[0][rank])))

        ranked.sort(key=lambda x: x[1], reverse=True)
        ranked = ranked[: self.settings.retrieval_final_k]
        sorted_ids = [cid for cid, _ in ranked]
        score_by_id = dict(ranked)

        by_id = self.store.get_chunks_by_ids(sorted_ids)
        out: list[RetrievedChunk] = []
        for cid in sorted_ids:
            row = by_id.get(cid)
            if row is None:
                continue
            out.append(
                RetrievedChunk(
                    chunk_id=row.id,
                    text=row.text,
                    score=score_by_id[cid],
                    source=row.source,
                    section=row.section,
                    doc_type=row.doc_type,
                )
            )
        return out
