from __future__ import annotations

from rank_bm25 import BM25Okapi

from health_rag.config import Settings
from health_rag.domain.models import RetrievedChunk
from health_rag.embeddings import Embedder
from health_rag.index.faiss_index import load_index
from health_rag.retrieval.base import QueryEmbedCache
from health_rag.storage.store import ChunkStore


class HybridRetriever:
    """Dense (FAISS inner product on L2-normalized vectors) + BM25 fused with RRF."""

    def __init__(self, settings: Settings, store: ChunkStore, embedder: Embedder) -> None:
        self.settings = settings
        self.store = store
        self.embedder = embedder
        self._rows = store.iter_chunks_ordered_by_id()
        self._ids_list = [r.id for r in self._rows]
        self._texts = [r.text for r in self._rows]
        tokenized = [t.lower().split() for t in self._texts]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None
        self._index = load_index(settings.faiss_path)
        self._query_cache = QueryEmbedCache(embedder, settings.query_embedding_cache_size)

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        q = query.strip()
        if not q or not self._ids_list:
            return []

        qv = self._query_cache.get(q)
        k_dense = min(self.settings.retrieval_top_k_dense, len(self._ids_list))
        scores, indices = self._index.search(qv, k_dense)

        dense_ids: list[str] = []
        for idx in indices[0]:
            if int(idx) < 0:
                continue
            cid = self._ids_list[int(idx)]
            if cid not in dense_ids:
                dense_ids.append(cid)

        bm25_ids: list[str] = []
        if self._bm25 is not None:
            q_tokens = q.lower().split()
            if q_tokens:
                bm25_scores = self._bm25.get_scores(q_tokens)
                order = sorted(
                    range(len(bm25_scores)),
                    key=lambda i: bm25_scores[i],
                    reverse=True,
                )
                k_bm = min(self.settings.retrieval_top_k_bm25, len(order))
                for i in order[:k_bm]:
                    bm25_ids.append(self._ids_list[i])

        rrf_k = 60
        fused: dict[str, float] = {}
        for rank, cid in enumerate(dense_ids):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, cid in enumerate(bm25_ids):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)

        sorted_ids = sorted(fused.keys(), key=lambda cid: fused[cid], reverse=True)
        sorted_ids = sorted_ids[: self.settings.retrieval_final_k]

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
                    score=float(fused[cid]),
                    source=row.source,
                    section=row.section,
                    doc_type=row.doc_type,
                )
            )
        return out
