from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Literal

from health_rag.config import Settings, validate_persisted_store
from health_rag.domain.models import RetrievedChunk
from health_rag.domain_gate import HealthDomainGate
from health_rag.embeddings import Embedder
from health_rag.llm.ollama_client import stream_chat
from health_rag.query_normalize import SimilarityQueryNormalizer
from health_rag.rag.context_builder import build_context
from health_rag.rag.prompt import build_messages
from health_rag.retrieval.factory import build_chunk_retriever
from health_rag.storage.store import ChunkStore

logger = logging.getLogger(__name__)

PrepareTag = Literal["empty", "off_domain", "no_hits", "rag"]


class RAGService:
    """Query-time orchestration: retrieve → context → prompt → LLM stream."""

    def __init__(self, settings: Settings) -> None:
        validate_persisted_store(
            settings.store_config_path,
            settings.sqlite_path,
            settings.faiss_path,
            settings,
        )
        self.settings = settings
        self.store = ChunkStore(settings.sqlite_path)
        self.embedder = Embedder(settings.embedding_model)
        self.retriever = build_chunk_retriever(settings, self.store, self.embedder)
        self._domain_gate = HealthDomainGate(
            self.embedder,
            settings.domain_gate_min_similarity,
        )
        self._query_normalizer = SimilarityQueryNormalizer(
            self.embedder,
            min_replace_similarity=settings.query_typo_min_similarity,
        )

    def prepare_ask(
        self,
        query: str,
        *,
        skip_domain_gate: bool = False,
    ) -> tuple[PrepareTag, list[dict[str, str]], list[RetrievedChunk]]:
        """Classify query, then retrieve + build prompt only for on-domain RAG path."""
        q_raw = query.strip()
        if not q_raw:
            return "empty", [], []
        q = self._query_normalizer.normalize(q_raw)
        if q != q_raw:
            logger.debug("prepare_ask: normalized query %r -> %r", q_raw, q)

        if self.settings.domain_gate_enabled and not skip_domain_gate:
            if not self._domain_gate.is_health_related(q):
                logger.info(
                    "prepare_ask: off_domain, skipping retrieval raw=%r normalized=%r",
                    q_raw,
                    q,
                )
                return "off_domain", [], []

        chunks = self.retriever.retrieve(q)
        if not chunks:
            logger.info("retrieval: no hits query=%r retriever=%s", q, type(self.retriever).__name__)
            return "no_hits", [], []

        ctx = build_context(chunks, self.settings.context_max_tokens)
        messages = build_messages(ctx, q)
        logger.info(
            "retrieval: query=%r retriever=%s hits=%d context_chars=%d",
            q,
            type(self.retriever).__name__,
            len(chunks),
            len(ctx),
        )
        for i, c in enumerate(chunks, start=1):
            logger.info(
                "retrieval hit %d: id=%s score=%.6f source=%s section=%r doc_type=%s",
                i,
                c.chunk_id,
                c.score,
                c.source,
                c.section,
                c.doc_type,
            )
            logger.debug("retrieval hit %d text_preview: %s", i, _preview(c.text, 280))
        return "rag", messages, chunks

    async def stream_llm(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        async for piece in stream_chat(
            host=self.settings.ollama_host,
            model=self.settings.ollama_model,
            messages=messages,
        ):
            logger.debug("llm stream delta: %r", piece)
            yield piece

    @staticmethod
    def format_sources(answer: str, chunks: list[RetrievedChunk]) -> str:
        """One bullet per source file — but only when the LLM actually used the context.

        If the answer signals that the model could not answer from the provided
        context (e.g. "I don't know"), sources are suppressed.
        """
        if _answer_lacks_knowledge(answer):
            return ""
        seen: set[str] = set()
        lines: list[str] = []
        for c in chunks:
            src = (c.source or "").strip()
            if not src or src in seen:
                continue
            seen.add(src)
            lines.append(f"- {src}")
        if not lines:
            return ""
        return "Sources:\n" + "\n".join(lines)


_NO_ANSWER_SIGNALS = (
    "i don't know",
    "i do not know",
    "not enough information",
    "no information",
    "does not contain",
    "doesn't contain",
    "not mentioned",
    "cannot answer",
    "can't answer",
    "not covered",
    "outside the scope",
    "beyond the scope",
    "no relevant information",
)


def _answer_lacks_knowledge(answer: str) -> bool:
    """Heuristic: the LLM admitted it cannot answer from the provided context."""
    low = answer.strip().lower()
    return any(signal in low for signal in _NO_ANSWER_SIGNALS)


def _preview(text: str, max_len: int) -> str:
    t = " ".join(text.split())
    if len(t) <= max_len:
        return t
    return t[: max_len - 1] + "…"
