#!/usr/bin/env python3
"""Print retrieval results for a fixed query set (no LLM required)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from health_rag.config import Settings, validate_persisted_store  # noqa: E402
from health_rag.embeddings import Embedder  # noqa: E402
from health_rag.retrieval.factory import build_chunk_retriever  # noqa: E402
from health_rag.storage.store import ChunkStore  # noqa: E402

QUERIES = [
    "how to eat healthy in India",
    "how stress affects diabetes",
    "what should kids eat for growth",
    "how to build a rocket",
]


def main() -> None:
    settings = Settings()
    validate_persisted_store(
        settings.store_config_path,
        settings.sqlite_path,
        settings.faiss_path,
        settings,
    )
    store = ChunkStore(settings.sqlite_path)
    embedder = Embedder(settings.embedding_model)
    retriever = build_chunk_retriever(settings, store, embedder)
    for q in QUERIES:
        print("=" * 60)
        print("Q:", q)
        hits = retriever.retrieve(q)
        for h in hits:
            print(f"  score={h.score:.4f} id={h.chunk_id} source={h.source} section={h.section!r}")


if __name__ == "__main__":
    main()
