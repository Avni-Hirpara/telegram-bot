from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np

from health_rag.embeddings import Embedder
from health_rag.storage.store import ChunkStore


def build_and_save_index(store: ChunkStore, embedder: Embedder, faiss_path: Path) -> int:
    rows = store.iter_chunks_ordered_by_id()
    if not rows:
        if faiss_path.exists():
            faiss_path.unlink()
        return 0
    texts = [r.text for r in rows]
    ids = [r.id for r in rows]
    vectors = embedder.encode(texts)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(faiss_path))
    store.set_faiss_ids(ids)
    return len(ids)


def load_index(faiss_path: Path) -> faiss.Index:
    return faiss.read_index(str(faiss_path))
