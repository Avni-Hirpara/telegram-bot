from __future__ import annotations

import fnmatch
import hashlib
import json
from enum import Enum
from pathlib import Path

from health_rag.chunking import documents_to_chunks
from health_rag.config import SCHEMA_VERSION, PersistedStoreConfig, Settings
from health_rag.embeddings import Embedder
from health_rag.index.faiss_index import build_and_save_index
from health_rag.loaders import get_loader
from health_rag.storage.store import ChunkStore


class IngestMode(str, Enum):
    REBUILD = "rebuild"
    ADD_ONLY = "add_only"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as bf:
        for block in iter(lambda: bf.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _collect_files(
    docs_dir: Path,
    source_filter: str | None,
) -> list[Path]:
    paths: list[Path] = []
    if docs_dir.is_dir():
        paths.extend(docs_dir.rglob("*.md"))
        paths.extend(docs_dir.rglob("*.markdown"))
    uniq = sorted({p.resolve(): p for p in paths}.values(), key=lambda p: str(p))
    if source_filter:
        filt: list[Path] = []
        for p in uniq:
            if fnmatch.fnmatch(p.name, source_filter) or fnmatch.fnmatch(str(p), source_filter):
                filt.append(p)
        uniq = filt
    return uniq


def _write_store_config(settings: Settings) -> None:
    cfg = PersistedStoreConfig(
        schema_version=SCHEMA_VERSION,
        embedding_model=settings.embedding_model,
        chunk_size_tokens=settings.chunk_size_tokens,
        chunk_overlap_tokens=settings.chunk_overlap_tokens,
        normalization="l2",
    )
    settings.store_config_path.parent.mkdir(parents=True, exist_ok=True)
    settings.store_config_path.write_text(
        json.dumps(cfg.model_dump(), indent=2),
        encoding="utf-8",
    )


def run_ingest(
    settings: Settings,
    *,
    mode: IngestMode,
    docs_dir: Path | None = None,
    source_filter: str | None = None,
) -> int:
    """Run ingestion. Returns number of source files processed."""
    docs_dir = docs_dir or settings.docs_dir
    settings.rag_store_path.mkdir(parents=True, exist_ok=True)

    store = ChunkStore(settings.sqlite_path)
    store.init_db()

    all_files = _collect_files(docs_dir, source_filter)
    if mode == IngestMode.REBUILD:
        store.clear_all()
        files_to_process = list(all_files)
    else:
        existing = store.get_source_hashes()
        files_to_process = []
        for fp in all_files:
            key = str(fp.resolve())
            digest = _file_sha256(fp)
            if existing.get(key) != digest:
                files_to_process.append(fp)

    if mode == IngestMode.ADD_ONLY and files_to_process:
        store.delete_chunks_for_paths({str(p.resolve()) for p in files_to_process})

    processed = 0
    embedder: Embedder | None = None
    if files_to_process:
        embedder = Embedder(settings.embedding_model)
        for fp in files_to_process:
            loader = get_loader(fp)
            docs = loader.load(fp)
            chunks = documents_to_chunks(docs, settings)
            store.insert_chunks(chunks)
            store.upsert_source(str(fp.resolve()), fp.name, _file_sha256(fp))
            processed += 1

        build_and_save_index(store, embedder, settings.faiss_path)
    elif mode == IngestMode.REBUILD and not files_to_process:
        if settings.faiss_path.exists():
            settings.faiss_path.unlink()
    elif mode == IngestMode.ADD_ONLY and not files_to_process:
        if store.chunk_count() > 0 and not settings.faiss_path.exists():
            embedder = embedder or Embedder(settings.embedding_model)
            build_and_save_index(store, embedder, settings.faiss_path)

    if store.chunk_count() > 0:
        _write_store_config(settings)
    else:
        if settings.faiss_path.exists():
            settings.faiss_path.unlink()
        if settings.store_config_path.exists():
            settings.store_config_path.unlink()

    return processed
