"""Application settings derived from environment (no hardcoded artifact paths)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Bump when embedding model, chunk params, or DB schema change (requires re-ingest).
SCHEMA_VERSION = "v3"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    rag_store_path: Path = Path("var")
    docs_dir: Path = Path("Data/docs")

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size_tokens: int = 350
    chunk_overlap_tokens: int = 20
    context_max_tokens: int = 2000

    # dense: FAISS-only (good for small / homogenous corpora e.g. Markdown). hybrid: + BM25 RRF for scale.
    retrieval_backend: Literal["dense", "hybrid"] = "dense"
    retrieval_top_k_dense: int = 8
    retrieval_top_k_bm25: int = 8
    # Chunks passed to context + LLM (after ranking / fusion).
    retrieval_final_k: int = 3

    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "mistral"

    telegram_bot_token: str = ""

    stream_edit_min_interval_sec: float = 0.85
    stream_edit_min_chars: int = 320

    query_embedding_cache_size: int = 256

    # Logging (stderr); use DEBUG for chunk text previews and stream deltas.
    log_level: str = "INFO"

    # While the model streams, keep Telegram "typing" alive (seconds between actions).
    stream_typing_interval_sec: float = 4.5

    # Skip RAG + LLM when the question does not look health-related (embedding vs anchors).
    domain_gate_enabled: bool = True
    # Max cosine similarity to any anchor must be >= this (normalized MiniLM embeddings).
    domain_gate_min_similarity: float = 0.32

    # Typo correction: digit+letter tokens mapped to nearest health term (same embedder).
    query_typo_min_similarity: float = 0.50

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sqlite_path(self) -> Path:
        return self.rag_store_path / "chunks.sqlite3"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def faiss_path(self) -> Path:
        return self.rag_store_path / "faiss.index"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def store_config_path(self) -> Path:
        return self.rag_store_path / "config.json"


class PersistedStoreConfig(BaseModel):
    """Written to store_config_path after ingest; validated before retrieve."""

    schema_version: str
    embedding_model: str
    chunk_size_tokens: int
    chunk_overlap_tokens: int
    normalization: str = "l2"


def validate_persisted_store(store_config_path: Path, sqlite_path: Path, faiss_path: Path, settings: Settings) -> None:
    """Fail fast if the index was built with different parameters or is missing."""
    if not store_config_path.is_file():
        raise FileNotFoundError(
            f"RAG store is not initialized: missing {store_config_path}. Run: health-rag-ingest --rebuild"
        )
    if not sqlite_path.is_file():
        raise FileNotFoundError(f"Missing SQLite store at {sqlite_path}. Run ingest.")
    if not faiss_path.is_file():
        raise FileNotFoundError(f"Missing FAISS index at {faiss_path}. Run ingest.")
    raw = store_config_path.read_text(encoding="utf-8")
    persisted = PersistedStoreConfig.model_validate_json(raw)
    if persisted.schema_version != SCHEMA_VERSION:
        raise RuntimeError(
            f"Store schema_version {persisted.schema_version!r} != runtime {SCHEMA_VERSION!r}. "
            "Re-run: health-rag-ingest --rebuild"
        )
    if persisted.embedding_model != settings.embedding_model:
        raise RuntimeError(
            f"Store embedding_model {persisted.embedding_model!r} != runtime {settings.embedding_model!r}. "
            "Re-run ingest or align EMBEDDING_MODEL in .env."
        )
    if persisted.chunk_size_tokens != settings.chunk_size_tokens:
        raise RuntimeError(
            "Store chunk_size_tokens differs from runtime settings. Re-run: health-rag-ingest --rebuild"
        )
    if persisted.chunk_overlap_tokens != settings.chunk_overlap_tokens:
        raise RuntimeError(
            "Store chunk_overlap_tokens differs from runtime settings. Re-run: health-rag-ingest --rebuild"
        )
