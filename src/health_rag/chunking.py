from __future__ import annotations

import hashlib
import re
from pathlib import Path

from health_rag.config import Settings
from health_rag.domain.models import Chunk, Document


def _slug(s: str, max_len: int = 48) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()
    return (s[:max_len] or "sec")


def _char_window_chunks(text: str, chunk_size_tokens: int, overlap_tokens: int) -> list[str]:
    """Split using a char budget derived from pseudo-token targets (~4 chars / token)."""
    chars_per_token = 4
    chunk_chars = max(200, chunk_size_tokens * chars_per_token)
    overlap_chars = min(chunk_chars - 1, max(0, overlap_tokens * chars_per_token))
    if not text.strip():
        return []
    chunks_out: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_chars, n)
        chunks_out.append(text[start:end])
        if end >= n:
            break
        start = max(0, end - overlap_chars)
    return chunks_out


def documents_to_chunks(documents: list[Document], settings: Settings) -> list[Chunk]:
    """Turn loaded documents into sized chunks with stable ids."""
    size = settings.chunk_size_tokens
    overlap = settings.chunk_overlap_tokens
    chunks: list[Chunk] = []
    for doc in documents:
        source = str(doc.metadata.get("source", "unknown"))
        section = str(doc.metadata.get("section", ""))
        doc_type = str(doc.metadata.get("doc_type", "markdown"))
        path_key = str(doc.metadata.get("path", source))
        path_hash = hashlib.sha256(path_key.encode("utf-8")).hexdigest()[:8]
        stem = Path(source).stem
        sec_slug = _slug(section)
        parts = _char_window_chunks(doc.text, size, overlap)
        if not parts and doc.text.strip():
            parts = [doc.text.strip()]
        for i, part in enumerate(parts):
            if not part.strip():
                continue
            cid = f"{stem}_{path_hash}_{sec_slug}_{i}"
            chunks.append(
                Chunk(
                    id=cid,
                    text=part.strip(),
                    source=source,
                    file_path=path_key,
                    section=section,
                    doc_type=doc_type,
                )
            )
    return chunks
