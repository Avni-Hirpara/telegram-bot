"""Domain models for documents, chunks, queries, and retrieval results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    file_path: str
    section: str
    doc_type: str


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    source: str
    section: str
    doc_type: str
