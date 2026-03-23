from __future__ import annotations

from datetime import datetime, timezone
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from health_rag.domain.models import Chunk


def _schema_sql() -> str:
    return (Path(__file__).resolve().parent / "schema.sql").read_text(encoding="utf-8")


@dataclass
class ChunkRow:
    id: str
    text: str
    source: str
    file_path: str
    section: str
    doc_type: str
    faiss_id: int | None


class ChunkStore:
    def __init__(self, sqlite_path: Path) -> None:
        self.sqlite_path = sqlite_path

    def _connect(self) -> sqlite3.Connection:
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(str(self.sqlite_path))

    def init_db(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(_schema_sql())
            conn.commit()
        finally:
            conn.close()

    def clear_all(self) -> None:
        conn = self._connect()
        try:
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM sources")
            conn.commit()
        finally:
            conn.close()

    def get_source_hashes(self) -> dict[str, str]:
        conn = self._connect()
        try:
            cur = conn.execute("SELECT path, content_hash FROM sources")
            return {str(r[0]): str(r[1]) for r in cur.fetchall()}
        finally:
            conn.close()

    def delete_chunks_for_paths(self, resolved_paths: set[str]) -> None:
        if not resolved_paths:
            return
        conn = self._connect()
        try:
            qmarks = ",".join("?" * len(resolved_paths))
            conn.execute(f"DELETE FROM chunks WHERE file_path IN ({qmarks})", tuple(resolved_paths))
            conn.commit()
        finally:
            conn.close()

    def upsert_source(self, path: str, source_name: str, content_hash: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO sources (path, source_name, content_hash, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    source_name = excluded.source_name,
                    content_hash = excluded.content_hash,
                    updated_at = excluded.updated_at
                """,
                (path, source_name, content_hash, now),
            )
            conn.commit()
        finally:
            conn.close()

    def insert_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        conn = self._connect()
        try:
            conn.executemany(
                """
                INSERT INTO chunks (id, text, source, file_path, section, doc_type, faiss_id)
                VALUES (?, ?, ?, ?, ?, ?, NULL)
                """,
                [
                    (
                        c.id,
                        c.text,
                        c.source,
                        c.file_path,
                        c.section,
                        c.doc_type,
                    )
                    for c in chunks
                ],
            )
            conn.commit()
        finally:
            conn.close()

    def iter_chunks_ordered_by_id(self) -> list[ChunkRow]:
        conn = self._connect()
        try:
            cur = conn.execute(
                "SELECT id, text, source, file_path, section, doc_type, faiss_id FROM chunks ORDER BY id ASC"
            )
            rows = cur.fetchall()
            return [
                ChunkRow(
                    id=str(r[0]),
                    text=str(r[1]),
                    source=str(r[2]),
                    file_path=str(r[3]),
                    section=str(r[4] or ""),
                    doc_type=str(r[5]),
                    faiss_id=int(r[6]) if r[6] is not None else None,
                )
                for r in rows
            ]
        finally:
            conn.close()

    def get_chunks_by_ids(self, ids: list[str]) -> dict[str, ChunkRow]:
        if not ids:
            return {}
        conn = self._connect()
        try:
            qmarks = ",".join("?" * len(ids))
            cur = conn.execute(
                f"SELECT id, text, source, file_path, section, doc_type, faiss_id FROM chunks WHERE id IN ({qmarks})",
                tuple(ids),
            )
            out: dict[str, ChunkRow] = {}
            for r in cur.fetchall():
                out[str(r[0])] = ChunkRow(
                    id=str(r[0]),
                    text=str(r[1]),
                    source=str(r[2]),
                    file_path=str(r[3]),
                    section=str(r[4] or ""),
                    doc_type=str(r[5]),
                    faiss_id=int(r[6]) if r[6] is not None else None,
                )
            return out
        finally:
            conn.close()

    def set_faiss_ids(self, ordered_ids: list[str]) -> None:
        conn = self._connect()
        try:
            for i, cid in enumerate(ordered_ids):
                conn.execute("UPDATE chunks SET faiss_id = ? WHERE id = ?", (i, cid))
            conn.commit()
        finally:
            conn.close()

    def chunk_count(self) -> int:
        conn = self._connect()
        try:
            cur = conn.execute("SELECT COUNT(*) FROM chunks")
            return int(cur.fetchone()[0])
        finally:
            conn.close()
