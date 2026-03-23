CREATE TABLE IF NOT EXISTS sources (
    path TEXT PRIMARY KEY,
    source_name TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    source TEXT NOT NULL,
    file_path TEXT NOT NULL,
    section TEXT,
    doc_type TEXT NOT NULL,
    faiss_id INTEGER
);

CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);
CREATE INDEX IF NOT EXISTS idx_chunks_file_path ON chunks(file_path);
CREATE INDEX IF NOT EXISTS idx_chunks_faiss ON chunks(faiss_id);
