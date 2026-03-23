from __future__ import annotations

import argparse
from pathlib import Path

from health_rag.config import Settings
from health_rag.pipeline.ingest_pipeline import IngestMode, run_ingest
from health_rag.storage.store import ChunkStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Markdown docs into the local RAG store.")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--rebuild",
        action="store_true",
        help="Clear chunks/sources and re-ingest all files.",
    )
    mode_group.add_argument(
        "--add-only",
        action="store_true",
        help="Ingest only new or changed files (default if neither flag is set).",
    )
    parser.add_argument(
        "--source",
        dest="source_filter",
        default=None,
        metavar="GLOB",
        help="Process only files whose basename matches this glob (e.g. '*diabetes*.md').",
    )
    parser.add_argument(
        "--docs",
        type=Path,
        default=None,
        help="Override Markdown root directory (default: DOCS_DIR / settings.docs_dir).",
    )
    args = parser.parse_args()
    settings = Settings()
    mode = IngestMode.REBUILD if args.rebuild else IngestMode.ADD_ONLY
    processed = run_ingest(
        settings,
        mode=mode,
        docs_dir=args.docs,
        source_filter=args.source_filter,
    )
    count = ChunkStore(settings.sqlite_path).chunk_count()
    print(f"Ingest complete. Files processed this run: {processed}. Total chunks: {count}.")


if __name__ == "__main__":
    main()
