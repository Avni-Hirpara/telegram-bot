from pathlib import Path

from health_rag.loaders.base import DocumentLoader
from health_rag.loaders.markdown import MarkdownLoader

LOADER_REGISTRY: dict[str, type[DocumentLoader]] = {
    ".md": MarkdownLoader,
    ".markdown": MarkdownLoader,
}


def get_loader(path: Path) -> DocumentLoader:
    ext = path.suffix.lower()
    if ext not in LOADER_REGISTRY:
        msg = f"No loader for extension {ext!r}: {path}"
        raise ValueError(msg)
    return LOADER_REGISTRY[ext]()
