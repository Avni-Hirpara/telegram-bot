from __future__ import annotations

from pathlib import Path

from health_rag.domain.models import Document
from health_rag.loaders.base import DocumentLoader


class MarkdownLoader(DocumentLoader):
    """Split on ## headings. A lone # title line is ignored (not stored as metadata or body)."""

    def load(self, path: Path) -> list[Document]:
        lines = path.read_text(encoding="utf-8").splitlines()
        basename = path.name
        resolved = str(path.resolve())
        sections: list[tuple[str, list[str]]] = []
        current_heading = "Preamble"
        current_lines: list[str] = []

        for line in lines:
            if line.startswith("## ") and not line.startswith("###"):
                if current_lines:
                    sections.append((current_heading, current_lines))
                current_heading = line[3:].strip() or "Section"
                current_lines = []
            elif line.startswith("# ") and not line.startswith("##"):
                continue
            else:
                current_lines.append(line)

        sections.append((current_heading, current_lines))

        out: list[Document] = []
        for heading, body_lines in sections:
            body = "\n".join(body_lines).strip()
            if not body:
                continue
            out.append(
                Document(
                    text=body,
                    metadata={
                        "source": basename,
                        "path": resolved,
                        "section": heading,
                        "doc_type": "markdown",
                    },
                )
            )
        if not out:
            raw = path.read_text(encoding="utf-8").strip()
            if raw:
                out.append(
                    Document(
                        text=raw,
                        metadata={
                            "source": basename,
                            "path": resolved,
                            "section": "Body",
                            "doc_type": "markdown",
                        },
                    )
                )
        return out
