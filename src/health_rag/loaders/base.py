from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from health_rag.domain.models import Document


class DocumentLoader(ABC):
    @abstractmethod
    def load(self, path: Path) -> list[Document]:
        """Load a file into one or more logical documents (sections)."""
