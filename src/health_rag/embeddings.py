from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np


@lru_cache(maxsize=4)
def _get_model(model_name: str) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class Embedder:
    """Sentence-transformers wrapper with L2-normalized outputs for inner-product / cosine."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = _get_model(model_name)
        self.dim: int = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 32,
        )
        return np.asarray(vecs, dtype=np.float32)
