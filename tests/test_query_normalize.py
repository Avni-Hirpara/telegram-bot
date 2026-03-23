"""Tests for SimilarityQueryNormalizer (mocked embedder; no model download)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from health_rag.query_normalize import SimilarityQueryNormalizer


def _l2(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1.0, norms)


def _make_normalizer(
    *,
    lexicon: list[str],
    lexicon_matrix: np.ndarray,
    token_vectors: dict[str, list[float]],
    min_sim: float = 0.85,
) -> SimilarityQueryNormalizer:
    embedder = MagicMock()
    n = SimilarityQueryNormalizer(embedder, min_replace_similarity=min_sim)
    n._lexicon = list(lexicon)
    n._lexicon_set = frozenset(n._lexicon)
    n._matrix = _l2(np.asarray(lexicon_matrix, dtype=np.float64))

    def encode(texts: list[str]) -> np.ndarray:
        return _l2(np.asarray([token_vectors[t] for t in texts], dtype=np.float64))

    embedder.encode.side_effect = encode
    return n


def test_digit_letter_typo_corrected() -> None:
    n = _make_normalizer(
        lexicon=["diabetes", "pressure"],
        lexicon_matrix=np.array([[1.0, 0.0], [0.0, 1.0]]),
        token_vectors={"2diabetes": [0.97, 0.1]},
        min_sim=0.85,
    )
    assert n.normalize("type 2diabetes") == "type diabetes"


def test_trailing_punctuation_preserved() -> None:
    n = _make_normalizer(
        lexicon=["diabetes"],
        lexicon_matrix=np.array([[1.0, 0.0]]),
        token_vectors={"2diabetes": [1.0, 0.0]},
        min_sim=0.5,
    )
    assert n.normalize("what about 2diabetes?") == "what about diabetes?"


def test_pure_words_not_touched() -> None:
    n = _make_normalizer(
        lexicon=["diabetes"],
        lexicon_matrix=np.array([[1.0, 0.0]]),
        token_vectors={},
        min_sim=0.5,
    )
    assert n.normalize("  hello   recover   world  ") == "hello recover world"


def test_pure_numbers_not_touched() -> None:
    n = _make_normalizer(
        lexicon=["diabetes"],
        lexicon_matrix=np.array([[1.0, 0.0]]),
        token_vectors={},
        min_sim=0.5,
    )
    assert n.normalize("year 2024 stats") == "year 2024 stats"


def test_below_threshold_keeps_original() -> None:
    n = _make_normalizer(
        lexicon=["diabetes"],
        lexicon_matrix=np.array([[1.0, 0.0]]),
        token_vectors={"2diabetes": [0.0, 1.0]},
        min_sim=0.85,
    )
    assert n.normalize("2diabetes") == "2diabetes"
