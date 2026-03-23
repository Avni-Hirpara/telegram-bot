from health_rag.domain.models import RetrievedChunk
from health_rag.rag.context_builder import build_context
from health_rag.services.rag_service import RAGService

_SAMPLE_CHUNKS = [
    RetrievedChunk(
        chunk_id="a", text="alpha", score=1.0,
        source="s.md", section="One", doc_type="markdown",
    ),
]


def test_build_context_dedupes_and_respects_budget() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id="a",
            text="short a",
            score=1.0,
            source="s.md",
            section="One",
            doc_type="markdown",
        ),
        RetrievedChunk(
            chunk_id="a",
            text="duplicate",
            score=0.5,
            source="s.md",
            section="One",
            doc_type="markdown",
        ),
        RetrievedChunk(
            chunk_id="b",
            text="short b",
            score=0.9,
            source="t.md",
            section="Two",
            doc_type="markdown",
        ),
    ]
    ctx = build_context(chunks, max_tokens=500)
    assert "short a" in ctx
    assert "duplicate" not in ctx
    assert "short b" in ctx


def test_format_sources_normal_answer() -> None:
    result = RAGService.format_sources("Eat more vegetables for better health.", _SAMPLE_CHUNKS)
    assert "Sources:" in result
    assert "s.md" in result


def test_format_sources_suppressed_when_llm_cannot_answer() -> None:
    for answer in [
        "I don't know.",
        "The context does not contain information about rockets.",
        "I cannot answer this based on the provided context.",
        "There is no relevant information in the context provided.",
    ]:
        assert RAGService.format_sources(answer, _SAMPLE_CHUNKS) == "", f"Expected no sources for: {answer!r}"
