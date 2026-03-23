from __future__ import annotations

from health_rag.domain.models import RetrievedChunk
from health_rag.token_estimate import pseudo_token_len


def build_context(chunks: list[RetrievedChunk], max_tokens: int) -> str:
    """Dedupe by chunk_id, preserve retrieval order, trim to a token budget."""
    seen: set[str] = set()
    parts: list[str] = []
    used = 0
    for c in chunks:
        if c.chunk_id in seen:
            continue
        seen.add(c.chunk_id)
        label = f"{c.source} — {c.section}" if c.section else c.source
        block = f"[{label}]\n{c.text}\n"
        n = pseudo_token_len(block)
        if used + n > max_tokens:
            break
        parts.append(block)
        used += n
    return "\n".join(parts).strip()
