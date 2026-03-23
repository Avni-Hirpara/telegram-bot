"""Offline-friendly ~token counts (no remote encoding downloads)."""


def pseudo_token_len(text: str) -> int:
    """Rough OpenAI-style token estimate: ~4 characters per token."""
    if not text:
        return 0
    return max(1, len(text) // 4)
