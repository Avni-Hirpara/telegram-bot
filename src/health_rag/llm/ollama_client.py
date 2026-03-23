from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

import httpx

logger = logging.getLogger(__name__)


async def stream_chat(
    *,
    host: str,
    model: str,
    messages: list[dict[str, str]],
    timeout_sec: float = 120.0,
) -> AsyncIterator[str]:
    """Stream assistant tokens from Ollama's /api/chat endpoint."""
    base = host.rstrip("/")
    url = f"{base}/api/chat"
    payload = {"model": model, "messages": messages, "stream": True}
    logger.info(
        "ollama stream start host=%s model=%s message_roles=%s",
        base,
        model,
        [m.get("role") for m in messages],
    )
    line_events = 0
    chars_out = 0
    async with httpx.AsyncClient(timeout=timeout_sec) as client:
        async with client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug("ollama skip non-json line: %r", line[:200])
                    continue
                line_events += 1
                if data.get("done"):
                    break
                msg = data.get("message") or {}
                piece = msg.get("content") or ""
                if piece:
                    s = str(piece)
                    chars_out += len(s)
                    yield s
    logger.info(
        "ollama stream end model=%s json_lines=%d assistant_chars=%d",
        model,
        line_events,
        chars_out,
    )
