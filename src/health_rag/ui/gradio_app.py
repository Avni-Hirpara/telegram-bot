"""
Local Gradio UI for debugging RAG: retrieval trace + model answer (assignment / demo capture).

Run (after ingest + Ollama):
  pip install -e ".[ui]"
  health-rag-ui
"""

from __future__ import annotations

import asyncio
import os

from health_rag.config import Settings
from health_rag.domain.models import RetrievedChunk
from health_rag.logging_config import configure_logging
from health_rag.rag.prompt import OFF_DOMAIN_REPLY
from health_rag.services.rag_service import RAGService


def _format_retrieval_debug(
    tag: str,
    chunks: list[RetrievedChunk],
    retriever_name: str,
) -> str:
    lines = [
        f"**Path:** `{tag}`",
        f"**Retriever:** `{retriever_name}`",
        "",
    ]
    if not chunks:
        lines.append("_No chunks retrieved._")
        return "\n".join(lines)
    lines.append("**Top chunks:**")
    for i, c in enumerate(chunks, start=1):
        preview = " ".join((c.text or "").split())[:320]
        if len((c.text or "")) > 320:
            preview += "…"
        lines.append(
            f"{i}. `score={c.score:.4f}` **{c.source}** — _{c.section}_\n"
            f"   `id=` `{c.chunk_id}`\n"
            f"   {preview}"
        )
    return "\n".join(lines)


async def _run_ask(rag: RAGService, query: str, skip_domain_gate: bool) -> tuple[str, str]:
    q = (query or "").strip()
    if not q:
        return "_Enter a question._", ""

    tag, messages, chunks = rag.prepare_ask(q, skip_domain_gate=skip_domain_gate)
    retriever = type(rag.retriever).__name__
    debug = _format_retrieval_debug(tag, chunks, retriever)

    if tag == "empty":
        return debug, ""
    if tag == "off_domain":
        return debug, OFF_DOMAIN_REPLY
    if tag == "no_hits":
        return (
            debug,
            "No relevant passages in the index. Run `health-rag-ingest --rebuild` or try another question.",
        )

    parts: list[str] = []
    async for piece in rag.stream_llm(messages):
        parts.append(piece)
    answer = "".join(parts).strip() or "I don't know."
    sources = RAGService.format_sources(answer, chunks)
    full = f"{answer}\n\n{sources}" if sources else answer
    return debug, full


def main() -> None:
    try:
        import gradio as gr
    except ImportError as e:
        raise SystemExit(
            "Gradio is required. Install with: pip install -e \".[ui]\""
        ) from e

    configure_logging(os.environ.get("LOG_LEVEL", "INFO"))
    settings = Settings()
    rag = RAGService(settings)

    def sync_wrapper(query: str, skip_gate: bool) -> tuple[str, str]:
        return asyncio.run(_run_ask(rag, query, skip_gate))

    with gr.Blocks(title="Health RAG — debug UI") as demo:
        gr.Markdown(
            "## Health mini-RAG (local debug)\n"
            "Same pipeline as the Telegram bot: domain gate → retrieve top chunks → Ollama. "
            "Use **Skip domain gate** to force retrieval on any text (debug only)."
        )
        q = gr.Textbox(label="Question", lines=2, placeholder="e.g. How can I eat healthier in India?")
        skip = gr.Checkbox(label="Skip domain gate (debug)", value=False)
        go = gr.Button("Run RAG", variant="primary")
        gr.Markdown("### Retrieval trace")
        dbg = gr.Markdown()
        ans = gr.Textbox(label="Reply (answer + Sources)", lines=16, max_lines=40)
        go.click(sync_wrapper, inputs=[q, skip], outputs=[dbg, ans])

    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    host = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    demo.launch(server_name=host, server_port=port, show_error=True)


if __name__ == "__main__":
    main()
