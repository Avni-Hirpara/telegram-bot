from __future__ import annotations

import asyncio
import logging
import time

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from health_rag.config import Settings
from health_rag.rag.prompt import OFF_DOMAIN_REPLY
from health_rag.services.rag_service import RAGService

logger = logging.getLogger(__name__)

HELP_TEXT = (
    "I'm a health coach bot. I answer questions from indexed health documents "
    "using a local LLM.\n\n"
    "Commands:\n"
    "/start — Short intro\n"
    "/help — This message\n"
    "/ask <your question> — Ask a health question\n\n"
    "Topics I cover: nutrition, eating habits, exercise, sleep, stress, "
    "blood pressure, diabetes prevention, and similar lifestyle topics.\n\n"
    "Off-topic questions get a short reply without document search."
)


def _truncate_telegram(text: str, max_len: int = 4090) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


async def _typing_while_streaming(
    bot,
    chat_id: int,
    interval_sec: float,
    stop: asyncio.Event,
) -> None:
    """Keep Telegram showing 'typing' while the LLM streams (bounded by interval)."""
    while not stop.is_set():
        try:
            await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        except Exception:
            logger.debug("send_chat_action failed", exc_info=True)
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_sec)
            return
        except asyncio.TimeoutError:
            continue


async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    rag: RAGService = context.bot_data["rag"]
    query = " ".join(context.args or []).strip()
    user = update.effective_user
    chat = update.effective_chat
    user_label = user.id if user else None
    chat_label = chat.id if chat else None
    username = user.username if user else None

    if not query:
        if update.effective_message:
            await update.effective_message.reply_text("Usage: /ask <your question>")
        return

    logger.info(
        "telegram /ask user_id=%s chat_id=%s username=%r query=%r",
        user_label,
        chat_label,
        username,
        query,
    )

    tag, messages, chunks = rag.prepare_ask(query)
    if tag == "off_domain":
        logger.info(
            "telegram /ask off_domain user_id=%s chat_id=%s query=%r",
            user_label,
            chat_label,
            query,
        )
        if update.effective_message:
            await update.effective_message.reply_text(OFF_DOMAIN_REPLY)
        return
    if tag == "no_hits":
        logger.info(
            "telegram /ask no_context user_id=%s chat_id=%s query=%r",
            user_label,
            chat_label,
            query,
        )
        if update.effective_message:
            await update.effective_message.reply_text(
                "I could not find relevant passages in the index. "
                "Try another question on health and wellness topics."
            )
        return

    if not messages or not chunks:
        logger.error("telegram /ask inconsistent rag payload tag=%r", tag)
        if update.effective_message:
            await update.effective_message.reply_text("Something went wrong. Please try again.")
        return

    if not update.effective_message:
        return
    placeholder = await update.effective_message.reply_text("Thinking…")
    full: list[str] = []
    last_flush = time.monotonic()
    pending_chars = 0

    stop_typing = asyncio.Event()
    typing_task = asyncio.create_task(
        _typing_while_streaming(
            context.bot,
            update.effective_chat.id,
            settings.stream_typing_interval_sec,
            stop_typing,
        )
    )
    try:
        async for delta in rag.stream_llm(messages):
            full.append(delta)
            pending_chars += len(delta)
            now = time.monotonic()
            if (
                now - last_flush >= settings.stream_edit_min_interval_sec
                or pending_chars >= settings.stream_edit_min_chars
            ):
                body = _truncate_telegram("".join(full), 3900)
                try:
                    await placeholder.edit_text(body or "…")
                except Exception:
                    logger.debug("edit_text during stream failed", exc_info=True)
                last_flush = now
                pending_chars = 0

        answer = "".join(full).strip() or "I don't know."
        sources = RAGService.format_sources(answer, chunks)
        final = answer
        if sources:
            final = f"{answer}\n\n{sources}"
        await placeholder.edit_text(_truncate_telegram(final))

        preview_len = 1200
        preview = answer if len(answer) <= preview_len else answer[: preview_len - 1] + "…"
        logger.info(
            "telegram /ask llm_done user_id=%s chat_id=%s answer_chars=%d answer_preview=%r",
            user_label,
            chat_label,
            len(answer),
            preview,
        )
        logger.debug("telegram /ask llm_full_answer: %s", answer)
        if sources:
            logger.debug("telegram /ask sources_block: %s", sources)
    except Exception as exc:
        logger.exception(
            "telegram /ask error user_id=%s chat_id=%s query=%r",
            user_label,
            chat_label,
            query,
        )
        await placeholder.edit_text(f"Sorry, something went wrong: {exc}")
    finally:
        stop_typing.set()
        try:
            await typing_task
        except asyncio.CancelledError:
            pass


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_message:
        await update.effective_message.reply_text(
            "Health coach bot. Ask a health question with /ask <your question>.\n"
            "Send /help for details."
        )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_message:
        await update.effective_message.reply_text(HELP_TEXT)
