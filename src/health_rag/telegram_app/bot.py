from __future__ import annotations

from telegram import Update
from telegram.ext import Application, CommandHandler

from health_rag.config import Settings
from health_rag.logging_config import configure_logging
from health_rag.services.rag_service import RAGService
from health_rag.telegram_app.handlers import ask_command, help_command, start_command


def main() -> None:
    settings = Settings()
    configure_logging(settings.log_level)
    if not settings.telegram_bot_token.strip():
        raise SystemExit("Set TELEGRAM_BOT_TOKEN in your environment or .env file.")
    rag = RAGService(settings)
    app = (
        Application.builder()
        .token(settings.telegram_bot_token.strip())
        .build()
    )
    app.bot_data["settings"] = settings
    app.bot_data["rag"] = rag
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("ask", ask_command))
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
