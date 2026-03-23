FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

ENV PYTHONUNBUFFERED=1

# Ollama is expected on the host; set OLLAMA_HOST (e.g. http://host.docker.internal:11434 on Docker Desktop).
CMD ["python", "-m", "health_rag.telegram_app.bot"]
