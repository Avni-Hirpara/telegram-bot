# Health RAG Telegram Bot

A simple, local-first chatbot that answers health and wellness questions using your own documents.

It reads Markdown health guides, breaks them into chunks, finds the most relevant ones using embeddings + FAISS, and generates answers using a local LLM (via Ollama).

---

## What this project does

This project turns your health and wellness documents into a conversational assistant. You can interact with it through a Telegram bot, ask questions in natural language, and receive real-time responses grounded in your content. Since everything runs locally, your data stays private, and each answer includes source references so you can easily verify the information.

---

## ⚙️ Tech stack

- Embeddings: `all-MiniLM-L6-v2`
- Vector search: FAISS
- Storage: SQLite
- LLM: Ollama (model: mistral)
- Interface: Telegram bot

---

## 🚀 Quick start

```bash
git clone <repo-url>
cd telegram-bot

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -e ".[dev,ui]"

cp .env.example .env
```

---

### 1. Setup `.env`

Add your Telegram bot token:

```env
TELEGRAM_BOT_TOKEN=your_token_here
OLLAMA_MODEL=mistral
```

---

### 2. Start Ollama

```bash
ollama serve
ollama pull mistral
```

---

### 3. Add your documents

Put Markdown files inside:

```
Data/docs/
```

---

### 4. Index your docs

```bash
health-rag-ingest --rebuild
```

---

### 5. Run the bot

```bash
python app.py
```

---

### 6. Test it in Telegram

Open your bot and try:

```
/help
/ask how to reduce blood pressure
/ask what should kids eat daily
```

---

## 🧠 How it works

### Ingestion (offline)

```
Docs → chunk → embed → store (SQLite + FAISS)
```

### Query (online)

```
User → Telegram → retrieve chunks → build prompt → LLM → answer
```

---

## 📊 System design (high level)

```mermaid
flowchart LR

  subgraph Offline [Ingestion]
    A[Markdown Docs] --> B[Chunk + Embed]
    B --> C[FAISS Index]
    B --> D[SQLite Store]
  end

  subgraph Online [Query]
    U[User] --> T[Telegram Bot]
    T --> R[Retriever (FAISS)]
    R --> D
    R --> C
    R --> P[Prompt Builder]
    P --> L[Ollama LLM]
    L --> T
  end
```

---

## Debug UI (Gradio)

Run:

```bash
health-rag-ui
```

Then open:

```
http://127.0.0.1:7860
```

---

## 🧰 Useful commands

```bash
health-rag-ingest --rebuild                       # full re-index
health-rag-ingest --add-only                      # changed files only
pytest                                            # unit tests
python scripts/eval_queries.py                    # print retrieval hits (no LLM)
python scripts/eval_response.py                   # LLM-as-a-judge eval (needs Ollama)
python scripts/eval_response.py --out results.json
```

---

## 📋 Response evaluation

`scripts/eval_response.py` runs a set of queries through the full RAG pipeline, then uses the LLM itself as a judge to score each answer on:

- **Faithfulness** (1-5): Is the answer grounded in the retrieved context?
- **Relevance** (1-5): Does the answer address the user's question?

Use `--judge-model` to use a different model for judging, and `--out` to save detailed JSON results.

---

## ⚡ Notes

- Default setup uses **Markdown + dense retrieval**
- Designed to be simple, modular, and easy to extend
- When the LLM says it cannot answer from the context, sources are omitted from the reply

---
