#!/usr/bin/env python3
"""LLM-as-a-judge evaluation: run queries through the full RAG pipeline, then
ask the same (or a different) model to score each answer on faithfulness and
relevance.

Usage (after ingest + Ollama running):
  python scripts/eval_response.py
  python scripts/eval_response.py --judge-model mistral --out results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from health_rag.config import Settings  # noqa: E402
from health_rag.llm.ollama_client import stream_chat  # noqa: E402
from health_rag.logging_config import configure_logging  # noqa: E402
from health_rag.services.rag_service import RAGService  # noqa: E402

# ------------------------------------------------------------------
# Eval query set — mix of on-topic, off-topic, and edge cases
# ------------------------------------------------------------------

EVAL_QUERIES = [
    "how to eat healthy in India",
    "what are good exercises for beginners",
    "how stress affects diabetes",
    "what should kids eat for growth",
    "how to manage high blood pressure",
    "tips for better sleep",
    "how to build a rocket",
    "what is the capital of France",
    "type 2 diabetes prevention tips",
]


# ------------------------------------------------------------------
# Judge prompt — asks the LLM to return structured JSON scores
# ------------------------------------------------------------------

JUDGE_SYSTEM = textwrap.dedent("""\
    You are an evaluation judge for a health RAG system. You will be given:
    - A user QUESTION
    - The CONTEXT (retrieved document passages) that was provided to the answering model
    - The ANSWER the model produced

    Score the answer on two criteria (1-5 scale):

    FAITHFULNESS: Is every claim in the answer supported by the context?
      5 = fully grounded, no hallucination
      3 = mostly grounded, minor unsupported detail
      1 = significant hallucination or fabrication

    RELEVANCE: Does the answer address the user's question?
      5 = directly and completely answers the question
      3 = partially answers or is tangential
      1 = does not address the question at all

    Special cases:
    - If the answer correctly says "I don't know" or similar because the context
      lacks information, score faithfulness=5 and relevance=3 (honest but unhelpful).
    - If the question is off-topic (not health-related) and the system refused to
      answer, score both faithfulness=5 and relevance=5 (correct behavior).

    Respond with ONLY a JSON object, no markdown fences:
    {"faithfulness": <int>, "relevance": <int>, "rationale": "<one sentence>"}
""")


def _build_judge_prompt(question: str, context: str, answer: str) -> list[dict[str, str]]:
    user_msg = (
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{context or '(none — off-topic or no retrieval hits)'}\n\n"
        f"ANSWER:\n{answer}"
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user_msg},
    ]


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class EvalResult:
    question: str
    tag: str
    answer: str
    sources: list[str]
    faithfulness: int
    relevance: int
    rationale: str


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

async def _collect_stream(rag: RAGService, messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    async for piece in rag.stream_llm(messages):
        parts.append(piece)
    return "".join(parts).strip()


async def _judge_answer(
    host: str, model: str, question: str, context: str, answer: str,
) -> tuple[int, int, str]:
    """Ask the judge model to score a single answer. Returns (faithfulness, relevance, rationale)."""
    messages = _build_judge_prompt(question, context, answer)
    parts: list[str] = []
    async for piece in stream_chat(host=host, model=model, messages=messages):
        parts.append(piece)
    raw = "".join(parts).strip()
    # Strip markdown fences if the model wraps its JSON
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        obj = json.loads(raw)
        return (
            int(obj.get("faithfulness", 0)),
            int(obj.get("relevance", 0)),
            str(obj.get("rationale", "")),
        )
    except (json.JSONDecodeError, ValueError):
        return 0, 0, f"judge parse error: {raw[:200]}"


async def _run_eval(settings: Settings, judge_model: str) -> list[EvalResult]:
    rag = RAGService(settings)
    results: list[EvalResult] = []

    for question in EVAL_QUERIES:
        tag, messages, chunks = rag.prepare_ask(question, skip_domain_gate=False)

        context_text = "\n\n".join(c.text[:500] for c in chunks) if chunks else ""
        source_files = list(dict.fromkeys(c.source for c in chunks if c.source))

        if tag == "off_domain":
            answer = "(off-topic — refused by domain gate)"
        elif tag == "no_hits":
            answer = "(no retrieval hits)"
        elif messages:
            answer = await _collect_stream(rag, messages) or "I don't know."
        else:
            answer = "(empty)"

        faith, rel, rationale = await _judge_answer(
            settings.ollama_host, judge_model, question, context_text, answer,
        )

        results.append(EvalResult(
            question=question,
            tag=tag,
            answer=answer,
            sources=source_files,
            faithfulness=faith,
            relevance=rel,
            rationale=rationale,
        ))

        # Live progress
        status = "✓" if faith >= 4 and rel >= 4 else "△" if faith >= 3 else "✗"
        print(f"  {status} F={faith} R={rel}  {question}")

    return results


# ------------------------------------------------------------------
# Output
# ------------------------------------------------------------------

def _print_summary(results: list[EvalResult]) -> None:
    scored = [r for r in results if r.faithfulness > 0]
    if not scored:
        print("\nNo valid judge scores — check Ollama connectivity.")
        return

    avg_f = sum(r.faithfulness for r in scored) / len(scored)
    avg_r = sum(r.relevance for r in scored) / len(scored)

    print(f"\n{'=' * 72}")
    print(f"{'Question':<45} {'Tag':<12} {'F':>2} {'R':>2}  Rationale")
    print(f"{'-' * 72}")
    for r in results:
        q = r.question[:43] + "…" if len(r.question) > 44 else r.question
        rat = r.rationale[:40] + "…" if len(r.rationale) > 40 else r.rationale
        print(f"{q:<45} {r.tag:<12} {r.faithfulness:>2} {r.relevance:>2}  {rat}")
    print(f"{'-' * 72}")
    print(f"{'Averages':<45} {'':12} {avg_f:4.1f} {avg_r:4.1f}")
    print(f"{'=' * 72}")
    print(f"\nQueries: {len(results)}  |  Avg faithfulness: {avg_f:.2f}/5  |  Avg relevance: {avg_r:.2f}/5")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-as-a-judge RAG evaluation.")
    parser.add_argument(
        "--judge-model", default=None,
        help="Ollama model for judging (default: same as OLLAMA_MODEL).",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Write detailed results to a JSON file.",
    )
    args = parser.parse_args()

    configure_logging("WARNING")
    settings = Settings()
    judge_model = args.judge_model or settings.ollama_model

    print(f"RAG model : {settings.ollama_model}")
    print(f"Judge model: {judge_model}")
    print(f"Queries    : {len(EVAL_QUERIES)}\n")

    results = asyncio.run(_run_eval(settings, judge_model))
    _print_summary(results)

    if args.out:
        args.out.write_text(
            json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nDetailed results written to {args.out}")


if __name__ == "__main__":
    main()
