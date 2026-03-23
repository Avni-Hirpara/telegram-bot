from __future__ import annotations

OFF_DOMAIN_REPLY = (
    "I'm your health coach and I only answer questions about health, wellness, "
    "nutrition, fitness, sleep, stress, and similar lifestyle topics. "
    "Ask me something in that area!"
)


def build_messages(context: str, query: str) -> list[dict[str, str]]:
    system = (
        "You are a health assistant.\n\n"
        "Use ONLY the context below to answer.\n\n"
        'If unsure, say "I don\'t know".\n\n'
        "Do not add a Sources, References, or bibliography section to your answer; "
        "the app adds source attribution separately when appropriate."
    )
    user = f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
