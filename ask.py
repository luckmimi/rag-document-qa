"""Answer a question using the ingested documents as grounding.

Usage: python ask.py "your question here"
"""

import sys

import anthropic

from store import embed, get_collection

MODEL = "claude-opus-5"
TOP_K = 5

SYSTEM = """You answer questions using only the numbered sources provided.

Rules:
- Base every claim on the sources. Do not use outside knowledge.
- Cite the source number inline, like [2], after each claim.
- If the sources do not contain the answer, say so plainly. Do not guess."""


def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    collection = get_collection()
    results = collection.query(query_embeddings=embed([question]), n_results=top_k)
    return [
        {"text": text, "source": meta["source"], "chunk": meta["chunk"]}
        for text, meta in zip(results["documents"][0], results["metadatas"][0])
    ]


def build_prompt(question: str, passages: list[dict]) -> str:
    sources = "\n\n".join(
        f"[{i}] (from {p['source']}, chunk {p['chunk']})\n{p['text']}"
        for i, p in enumerate(passages, start=1)
    )
    return f"Sources:\n\n{sources}\n\nQuestion: {question}"


def answer(question: str) -> tuple[str, list[dict]]:
    passages = retrieve(question)
    if not passages:
        return "No documents have been ingested yet. Run ingest.py first.", []

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM,
        messages=[{"role": "user", "content": build_prompt(question, passages)}],
    )

    if response.stop_reason == "refusal":
        return "The model declined to answer this question.", passages

    text = "".join(b.text for b in response.content if b.type == "text")
    return text, passages


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python ask.py "your question here"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    text, passages = answer(question)

    print(f"\n{text}\n")
    if passages:
        print("Sources:")
        for i, p in enumerate(passages, start=1):
            print(f"  [{i}] {p['source']} (chunk {p['chunk']})")


if __name__ == "__main__":
    main()
