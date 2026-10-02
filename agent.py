"""Agentic question answering: Claude decides what to search and when to stop.

Unlike ask.py, which retrieves once and answers, this drives a loop so the model
can issue several searches, do arithmetic on what it finds, and keep going until
it has enough to answer.

Usage: python agent.py "your question here"
"""

import sys

import anthropic

from tools import TOOL_SCHEMAS, run_tool

MODEL = "claude-opus-5"
MAX_TURNS = 10

SYSTEM = """You answer questions about a set of indexed documents.

Rules:
- Search the documents before answering. Never answer from prior knowledge.
- Search more than once when a question needs facts from different places.
- Use the calculate tool for any arithmetic. Do not compute in your head.
- Cite the source filename and chunk number for each fact you use.
- If the documents do not contain the answer, say so plainly. Do not guess."""


def answer(question: str, verbose: bool = False) -> str:
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": question}]

    for _ in range(MAX_TURNS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )

        if response.stop_reason == "refusal":
            return "The model declined to answer this question."

        if response.stop_reason != "tool_use":
            return "".join(b.text for b in response.content if b.type == "text")

        messages.append({"role": "assistant", "content": response.content})

        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            output, is_error = run_tool(block.name, block.input)
            if verbose:
                print(f"  -> {block.name}({block.input})")
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": output,
                "is_error": is_error,
            })

        messages.append({"role": "user", "content": results})

    return f"Stopped after {MAX_TURNS} turns without a final answer."


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python agent.py "your question here"')
        sys.exit(1)

    print(f"\n{answer(' '.join(sys.argv[1:]), verbose=True)}\n")


if __name__ == "__main__":
    main()
