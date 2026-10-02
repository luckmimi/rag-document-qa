# Document Q&A Assistant

A retrieval-augmented question answering system. Documents are chunked, embedded,
and stored in a local vector database; questions retrieve the most relevant
chunks, which are passed to Claude as grounding with inline source citations.

## Corpus

`fetch_docs.py` downloads the documents rather than committing them: three
Apache Spark documentation pages and Disney's fiscal 2025 Form 10-K from SEC
EDGAR, with the filing's HTML flattened to text so financial tables survive as
pipe-separated rows.

The 10-K is pinned to one accession number rather than resolved to "latest" —
the eval set's expected answers are fiscal 2025 figures, and silently pulling a
newer filing would invalidate them.

Two unrelated corpora share one index on purpose, so retrieval has to
discriminate instead of returning whatever is nearest.

## How it works

```
docs/*.{txt,md,pdf}
    -> chunk_text()        250-word windows, 50-word overlap
    -> SentenceTransformer  all-MiniLM-L6-v2, 384-dim embeddings
    -> Chroma               local persistent vector store

question
    -> embed, cosine search -> top 5 chunks
    -> prompt with numbered sources
    -> Claude (claude-opus-5), answers with [n] citations
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

Fetch the corpus, index it, then ask:

```bash
python fetch_docs.py
python ask.py "what does the document say about X?"     # single-shot retrieval
python agent.py "how fast did revenue grow last year?"  # agentic, multi-search
python evals/run_eval.py                                # score against the eval set
```

## Two answering modes

`ask.py` does one retrieval and one model call. Fast and cheap, but it fails on
questions that need facts from two places, because the single query embedding
can only land in one neighbourhood of the index.

`agent.py` gives the model two tools — `search_documents` and `calculate` — and
loops until it stops calling them. It can search several times, narrow to one
file, and compute derived figures. Slower and more expensive per question.

Arithmetic goes through `calculate` rather than the model, which parses the
expression with `ast` and walks a whitelist of operators, so a tool argument
cannot execute arbitrary code.

## Evaluation

`evals/questions.json` holds 20 questions with verified answers, graded two ways:

- **`contains`** — the answer must include exact strings. Used where the ground
  truth is an unambiguous token (`200`, `1.1.0`, `September 27, 2025`).
- **`judge`** — a second model call grades the answer against a rubric. Used
  where correct answers can be worded or rounded differently (`94,425 million`
  vs `$94.4 billion`).

The set deliberately includes cases that should *fail* a naive implementation:

- **Multi-step** questions needing two retrievals plus arithmetic.
- **Out-of-scope** questions (Netflix revenue, Apple's CEO) that pass only if the
  system refuses rather than answering from the model's own knowledge.
- **Cross-domain** questions where the index holds two unrelated corpora, so
  retrieval has to discriminate rather than return whatever is nearest.

## Design notes

**Chunk overlap.** Windows overlap by 50 words so a fact spanning a chunk
boundary still appears whole in at least one chunk.

**Grounding.** The system prompt restricts the model to the retrieved sources and
requires inline citations, so answers can be traced back and unsupported claims
are visible. The model is instructed to say when the sources do not answer the
question rather than fall back on its own knowledge.

**Local embeddings.** `sentence-transformers` runs on the CPU with no API key and
no per-call cost, which keeps ingestion of large document sets free.
