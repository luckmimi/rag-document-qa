"""Chunk every document in docs/, embed the chunks, and store them in Chroma.

Usage: python ingest.py
"""

from pathlib import Path

from chunking import chunk_text, load_documents
from store import embed, get_collection

DOCS_DIR = Path(__file__).parent / "docs"
BATCH_SIZE = 64


def main() -> None:
    documents = load_documents(DOCS_DIR)
    if not documents:
        print(f"No .txt, .md, or .pdf files found in {DOCS_DIR}")
        return

    collection = get_collection()
    total = 0

    for filename, text in documents:
        chunks = chunk_text(text)
        if not chunks:
            print(f"  {filename}: no text extracted, skipping")
            continue

        ids = [f"{filename}:{i}" for i in range(len(chunks))]
        metadatas = [{"source": filename, "chunk": i} for i in range(len(chunks))]

        for start in range(0, len(chunks), BATCH_SIZE):
            end = start + BATCH_SIZE
            collection.upsert(
                ids=ids[start:end],
                documents=chunks[start:end],
                embeddings=embed(chunks[start:end]),
                metadatas=metadatas[start:end],
            )

        total += len(chunks)
        print(f"  {filename}: {len(chunks)} chunks")

    print(f"\nIngested {total} chunks from {len(documents)} documents.")


if __name__ == "__main__":
    main()
