"""Load documents from disk and split them into overlapping chunks."""

from pathlib import Path

from pypdf import PdfReader

CHUNK_WORDS = 250
OVERLAP_WORDS = 50


def load_documents(docs_dir: Path) -> list[tuple[str, str]]:
    """Return (filename, text) for every .txt, .md, and .pdf under docs_dir."""
    documents = []
    for path in sorted(docs_dir.iterdir()):
        if path.suffix.lower() in {".txt", ".md"}:
            documents.append((path.name, path.read_text(encoding="utf-8")))
        elif path.suffix.lower() == ".pdf":
            pages = [page.extract_text() or "" for page in PdfReader(path).pages]
            documents.append((path.name, "\n".join(pages)))
    return documents


def chunk_text(text: str, chunk_words: int = CHUNK_WORDS,
               overlap_words: int = OVERLAP_WORDS) -> list[str]:
    """Split text into word windows that overlap, so answers spanning a
    boundary are still retrievable from at least one chunk."""
    words = text.split()
    if not words:
        return []

    step = chunk_words - overlap_words
    chunks = []
    for start in range(0, len(words), step):
        chunk = words[start:start + chunk_words]
        if chunk:
            chunks.append(" ".join(chunk))
        if start + chunk_words >= len(words):
            break
    return chunks
