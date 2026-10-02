"""Download the corpus into docs/.

The documents are public but bulky, so they are fetched rather than committed.

The Disney filing is pinned to a specific accession number, not resolved to
"latest": the expected answers in evals/questions.json are the fiscal 2025
figures, and pointing this at a newer 10-K would silently invalidate them.

Usage: python fetch_docs.py
"""

import re
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

DOCS_DIR = Path(__file__).parent / "docs"

# SEC requires a descriptive User-Agent with contact details on API requests.
USER_AGENT = "rag-document-qa (lixuemin21@gmail.com)"

SPARK_BASE = "https://raw.githubusercontent.com/apache/spark/master/docs"
SPARK_DOCS = {
    "spark-config.md": f"{SPARK_BASE}/configuration.md",
    "spark-tuning.md": f"{SPARK_BASE}/sql-performance-tuning.md",
    "spark-tuning-guide.md": f"{SPARK_BASE}/tuning.md",
}

DISNEY_10K_URL = (
    "https://www.sec.gov/Archives/edgar/data/1744489/"
    "000174448925000155/dis-20250927.htm"
)
DISNEY_FILENAME = "disney-10k-fy2025.txt"

# Inline XBRL puts a long block of machine-readable facts ahead of the document
# body; the filing proper starts at the cover page.
DISNEY_BODY_START = "UNITED STATES\n\nSECURITIES AND EXCHANGE COMMISSION"

BLOCK_TAGS = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "table"}
CELL_TAGS = {"td", "th"}


class _TextExtractor(HTMLParser):
    """Flatten HTML to text, keeping table cells separated by pipes so that
    financial tables stay readable after extraction."""

    def __init__(self):
        super().__init__()
        self.parts = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip_depth += 1
        if tag in BLOCK_TAGS:
            self.parts.append("\n")
        if tag in CELL_TAGS:
            self.parts.append(" | ")

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip_depth:
            self._skip_depth -= 1
        if tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip_depth:
            self.parts.append(data)


def download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request) as response:
        return response.read()


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    text = "".join(parser.parts).replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return re.sub(r"(\| *)+\|", " | ", text).strip()


def main() -> None:
    DOCS_DIR.mkdir(exist_ok=True)

    for filename, url in SPARK_DOCS.items():
        (DOCS_DIR / filename).write_bytes(download(url))
        print(f"  {filename}")

    html = download(DISNEY_10K_URL).decode("utf-8", errors="replace")
    text = html_to_text(html)

    start = text.find(DISNEY_BODY_START)
    if start == -1:
        raise SystemExit(
            "Could not locate the filing body. The SEC document layout may have "
            "changed; check DISNEY_BODY_START."
        )

    (DOCS_DIR / DISNEY_FILENAME).write_text(text[start:].strip(), encoding="utf-8")
    print(f"  {DISNEY_FILENAME}")

    print(f"\nFetched {len(SPARK_DOCS) + 1} documents into {DOCS_DIR}/")
    print("Next: python ingest.py")


if __name__ == "__main__":
    main()
