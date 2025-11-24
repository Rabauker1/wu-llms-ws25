
import json
import re
from pathlib import Path
from bs4 import BeautifulSoup

try:
    import trafilatura
except ImportError:
    trafilatura = None

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    RecursiveCharacterTextSplitter = None

# Config
INPUT_FILE = "wu_docs-initial.jsonl"         # raw scraped JSON
OUTPUT_FILE = "wu_corpus_clean.json"   # cleaned + chunked output
CHUNK_SIZE = 800                       # chars ≈ 600–1000 tokens
CHUNK_OVERLAP = 100

# Helper: clean raw HTML
def clean_html(html: str) -> str:
    if not html:
        return ""

    extracted = None
    if trafilatura:
        extracted = trafilatura.extract(html, include_comments=False)
    if extracted:
        text = extracted
    else:
        # fallback to BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator=" ")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Load input JSON
def load_documents(path: str):
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    return docs


# Chunk text into RAG-ready segments
def chunk_text(text: str):
    if RecursiveCharacterTextSplitter is None:
        return _simple_chunks(text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)


def _simple_chunks(text: str):
    text = text.strip()
    if not text:
        return []

    step = max(CHUNK_SIZE - CHUNK_OVERLAP, 1)
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + CHUNK_SIZE)
        chunks.append(text[start:end])
        start += step
    return chunks


# Main preprocess function
def preprocess():
    raw = load_documents(INPUT_FILE)
    cleaned_docs = []

    for idx, doc in enumerate(raw):
        url = doc.get("url", "")
        title = doc.get("title", "")

        # 1. Extract text: prefer HTML field, fall back to plain 'text' field
        # Some inputs (like the provided JSONL) use 'text' instead of 'html'.
        raw_html = doc.get("html")
        raw_text_field = doc.get("text")
        source_content = raw_html if raw_html else (raw_text_field if raw_text_field else "")
        text = clean_html(source_content)

        # Skip empty pages
        if len(text) < 100:
            continue

        # 2. Chunk
        chunks = chunk_text(text)

        # 3. Save each chunk with metadata
        for chunk_i, chunk in enumerate(chunks):
            cleaned_docs.append({
                "id": f"wu_{idx}_{chunk_i}",
                "url": url,
                "title": title,
                "chunk_index": chunk_i,
                "text": chunk
            })

    # Save output
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(cleaned_docs, f, indent=2, ensure_ascii=False)

    print(f"Done. Saved {len(cleaned_docs)} cleaned chunks → {OUTPUT_FILE}")


if __name__ == "__main__":
    preprocess()
