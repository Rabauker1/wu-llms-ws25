import os
# Prevent `tokenizers` parallelism deadlock warning when processes are forked.
# Set before importing libraries that may use Hugging Face tokenizers.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import jsonlines
import itertools

# CONFIG
INPUT_FILE = "wu_corpus_clean_v1.json"       # output of step 1
OUTPUT_FILE = "wu_embeddings_v1.jsonl"       # embeddings + metadata result
MODEL_NAME = "BAAI/bge-small-en"                # smaller model for faster speed
#MODEL_NAME = "BAAI/bge-m3"                # Embedding model

# Load cleaned chunks
def load_corpus():
    # Support both JSONL (one JSON object per line) and a JSON array file.
    # Limit to the first 5000 chunks to keep memory/time reasonable.
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        # Peek the first non-whitespace character to determine format
        pos = f.tell()
        first = f.read(1)
        while first and first.isspace():
            pos = f.tell()
            first = f.read(1)

        if not first:
            return []

        f.seek(pos)

        if first == '[':
            # JSON array file
            arr = json.load(f)
            return arr[:5000]
        else:
            # JSONL: read line-by-line (we didn't consume the first line)
            data = []
            for i, line in enumerate(itertools.islice(f, 5000)):
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
            return data
    
    


# Embedding generator
def main():
    print("Loading model:", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    corpus = load_corpus()
    print(f"Loaded {len(corpus)} chunks")

    with jsonlines.open(OUTPUT_FILE, mode="w") as writer:
        for i, doc in enumerate(corpus):
            text = doc["text"]

            # Generate embedding
            embedding = model.encode(text, normalize_embeddings=True).tolist()

            # Write one row per chunk (embedding + full metadata)
            writer.write({
                "id": doc["id"],
                "url": doc["url"],
                "title": doc["title"],
                "chunk_index": doc["chunk_index"],
                "text": text,
                "embedding": embedding
            })

            if (i + 1) % 200 == 0:
                print(f"Embedded {i+1}/{len(corpus)} chunks...")

    print(f"Done. Saved embeddings → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()