import jsonlines
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from difflib import SequenceMatcher
import re

# Config
CHROMA_DB_DIR = "chroma_wu_db"
COLLECTION = "wu_corpus"
EVAL_FILE = "wu_eval_set.jsonl"
K = 5
# Use the same embedding model as used to build the DB
# Must match `MODEL_NAME` in `embeddings.py` (used to create `wu_embeddings_v1.jsonl`)
EMBED_MODEL_NAME = "BAAI/bge-small-en"
#EMBED_MODEL_NAME = "BAAI/bge-m3"

def normalize(text):
    return " ".join(text.lower().strip().split())

def main():
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    col = client.get_collection(COLLECTION)

    # Load embedding model to produce query embeddings matching the collection
    model = SentenceTransformer(EMBED_MODEL_NAME)

    total = 0
    hits = 0

    with jsonlines.open(EVAL_FILE) as reader:
        for item in reader:
            q = item["question"]
            gold = normalize(item["answer"])

            # Encode query with same model used to create the DB embeddings
            q_emb = model.encode(q, normalize_embeddings=True).tolist()

            result = col.query(
                query_embeddings=[q_emb],
                n_results=K,
                include=['documents', 'metadatas', 'distances']
            )

            docs = [normalize(d) for d in result["documents"][0]]

            # (debug prints removed)

            # Retrieval hit rules:
            # - exact substring of first 60 chars (original brittle check), OR
            # - fuzzy match: SequenceMatcher ratio above threshold
            def fuzzy_match(a, b, thr=0.45):
                return SequenceMatcher(None, a, b).ratio() >= thr

            # token-overlap based fuzzy match (robust to paraphrase)
            STOPWORDS = set(["the", "and", "is", "in", "of", "a", "an", "to", "for", "with", "on", "as", "by", "its", "it", "that", "are", "be", "was", "this", "these"])
            def token_overlap(a, b, thr=0.40):
                # simple tokenizer: lowercase, remove punctuation, split
                ta = [w for w in re.sub(r"[^a-z0-9\s]"," ", a.lower()).split() if w and w not in STOPWORDS]
                tb = [w for w in re.sub(r"[^a-z0-9\s]"," ", b.lower()).split() if w and w not in STOPWORDS]
                if not ta:
                    return False
                sa = set(ta)
                sb = set(tb)
                return (len(sa & sb) / len(sa)) >= thr

            hit = any(gold[:60] in d for d in docs) or any(fuzzy_match(gold, d) for d in docs) or any(token_overlap(gold, d) for d in docs)

            if hit:
                hits += 1
            total += 1

    recall = hits / total
    print(f"Retrieval Recall@{K}: {recall:.2f}   ({hits}/{total})")

if __name__ == "__main__":
    main()