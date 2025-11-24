import jsonlines
import chromadb
from chromadb.config import Settings

# Config
EMBEDDINGS_FILE = "wu_embeddings_v1.jsonl"
CHROMA_DB_DIR = "chroma_wu_db"
COLLECTION_NAME = "wu_corpus"

# Initialize Chroma
def init_chroma():
    client = chromadb.PersistentClient(
        path=CHROMA_DB_DIR,
        settings=Settings(
            anonymized_telemetry=False
        )
    )

    # Create or load collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine similarity
        embedding_function=None             # we pass our own embeddings
    )
    return collection


# Load embeddings file
def load_embeddings():
    with jsonlines.open(EMBEDDINGS_FILE, "r") as reader:
        for obj in reader:
            yield obj


# Insert into Chroma
def main():
    collection = init_chroma()

    print(f"Inserting embeddings into Chroma collection: {COLLECTION_NAME}")

    ids = []
    embeddings = []
    metadatas = []
    documents = []

    batch_size = 1000  

    for i, row in enumerate(load_embeddings()):
        ids.append(row["id"])
        embeddings.append(row["embedding"])
        metadatas.append({
            "url": row["url"],
            "title": row["title"],
            "chunk_index": row["chunk_index"],
        })
        documents.append(row["text"])

        # Insert in batches
        if len(ids) >= batch_size:
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            print(f"Inserted {i+1} rows...")
            ids, embeddings, metadatas, documents = [], [], [], []

    # Final flush
    if ids:
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    print("Done. Chroma DB is ready.")
    print(f"Stored at: {CHROMA_DB_DIR}")


if __name__ == "__main__":
    main()