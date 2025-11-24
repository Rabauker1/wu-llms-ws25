import os
# Prevent `tokenizers` parallelism deadlock warning when processes are forked.
# Set before any libraries that may use Hugging Face tokenizers are imported.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import chromadb
import json
import subprocess
import json
import re
import logging
from html import escape
from typing import Optional
import os

# Config
CHROMA_DB_DIR = "chroma_wu_db"
COLLECTION_NAME = "wu_corpus"
LLAMA_MODEL = "llama3.1:8b"       # Ollama model (use ollama list to see available models)
TOP_K = 5

# Lazy-initialized Chroma client/collection to avoid import-time DB loading
_chroma_client: Optional[chromadb.PersistentClient] = None
_chroma_collection = None


def get_chroma_collection():
    """Return a cached Chroma collection. Initialize lazily and surface
    informative errors instead of failing at import time (which breaks
    uvicorn worker startup)."""
    global _chroma_client, _chroma_collection
    if _chroma_collection is not None:
        return _chroma_collection

    # Prefer configured dir; if missing, try a common backup suffix
    db_path = CHROMA_DB_DIR
    if not os.path.isdir(db_path):
        alt_path = f"{db_path}.bak"
        if os.path.isdir(alt_path):
            logging.warning("Chroma DB dir '%s' not found; using fallback '%s'", db_path, alt_path)
            db_path = alt_path

    # First try persistent client at chosen path
    try:
        _chroma_client = chromadb.PersistentClient(path=db_path)
        _chroma_collection = _chroma_client.get_collection(COLLECTION_NAME)
        return _chroma_collection
    except Exception as e:
        logging.exception("Failed to initialize Chroma collection at %s", db_path)
        # Try a safe fallback to an in-memory client so the API can still run
        try:
            logging.info("Attempting fallback to in-memory Chroma client")
            _chroma_client = chromadb.Client()
            try:
                _chroma_collection = _chroma_client.get_collection(COLLECTION_NAME)
            except Exception:
                _chroma_collection = _chroma_client.create_collection(name=COLLECTION_NAME)
            return _chroma_collection
        except Exception as e2:
            logging.exception("Fallback to in-memory Chroma client failed")
            raise RuntimeError(
                f"Failed to open or create Chroma DB at '{CHROMA_DB_DIR}' (tried '{db_path}'): {e}; fallback error: {e2}"
            ) from e2

# FastAPI App
app = FastAPI(title="WU RAG API", version="1.0")


# Request Body
class QueryRequest(BaseModel):
    query: str
    top_k: int | None = TOP_K


# Call Ollama locally
def call_ollama(prompt: str, model: str = LLAMA_MODEL) -> str:
    """Calls local Ollama via subprocess and returns response."""
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt,
        text=True,
        capture_output=True
    )
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    if result.returncode != 0:
        logging.error("ollama failed (rc=%s): %s", result.returncode, stderr)
        raise RuntimeError(f"ollama failed (rc={result.returncode}): {stderr}")
    if not stdout:
        # No output from the model is unexpected; surface as an error so callers
        # don't silently return empty answers.
        logging.error("ollama returned empty stdout; stderr: %s", stderr)
        raise RuntimeError(f"ollama returned empty output: {stderr}")

    return stdout


# Build RAG prompt
def build_prompt(question: str, retrieved_docs: list) -> str:
    context_blocks = []

    for doc in retrieved_docs:
        ctx = f"""Source: {doc['metadata']['title']} ({doc['metadata']['url']})
Content: {doc['document']}
"""
        context_blocks.append(ctx)

    context_text = "\n---\n".join(context_blocks)

    prompt = f"""
You are an expert assistant for WU Vienna. Answer the user's question using ONLY the context below.
If the answer is not present in the context, say "I don't know based on the provided WU documents."

Context:
{context_text}

Question: {question}

Provide a clear and concise answer. At the end, list the sources you used as bullet points.
"""
    return prompt

# API Endpoint: full RAG
@app.post("/answer")
def answer(req: QueryRequest):
    query = req.query
    top_k = req.top_k or TOP_K

    # 1. Retrieve top-k chunks from Chroma (initialize lazily)
    try:
        collection = get_chroma_collection()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        results = collection.query(
            query_texts=[query],
            n_results=top_k
        )
    except Exception as e:
        logging.exception("Error querying Chroma collection")
        raise HTTPException(status_code=500, detail=f"Chroma query failed: {e}")

    docs = []
    for i in range(len(results["documents"][0])):
        docs.append({
            "id": results["ids"][0][i],
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
        })

    # 2. Build prompt
    prompt = build_prompt(query, docs)

    # 3. Generate answer with Ollama
    try:
        llm_output = call_ollama(prompt)
    except RuntimeError as e:
        logging.exception("LLM call failed for HTML endpoint")
        llm_output = f"Error generating answer: {e}"

    # Build an HTML response showing the query, the LLM answer, and citations.
    safe_query = escape(query)
    safe_answer = escape(llm_output)

    citations_html = ""
    if docs:
        items = []
        for d in docs:
            title = escape(d["metadata"].get("title", "(no title)"))
            url = escape(d["metadata"].get("url", ""))
            chunk = escape(str(d["metadata"].get("chunk_index", "")))
            items.append(f"<li><a href=\"{url}\">{title}</a> (chunk {chunk})</li>")
        citations_html = "\n".join(items)

    html_content = f"""
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width,initial-scale=1" />
        <title>RAG Answer</title>
        <style>body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial; padding: 20px; }} pre {{ background:#111a27; padding:12px; border-radius:6px; white-space:pre-wrap; }}</style>
    </head>
    <body>
        <h1>RAG Answer</h1>
        <p><strong>Query:</strong> {safe_query}</p>
        <h2>Answer</h2>
        <pre>{safe_answer}</pre>
        <h3>Sources</h3>
        <ul>
            {citations_html}
        </ul>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)


# JSON API: programmatic RAG responses for clients/tests
@app.post("/answer_json")
def answer_json(req: QueryRequest):
    query = req.query
    top_k = req.top_k or TOP_K

    try:
        collection = get_chroma_collection()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        results = collection.query(
            query_texts=[query],
            n_results=top_k
        )
    except Exception as e:
        logging.exception("Error querying Chroma collection")
        raise HTTPException(status_code=500, detail=f"Chroma query failed: {e}")

    docs = []
    for i in range(len(results["documents"][0])):
        docs.append({
            "id": results["ids"][0][i],
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
        })

    prompt = build_prompt(query, docs)
    try:
        llm_output = call_ollama(prompt)
    except RuntimeError as e:
        logging.exception("LLM call failed for JSON endpoint")
        raise HTTPException(status_code=500, detail=str(e))

    sources = []
    for d in docs:
        sources.append({
            "title": d["metadata"].get("title", ""),
            "url": d["metadata"].get("url", ""),
            "chunk_index": d["metadata"].get("chunk_index", ""),
        })

    return JSONResponse(content={"answer": llm_output, "sources": sources}, status_code=200)


@app.get("/")
def root():
        html_content = f"""
        <!doctype html>
        <html lang="en">
        <head>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width,initial-scale=1" />
            <title>WU RAG API</title>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial; padding: 24px; }}
                pre {{ background:#f6f8fa; padding:12px; border-radius:6px; white-space:pre-wrap; }}</style>
        </head>
        <body>
            <h1>WU RAG API</h1>
            <p><strong>Status:</strong> RAG API running</p>
            <p><strong>Model:</strong> {escape(LLAMA_MODEL)}</p>
            <p>POST to <code>/answer</code> with JSON <code>{"query": "..."}</code></p>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content, status_code=200)