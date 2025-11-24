from flask import Flask, render_template, request
import requests
import jsonlines
import re
import os
from collections import Counter

# Config
FASTAPI_URL = "http://127.0.0.1:8000/answer"  # RAG backend
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_FILE = os.path.join(BASE_DIR, "wu_eval_set.jsonl")

app = Flask(__name__)

def normalize(s):
    return " ".join(s.lower().strip().split())

def tokenize(s):
    return re.findall(r"\w+", s.lower())

def f1_score(pred, gold):
    pred_tokens = Counter(tokenize(pred))
    gold_tokens = Counter(tokenize(gold))

    common = pred_tokens & gold_tokens
    overlap = sum(common.values())

    if overlap == 0:
        return 0, 0, 0

    precision = overlap / sum(pred_tokens.values())
    recall = overlap / sum(gold_tokens.values())
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

# Load eval set into memory
EVAL_DATA = {}
try:
    with jsonlines.open(EVAL_FILE) as reader:
        for item in reader:
            EVAL_DATA[item["question"]] = item["answer"]
except FileNotFoundError:
    print(f"Warning: {EVAL_FILE} not found. Metrics will not be available.")
except Exception as e:
    print(f"Warning: Could not load eval set: {e}")

# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    citations = None
    metrics = None
    gold_answer = None
    message = None
    is_html = False

    if request.method == "POST":
        question = request.form.get("question", "").strip()

        if not question:
            message = "Please enter a question."
        else:
            try:
                # Send to RAG backend with a short timeout
                resp = requests.post(FASTAPI_URL, json={"query": question, "top_k": 5}, timeout=60)
                resp.raise_for_status()

                # Check if response is HTML
                content_type = resp.headers.get('Content-Type', '')
                if 'text/html' in content_type:
                    answer = resp.text
                    is_html = True
                    citations = None
                else:
                    try:
                        data = resp.json()
                    except ValueError:
                        message = "Received invalid JSON from the backend."
                        data = {}

                    answer = data.get("answer")
                    citations = data.get("citations")

                    if answer:
                        message = "Answer received."
                    else:
                        # If backend returned a status OK but no answer field
                        message = data.get("message") or "No answer returned from backend."

            except requests.exceptions.Timeout:
                message = "Request to backend timed out (60s)."
            except requests.exceptions.ConnectionError:
                message = f"Could not connect to RAG backend at {FASTAPI_URL}. Is it running?"
            except requests.exceptions.HTTPError as e:
                # Try to show backend's JSON error message if available
                try:
                    err = resp.json()
                    message = err.get("detail") or str(e)
                except Exception:
                    message = f"Backend returned HTTP error: {e}"
            except Exception as e:
                message = f"Unexpected error: {e}"

        # Compute metrics only if question matches eval set and we have an answer
        if question in EVAL_DATA and answer:
            gold_answer = EVAL_DATA[question]
            em = 1 if normalize(gold_answer) in normalize(answer) else 0
            p, r, f1 = f1_score(answer, gold_answer)

            metrics = {
                "exact_match": round(em, 3),
                "precision": round(p, 3),
                "recall": round(r, 3),
                "f1": round(f1, 3),
            }

    return render_template(
        "index.html",
        answer=answer,
        citations=citations,
        metrics=metrics,
        gold_answer=gold_answer,
        message=message,
        is_html=is_html,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)