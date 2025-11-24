import jsonlines
import requests
from collections import Counter
import re

API_URL = "http://127.0.0.1:8000/answer_json"
EVAL_FILE = "wu_eval_set.jsonl"

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

def main():
    total = 0
    exact_matches = 0
    macro_f1 = []

    with jsonlines.open(EVAL_FILE) as reader:
        for item in reader:
            gold = item["answer"]
            q = item["question"]

            response = requests.post(API_URL, json={"query": q, "top_k": 5})

            # Handle non-200 or non-JSON responses gracefully so the script
            # doesn't crash when the server returns HTML or an error page.
            if not response.ok:
                print(f"Warning: request failed (status={response.status_code}) for question: {q}")
                print("Response text:", response.text[:1000])

            try:
                data = response.json()
            except requests.exceptions.JSONDecodeError:
                # Print a short preview of the body to help debugging (truncated)
                print(f"Warning: could not parse JSON response (status={response.status_code}) for question: {q}")
                print("Response text:", response.text[:1000])
                pred = ""
            except Exception as e:
                print(f"Unexpected error parsing response JSON: {e}")
                print("Response text:", response.text[:1000])
                pred = ""
            else:
                pred = data.get("answer", "")

            # Exact match (rare but used as a baseline)
            if gold.lower().strip() in pred.lower():
                exact_matches += 1

            # F1
            _, _, f1 = f1_score(pred, gold)
            macro_f1.append(f1)

            total += 1

    print(f"RAG Exact-Match Accuracy: {exact_matches/total:.2f}")
    print(f"RAG Macro-F1: {sum(macro_f1)/len(macro_f1):.2f}")

if __name__ == "__main__":
    main()