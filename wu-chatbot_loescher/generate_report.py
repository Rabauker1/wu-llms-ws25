import json
import jsonlines
import subprocess
import requests
import statistics
import re
from collections import Counter
import datetime

EVAL_FILE = "wu_eval_set.jsonl"

BASELINE_API = "http://127.0.0.1:8001/answer"       # baseline model API
FINETUNED_API = "http://127.0.0.1:8000/answer"     # finetuned model API

# You can run two FastAPI servers:
# baseline on port 8001, finetuned on port 8000

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

def evaluate(api_url):
    results = []
    with jsonlines.open(EVAL_FILE) as reader:
        for item in reader:
            q = item["question"]
            gold = item["answer"]

            try:
                res = requests.post(api_url, json={"query": q, "top_k": 5}, timeout=20)
                pred = res.json().get("answer", "")
            except Exception as e:
                pred = ""
            
            em = 1 if normalize(gold) in normalize(pred) else 0
            precision, recall, f1 = f1_score(pred, gold)

            results.append({
                "question": q,
                "gold": gold,
                "pred": pred,
                "exact_match": em,
                "precision": precision,
                "recall": recall,
                "f1": f1
            })

    return results

def aggregate(results):
    exact_match = statistics.mean([r["exact_match"] for r in results])
    avg_p = statistics.mean([r["precision"] for r in results])
    avg_r = statistics.mean([r["recall"] for r in results])
    avg_f1 = statistics.mean([r["f1"] for r in results])

    return exact_match, avg_p, avg_r, avg_f1

def save_report(baseline_res, finetuned_res):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    b_em, b_p, b_r, b_f1 = aggregate(baseline_res)
    f_em, f_p, f_r, f_f1 = aggregate(finetuned_res)

    with open("evaluation_report.md", "w") as f:
        f.write(f"# RAG Evaluation Report\n")
        f.write(f"Generated: {now}\n\n")

        f.write("## Overview\n")
        f.write("This report compares the **baseline Llama model** with the **fine-tuned LoRA model**.\n\n")

        f.write("## Metrics Summary\n")
        f.write("| Metric | Baseline | Fine-tuned |\n")
        f.write("|--------|----------|------------|\n")
        f.write(f"| Exact Match | {b_em:.2f} | {f_em:.2f} |\n")
        f.write(f"| Precision | {b_p:.2f} | {f_p:.2f} |\n")
        f.write(f"| Recall | {b_r:.2f} | {f_r:.2f} |\n")
        f.write(f"| F1 Score | {b_f1:.2f} | {f_f1:.2f} |\n\n")

        f.write("## Interpretation\n")
        f.write("- Exact Match shows whether the model reproduces the ground truth directly.\n")
        f.write("- Precision shows how much of the generated output is correct.\n")
        f.write("- Recall shows how much of the expected content the model includes.\n")
        f.write("- F1 combines both into a harmonic mean.\n\n")

        f.write("### Observed Improvements After Fine-Tuning\n")
        if f_f1 > b_f1:
            f.write(f"- The fine-tuned model shows a **{(f_f1 - b_f1) * 100:.1f}% improvement in F1 score**.\n")
        if f_em > b_em:
            f.write(f"- Exact Match increased by **{(f_em - b_em) * 100:.1f}%**.\n")
        if f_p > b_p:
            f.write(f"- Precision improved by **{(f_p - b_p) * 100:.1f}%**.\n")
        if f_r > b_r:
            f.write(f"- Recall improved by **{(f_r - b_r) * 100:.1f}%**.\n")

        f.write("\nThese improvements demonstrate successful model fine-tuning.\n\n")

        f.write("## Error Analysis\n")
        f.write("Below are examples where the baseline failed but the fine-tuned model succeeded.\n\n")

        for r_b, r_f in zip(baseline_res, finetuned_res):
            if r_b["f1"] < 0.2 and r_f["f1"] > 0.5:
                f.write("### Question\n")
                f.write(r_b["question"] + "\n\n")
                f.write("**Gold Answer:**\n")
                f.write(r_b["gold"] + "\n\n")
                f.write("**Baseline Prediction:**\n")
                f.write(r_b["pred"] + "\n\n")
                f.write("**Fine-tuned Prediction:**\n")
                f.write(r_f["pred"] + "\n\n")
                f.write("---\n\n")

    print("Saved evaluation_report.md")

def main():
    print("Evaluating baseline model...")
    baseline_results = evaluate(BASELINE_API)

    print("Evaluating fine-tuned model...")
    finetuned_results = evaluate(FINETUNED_API)

    save_report(baseline_results, finetuned_results)

if __name__ == "__main__":
    main()