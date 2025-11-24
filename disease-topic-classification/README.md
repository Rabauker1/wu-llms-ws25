# Disease Topic Classification with Large Language Models

## Project overview

This project uses large language models to classify free-text patient notes into a fixed set of **10 disease topics**  
(e.g. sepsis, pneumonia, periodontitis, status epilepticus).  
The task is **multi-label**: each note can receive zero, one, or several of these topics.

The goals are:
- design and compare different **prompt strategies** (v0–v3),
- evaluate several **LLM families** (GPT-4.1-mini, GPT-4.1, LLaMA 3-8B),
- measure performance with multi-label metrics,
- analyse typical error patterns.

## Data

Input files:

- `data-note-with-diseases-label.xlsx`  
  - sheet `few-shot examples`: 5 labeled notes used as in-context examples  
  - sheet `tests-with-the-ground-truth`: 250 test notes with ground-truth disease labels
  
> Note: The dataset originates from the **Open-Patients** collection on Hugging Face.  
> The Excel file itself is not included in this repository and must be added locally.

Derived files created during the work:

- `data-note-with-diseases-label - tests-with-the-ground-truth.csv`  
  CSV export of the test sheet, used for evaluation.
- `tests_with_predictions_all_models.csv`  
  Model predictions for all 12 model/prompt combinations (A0–A3, B0–B3, C0–C3).
- `model_evaluation_summary.csv`  
  Summary table with micro/macro precision, recall, F1 and subset accuracy for all systems.

## Models and prompts

Model families:

- **A: GPT-4.1-mini** (OpenAI)
- **B: GPT-4.1** (OpenAI)
- **C: LLaMA 3-8B Instruct** (via Hugging Face Inference API)

Prompt variants:

- **v0** – zero-shot (task + label list only)  
- **v1** – few-shot: + 5 labeled examples  
- **v2** – v1 + short medical definitions for each label  
- **v3** – v2 + explicit “Reasoning: … / Labels: …” format

This yields **12 systems**: A0–A3, B0–B3, C0–C3.

## Main notebook

- `Disease_Topic_Classification_12322607.ipynb`

What it does:

1. Loads `data-note-with-diseases-label.xlsx` and extracts:
   - the 10 disease topics from the few-shot sheet,
   - the 250 test notes with ground-truth labels.
2. Defines the four prompt variants (v0–v3).
3. Calls three model families (GPT-4.1-mini, GPT-4.1, LLaMA 3-8B) via API.
4. Runs all 12 model/prompt combinations on the test notes.
5. Saves predictions to `tests_with_predictions_all_models.csv`.
6. Merges predictions with ground truth and computes:
   - micro/macro precision, recall, F1,
   - subset accuracy (exact-match),
   - metrics on all notes and on notes with ≥1 target label.
7. Writes the aggregated metrics to `model_evaluation_summary.csv`.

## Requirements

Python packages:

- `openai`
- `pandas`
- `numpy`
- `scikit-learn`
- `huggingface_hub`

Example installation:

```bash
pip install openai pandas numpy scikit-learn huggingface_hub
