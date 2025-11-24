****README — LLM Tennis Match Report Project****

This project explores the generation of tennis match reports using Large Language Models (LLMs).
It compares a baseline T5-small model with a fine-tuned variant, and uses GPT-4.1-mini as an external reference.
The goal is to evaluate how fine-tuning improves the ability to generate accurate, structured match summaries.

**Notebooks**

	• *FINAL_Tennis_Project.ipynb* - contains all code relevant for the presentation
	• *t5_training.ipynb* - contains the full training pipeline for the T5 model

**Data**

Training and test data are located in: */data*

Models

	•	Baseline model: T5-small
  
	•	Fine-tuned model: generated during training (the model itself not included in this repository due to file size limits)
  
	•	External reference model: GPT-4.1-mini


The notebook evaluates:

	•	ROUGE metrics
  
	•	Cosine similarity (SentenceTransformer)
  
	•	LLM-as-a-Judge
  
	•	Example outputs for all models


**Results**

The fine-tuned T5 model showed noticeably improved performance compared to the baseline.
While the generated reports were still far from fully coherent, some match facts were already accurate, indicating meaningful learning despite the small dataset.

The clear limitation of the project is the very limited sample size, which heavily restricts model quality and generalization.
Although the fine-tuned T5 model outperformed the baseline—and even achieved the best scores in certain metrics—GPT-4.1-mini produced by far the most coherent and accurate match reports.
This was also reflected in the LLM-as-a-Judge evaluations.