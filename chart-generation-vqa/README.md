# Chart Generation & Visual Question Answering Pipeline

An end-to-end system that generates charts from text descriptions and answers questions about them using two fine-tuned models.

## 📋 Overview

This project demonstrates a complete pipeline combining:

1. **Chart Generation** – Converting natural language descriptions into executable chart code
2. **Visual Question Answering** – Answering questions about generated or existing charts

### Models Used

- **StarCoder2-1B** (fine-tuned with LoRA) – Generates Python (matplotlib) chart code from text
- **Qwen2-VL-2B-Instruct** (fine-tuned with QLoRA) – Performs visual question answering on chart images


---

## 📂 Project Structure
```
├── Stage1_Chart_Generation.ipynb # Fine-tuning StarCoder2-1B
├── Stage2_Visual_QA.ipynb # Fine-tuning Qwen2-VL-2B
├── Stage3_Full_Pipeline.ipynb # End-to-end demo
├── README.md
```

---

## 🧩 Stage Summaries

### **Stage 1 — Chart Generation**
- Fine-tunes **StarCoder2-1B** using LoRA + 8-bit quantization  
- Trained on a subset of **ChartGen-200K**  
- Outputs valid matplotlib code based on natural-language prompts  

### **Stage 2 — Visual Question Answering**
- Fine-tunes **Qwen2-VL-2B-Instruct** with QLoRA  
- Trained on a subset of **ChartQA**  
- Answers questions about charts

### **Stage 3 — Full Pipeline**
- Takes a text description  
- Generates chart code  
- Executes the code  
- Answers user questions about the produced chart  

---

## Authors
Sofia Papushina\
Vladyslava Stepanovska


