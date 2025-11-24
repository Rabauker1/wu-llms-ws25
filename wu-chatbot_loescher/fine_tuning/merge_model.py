import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Use the same base model as training
MODEL_NAME = os.environ.get("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
LORA_PATH = "./lora"
OUTPUT_PATH = "./wu_llm_finetuned"

print(f"Loading base model: {MODEL_NAME}")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float32,
)

print(f"Loading LoRA adapter from: {LORA_PATH}")
model = PeftModel.from_pretrained(base_model, LORA_PATH)

print("Merging LoRA weights into base model...")
merged_model = model.merge_and_unload()

print(f"Saving merged model to: {OUTPUT_PATH}")
merged_model.save_pretrained(OUTPUT_PATH)

print("Loading and saving tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.save_pretrained(OUTPUT_PATH)

print(f"Done! Merged model saved to {OUTPUT_PATH}")