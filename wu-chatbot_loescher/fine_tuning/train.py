from dotenv import load_dotenv
import os
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from huggingface_hub import login

load_dotenv()  # Load environment variables from .env file

# Hugging Face login (uses HF_TOKEN env var or prompts for token)
if "HF_TOKEN" in os.environ:
    login(token=os.environ["HF_TOKEN"])
    print("Logged in to Hugging Face using HF_TOKEN")
else:
    print("HF_TOKEN not found.")
    raise ValueError("HF_TOKEN environment variable is required for authentication.")

# Model selection: smaller model that doesn't require authentication
MODEL_NAME = os.environ.get("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
# Bigger models would improve performance but do not fit into my CPU memory

# Using cpu
device = "cpu"
print(f"Using device: {device}")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

# Set pad token if not present (required for training)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model (may take a while)...")
# Load model with low_cpu_mem_usage; do not attempt 4bit/8bit on CPU/MPS
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    low_cpu_mem_usage=True,
)

# Move model to device (MPS/CPU). For large models you may need a machine with enough RAM.
model.to(device)

# Apply LoRA using PEFT
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# Load dataset
ds = load_from_disk("wu_finetune_dataset")

def format_example(example):
    return f"""You are a WU Vienna expert. Answer briefly and factually.

### Question:
{example['instruction']}

### Answer:
{example['response']}
"""

ds = ds.map(lambda x: {"text": format_example(x)})

# Training arguments
args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    logging_steps=10,
    learning_rate=2e-4,
    bf16=False,
    fp16=(device == "mps"),
    optim="adamw_torch",
    warmup_steps=10,
)

# Training with TRL's SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=ds,
    args=args,
    formatting_func=lambda x: x["text"],
)

trainer.train()

# Save LoRA adapter
model.save_pretrained("./lora")
tokenizer.save_pretrained("./lora")

print("Done! LoRA adapter saved to ./lora")