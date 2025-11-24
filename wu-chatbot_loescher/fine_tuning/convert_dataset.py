import jsonlines
from datasets import Dataset

items = []
with jsonlines.open("wu_eval_set.jsonl") as reader:
    for row in reader:
        items.append({
            "instruction": row["question"],
            "response": row["answer"],
        })

ds = Dataset.from_list(items)
ds.save_to_disk("wu_finetune_dataset")
print("Saved dataset to wu_finetune_dataset/")