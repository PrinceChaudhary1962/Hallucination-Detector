"""
Download VitaminC test split into data/vitaminc_test.jsonl

Run from the project root:
    python scripts/download_vitaminc.py
"""

import json
import os

from datasets import load_dataset

dataset = load_dataset("tals/vitaminc", split="test")

os.makedirs("data", exist_ok=True)

# Save as jsonl
with open("data/vitaminc_test.jsonl", "w") as f:
    for item in dataset:
        f.write(json.dumps(item) + "\n")

print(f"Saved {len(dataset)} samples")
print("Sample:", dataset[0])