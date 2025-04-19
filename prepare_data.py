# save as prepare_data.py
import os
from datasets import load_dataset

def prepare_text(output_path="data/combined.txt", lines=500000):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as out:
        # 1. Local book
        with open("data/pride_and_prejudice.txt", "r", encoding="utf-8") as f:
            out.write(f.read() + "\n")

        # 2. WikiText streaming
        ds = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True)
        for i, ex in enumerate(ds):
            out.write(ex['text'].replace('\n', ' ') + "\n")
            if i >= lines:
                break
    print(f"Data saved to {output_path}")
