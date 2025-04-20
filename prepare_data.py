# prepare_data.py
import os
from datasets import load_dataset

def prepare_text(
    output_dir="data",
    wiki_lines=500_000,
    val_frac=0.1
):
    """
    Streams in your base text + wiki, then splits into train/val files.
    
    Args:
      output_dir: where to write train.txt and val.txt
      wiki_lines: how many wiki lines to pull
      val_frac:   fraction of TOTAL lines to reserve for validation
    """
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.txt")
    val_path   = os.path.join(output_dir, "val.txt")

    # Total lines = 1 (pride) + wiki_lines
    total_lines = 1 + wiki_lines
    split_idx   = int((1 - val_frac) * total_lines)

    with open(train_path, "w", encoding="utf-8") as f_train, \
         open(val_path,   "w", encoding="utf-8") as f_val:

        # helper to pick the right file by line-index
        def write_line(idx, text):
            target = f_train if idx < split_idx else f_val
            target.write(text.rstrip("\n") + "\n")

        # ---- 1) Pride & Prejudice counts as line 0 ----
        with open("data/pride_and_prejudice.txt", "r", encoding="utf-8") as f:
            raw = f.read()
        write_line(0, raw)

        # ---- 2) WikiText streaming lines 1..wiki_lines ----
        ds = load_dataset("wikitext", "wikitext-103-v1",
                          split="train", streaming=True)
        for i, ex in enumerate(ds, start=1):
            write_line(i, ex["text"].replace("\n", " "))
            if i >= wiki_lines:
                break

    print(f"→ Wrote {split_idx} lines to {train_path}")
    print(f"→ Wrote {total_lines - split_idx} lines to {val_path}")

if __name__ == "__main__":
    # tweak these if you like, or add argparse
    prepare_text(
        output_dir="data",
        wiki_lines=500_000,
        val_frac=0.1
    )
