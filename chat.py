import os
import torch
from model import LLM
from tokenizer import SimpleTokenizer
from evaluate import sample
from checkpoint import load_checkpoint
from config import CHECKPOINT_DIR, D_MODEL, N_LAYERS, N_HEADS, BLOCK_SIZE, DEVICE

def chat():
    # 1) Build tokenizer once
    text_path = 'data/pride_and_prejudice.txt'
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()
    tokenizer = SimpleTokenizer([text])
    vocab_size = tokenizer.vocab_size

    # 2) Load latest checkpoint
    ckpts = sorted(os.listdir(CHECKPOINT_DIR), reverse=True)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {CHECKPOINT_DIR}")
    ckpt_path = os.path.join(CHECKPOINT_DIR, ckpts[0])
    print(f"⏳ Loading {ckpt_path}…")

    # 3) Instantiate & load
    model = LLM(vocab_size, D_MODEL, N_LAYERS, N_HEADS, BLOCK_SIZE).to(DEVICE)
    try:
        load_checkpoint(ckpt_path, model, optimizer=None)
    except Exception:
        # fallback if your checkpoint dict is nested
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt['model'])
    model.eval()
    print("✅ Model loaded. Type your prompt (or ’quit’ to exit).")

    # 4) Chat loop
    while True:
        prompt = input("\nYou: ")
        if prompt.strip().lower() in ('quit', 'exit'):
            print("Goodbye!")
            break

        # generate N tokens (you could also let sample() stream tokens back)
        out = sample(
            model, tokenizer,
            start_text=prompt,
            length=200,         # adjust how many tokens to generate
            device=DEVICE
        )
        # strip off the original prompt if sample returns full text
        response = out[len(prompt):] if out.startswith(prompt) else out
        print(f"\nModel: {response}")

if __name__ == "__main__":
    chat()

