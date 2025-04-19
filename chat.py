import os
import torch
from model import LLM
from tokenizer import SimpleTokenizer
from evaluate import sample
from checkpoint import load_checkpoint
from config import *

def chat():
    # Load dialogue data for tokenizer context (same text used in training)
    text_path = 'data/your_textfile.txt'  # adjust path if needed
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()
    tokenizer = SimpleTokenizer([text])
    global VOCAB_SIZE
    VOCAB_SIZE = tokenizer.vocab_size

    # Load latest checkpoint
    checkpoint_files = sorted(os.listdir(CHECKPOINT_DIR), reverse=True)
    if not checkpoint_files:
        raise FileNotFoundError("No checkpoints found in " + CHECKPOINT_DIR)
    latest_ckpt = os.path.join(CHECKPOINT_DIR, checkpoint_files[0])

    # Initialize model and load weights
    model = LLM(VOCAB_SIZE, D_MODEL, N_LAYERS, N_HEADS, BLOCK_SIZE).to(DEVICE)
    load_checkpoint(latest_ckpt, model)

    print("=== Chat Mode ===\nType 'exit' to quit.\n")
    conversation = ""
    while True:
        user_input = input("User: ")
        if user_input.strip().lower() == "exit":
            print("Goodbye!")
            break
        # Append to history
        conversation += f"User: {user_input}\nBot: "
        # Generate a reply
        response = sample(model, tokenizer, start_text=conversation, length=100, device=DEVICE)
        # Extract only the new bot text
        bot_reply = response[len(conversation):].split("\nUser:")[0].strip()
        print(f"Bot: {bot_reply}\n")
        # Update history
        conversation += bot_reply + "\n"

if __name__ == "__main__":
    chat()
