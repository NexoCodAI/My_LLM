import os
import torch
import torch.nn as nn
from tqdm import tqdm
from config import *
from tokenizer import SimpleTokenizer
from model import LLM
from dataset import get_dataloader
from utils import set_seed, ensure_dir
from checkpoint import save_checkpoint


def train(data_path):
    set_seed()
    # Load data
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    tokenizer = SimpleTokenizer([text])
    global VOCAB_SIZE
    VOCAB_SIZE = tokenizer.vocab_size

    tokens = tokenizer.encode(text)
    dataloader = get_dataloader(tokens, BLOCK_SIZE+1, BATCH_SIZE)

    # Model
    model = LLM(VOCAB_SIZE, D_MODEL, N_LAYERS, N_HEADS, BLOCK_SIZE).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(1, EPOCHS+1):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        total_loss = 0
        for x, y in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} completed. Avg Loss: {avg_loss:.4f}")
        save_checkpoint(model, optimizer, epoch)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to text data file")
    args = parser.parse_args()
    train(args.data)