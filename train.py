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

# Optional: gradient accumulation for larger effective batch size
GRADIENT_ACCUM_STEPS = 4

def train(data_path):
    set_seed()
    torch.backends.cudnn.benchmark = True  # optimize for consistent input sizes

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

    # Mixed Precision
    scaler = torch.amp.GradScaler()

    # Training loop
    for epoch in range(1, EPOCHS+1):
        model.train()
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}")
        total_loss = 0
        optimizer.zero_grad()

        for i, (x, y) in pbar:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(DEVICE.type == 'cuda')):
                logits = model(x)
                loss = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1)) / GRADIENT_ACCUM_STEPS

            scaler.scale(loss).backward()

            if (i + 1) % GRADIENT_ACCUM_STEPS == 0 or (i + 1) == len(dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * GRADIENT_ACCUM_STEPS  # unscale loss back
            pbar.set_postfix(loss=loss.item() * GRADIENT_ACCUM_STEPS)

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} completed. Avg Loss: {avg_loss:.4f}")
        save_checkpoint(model, optimizer, epoch)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to text data file")
    args = parser.parse_args()
    train(args.data)
