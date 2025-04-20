import os 
import torch
import torch.nn as nn
from tqdm import tqdm
from config import *           # must define EPOCHS, BLOCK_SIZE, BATCH_SIZE, etc.
from tokenizer import SimpleTokenizer
from model import LLM
from dataset import get_dataloader
from utils import set_seed, ensure_dir
from checkpoint import save_checkpoint

# Optional: gradient accumulation for larger effective batch size
GRADIENT_ACCUM_STEPS = 4
LOSS_THRESHOLD      = 1.0   # <-- stop training as soon as raw batch-loss < this

def train(data_path):
    set_seed()
    torch.backends.cudnn.benchmark = True

    # Load & tokenize
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    tokenizer     = SimpleTokenizer([text])
    global VOCAB_SIZE
    VOCAB_SIZE    = tokenizer.vocab_size
    tokens        = tokenizer.encode(text)
    dataloader    = get_dataloader(tokens, BLOCK_SIZE+1, BATCH_SIZE)

    # Model / optimizer / loss / scaler
    model     = LLM(VOCAB_SIZE, D_MODEL, N_LAYERS, N_HEADS, BLOCK_SIZE).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    scaler    = torch.amp.GradScaler()

    stop_training = False

    for epoch in range(1, EPOCHS+1):
        if stop_training:
            break

        model.train()
        pbar       = tqdm(enumerate(dataloader), total=len(dataloader),
                           desc=f"Epoch {epoch}")
        total_loss = 0.0

        for i, (x, y) in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)

            # forward
            with torch.cuda.amp.autocast(enabled=(DEVICE.type=='cuda')):
                logits   = model(x)
                raw_loss = criterion(logits.view(-1, VOCAB_SIZE),
                                     y.view(-1))
                loss     = raw_loss / GRADIENT_ACCUM_STEPS

            # backward & step
            scaler.scale(loss).backward()
            if (i + 1) % GRADIENT_ACCUM_STEPS == 0 or (i + 1) == len(dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # logging
            batch_loss = raw_loss.item()
            total_loss += batch_loss
            avg_loss   = total_loss / (i + 1)
            pbar.set_postfix(batch_loss=f"{batch_loss:.4f}", avg_loss=f"{avg_loss:.4f}")

            # early stop check
            if batch_loss < LOSS_THRESHOLD:
                print(f"\n✅ Early stopping: batch {i+1} loss={batch_loss:.4f} < {LOSS_THRESHOLD}")
                stop_training = True
                break

        # end of inner loop
        print(f"Epoch {epoch} done. Avg loss so far: {avg_loss:.4f}")
        save_checkpoint(model, optimizer, epoch)

    if stop_training:
        print(f"🚀 Training halted at epoch {epoch}, batch-loss under threshold.")
    else:
        print(f"🏁 Finished all {EPOCHS} epochs; final avg-loss={avg_loss:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to text data file")
    args = parser.parse_args()
    train(args.data)
