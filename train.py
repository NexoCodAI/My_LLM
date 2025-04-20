import os 
import torch
import torch.nn as nn
from tqdm import tqdm
from itertools import islice

from config import *           # EPOCHS, BLOCK_SIZE, BATCH_SIZE, DEVICE, 
                                # VALID_LOSS_THRESHOLD, PATIENCE, EVAL_INTERVAL
from tokenizer import SimpleTokenizer
from model import LLM
from dataset import get_dataloader
from utils import set_seed, ensure_dir
from checkpoint import save_checkpoint
from evaluate import evaluate   # pull in the eval() we defined earlier

GRADIENT_ACCUM_STEPS = 4
def train(data_path):
    set_seed()
    torch.backends.cudnn.benchmark = True

    # ─── 1) READ & TOKENIZE ────────────────────────────────────────────────
    text      = open(data_path, 'r', encoding='utf-8').read()
    tokenizer = SimpleTokenizer([text])
    global VOCAB_SIZE
    VOCAB_SIZE = tokenizer.vocab_size
    tokens    = tokenizer.encode(text)
    # ─── 2) SPLIT ─────────────────────────────────────────────────────────
    split_idx    = int(0.9 * len(tokens))     # 90% train / 10% val
    train_tokens = tokens[:split_idx]
    val_tokens   = tokens[split_idx:]
    val_tokens = val_tokens[:2000]  # or even try 2000 for super fast evals


    train_loader = get_dataloader(train_tokens, BLOCK_SIZE+1, BATCH_SIZE)
    # we’ll recreate val_loader inside eval() as needed

    # ─── 3) MODEL + OPT + LOSS + SCALER ───────────────────────────────────
    model     = LLM(VOCAB_SIZE, D_MODEL, N_LAYERS, N_HEADS, BLOCK_SIZE).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    scaler    = torch.amp.GradScaler()

    best_val_loss = float('inf')
    patience_cnt  = 0
    stop_training = False
    global_step   = 0
    for epoch in range(1, EPOCHS+1):
        if stop_training:
            break

        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Epoch {epoch}")
        for i, (x, y) in pbar:
            global_step += 1
            x, y = x.to(DEVICE), y.to(DEVICE)

            # ‣ forward + accumulate
            with torch.cuda.amp.autocast(enabled=(DEVICE.type=='cuda')):
                logits   = model(x)
                raw_loss = criterion(logits.view(-1, VOCAB_SIZE),
                                     y.view(-1))
                loss     = raw_loss / GRADIENT_ACCUM_STEPS

            scaler.scale(loss).backward()
            if (i + 1) % GRADIENT_ACCUM_STEPS == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # ‣ logging
            batch_loss = raw_loss.item()
            pbar.set_postfix(batch_loss=f"{batch_loss:.4f}")

            # ─── EVALUATE PERIODICALLY ───────────────────────────────────────
            if global_step % EVAL_INTERVAL == 0:
                print(f"⏳ Starting evaluation at step {global_step}...")
                val_loss, val_ppl = evaluate(
                    model,
                    val_tokens,
                    BATCH_SIZE,
                    BLOCK_SIZE,
                    DEVICE
                )
                tqdm.write(f"[Step {global_step}]  Val Loss={val_loss:.4f}, PPL={val_ppl:.2f}")

                # ─── TRACK BEST & PATIENCE ────────────────────────────────
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_cnt  = 0
                    save_checkpoint(model, optimizer, epoch, suffix="best_val")
                    tqdm.write("  🎉 New best! Checkpoint saved.")
                else:
                    patience_cnt += 1
                    tqdm.write(f"  ⚠  No improvement (patience {patience_cnt}/{PATIENCE})")

                # ─── STOP IF BELOW YOUR THRESHOLD OR OUT OF PATIENCE ────
                if val_loss < VALID_LOSS_THRESHOLD:
                    tqdm.write(f"🔥 Reached target val loss {val_loss:.4f} < {VALID_LOSS_THRESHOLD}")
                    stop_training = True
                elif patience_cnt >= PATIENCE:
                    tqdm.write(f"⏳ Patience exhausted ({PATIENCE} evals).")
                    stop_training = True

            if stop_training:
                break

        # end of inner loop
        epoch_avg = None
        tqdm.write(f"Epoch {epoch} completed. Best val loss so far: {best_val_loss:.4f}")
        save_checkpoint(model, optimizer, epoch)

    if stop_training:
        print(f"🚀 Stopping training at step {global_step}, epoch {epoch}.")
    else:
        print(f"🏁 Finished {EPOCHS} epochs; best val loss = {best_val_loss:.4f}")
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to text data file")
    args = parser.parse_args()

    print("Starting training...")
    train(args.data)
