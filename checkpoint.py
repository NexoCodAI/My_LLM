import os
import torch
from config import CHECKPOINT_DIR

def save_checkpoint(model, optimizer, epoch, suffix=""):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    filename = f"ckpt_epoch_{epoch}"
    if suffix:
        filename += f"_{suffix}"
    filename += ".pt"

    path = os.path.join(CHECKPOINT_DIR, filename)

    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }, path)
    print(f"Saved checkpoint: {path}")
