import os
import torch
from config import CHECKPOINT_DIR

def save_checkpoint(model, optimizer, epoch):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f"ckpt_epoch_{epoch}.pt")
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }, path)
    print(f"Saved checkpoint: {path}")


def load_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['model'])
    if optimizer and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt.get('epoch', None)