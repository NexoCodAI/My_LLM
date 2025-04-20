import torch

# Model configuration
VOCAB_SIZE = None     # to be set after tokenizer initialization
D_MODEL = 256
N_LAYERS = 3
N_HEADS = 4
BLOCK_SIZE = 512      # context length

# Training hyperparameters
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
EPOCHS = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Checkpointing
CHECKPOINT_DIR = "checkpoints"
