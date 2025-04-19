import torch

# Model configuration
VOCAB_SIZE = None     # to be set after tokenizer initialization
D_MODEL = 512
N_LAYERS = 6
N_HEADS = 8
BLOCK_SIZE = 128      # context length

# Training hyperparameters
BATCH_SIZE = 256
LEARNING_RATE = 3e-4
EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Checkpointing
CHECKPOINT_DIR = "checkpoints"
