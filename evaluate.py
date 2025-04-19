import torch
from torch.nn import functional as F

def sample(model, tokenizer, start_text, length, device):
    model.eval()
    tokens = tokenizer.encode(start_text)
    input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    generated = tokens.copy()
    for _ in range(length):
        input_trim = input_ids if input_ids.size(1) <= model.block_size else input_ids[:, -model.block_size:]
        logits = model(input_trim)  # (1, T, vocab)
        next_token_logits = logits[0, -1, :]
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_token)
        input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)
    return tokenizer.decode(generated)