class SimpleTokenizer:
    """
    A basic character-level tokenizer.
    """
    def __init__(self, texts):
        # Build vocabulary from provided text(s)
        chars = sorted(set("".join(texts)))
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for ch,i in self.stoi.items() }
        self.vocab_size = len(self.stoi)

    def encode(self, text):
        return [self.stoi[c] for c in text]

    def decode(self, tokens):
        return ''.join(self.itos[t] for t in tokens)