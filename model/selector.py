import torch
import random


class Selector(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        # Use a two-layer MLP to encode the prefix
        self.embedding = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.trans = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size, config.prefix_hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(config.prefix_hidden_size, 4)
        )
        self.pool = torch.nn.AdaptiveMaxPool1d(1)

    def forward(self, prefix: torch.Tensor, routing=2):
        if routing == 0:
            return 0
        elif routing == 1:
            return random.randint(0, 3)
        else:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
            x = past_key_values.view(past_key_values.shape[0], 4, -1)
            x = self.pool(x)
            x = x.mean(dim=0).view(-1)
            return torch.argmax(x).cpu().numpy()
