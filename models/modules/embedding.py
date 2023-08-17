import math

import torch
import torch.nn as nn


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, time_emb_dim: int):
        super().__init__()

        self.time_emb_dim = time_emb_dim

    def forward(self, t: torch.Tensor):
        half_dim = self.time_emb_dim // 2

        embedding = math.log(10000) / (half_dim - 1)
        embedding = torch.exp(torch.arange(half_dim, device=t.device) * -embedding)
        embedding = t[:, None] * embedding[None, :]
        embedding = torch.cat([embedding.sin(), embedding.cos()], dim=-1)

        return embedding
