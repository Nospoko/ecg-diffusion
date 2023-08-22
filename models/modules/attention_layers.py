import torch
import einops
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, channels: int, heads: int = 4):
        """
        Multi Head Attention Layer

        Args:
            channels (int): number of channels
            heads (int): number of attention heads
        """

        super().__init__()

        self.heads = heads
        self.head_dim = channels // heads
        self.embedding_size = channels

        assert self.head_dim * heads == channels, "Channels needs to be dividable by heads"

        self.values_proj = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.keys_proj = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.queries_proj = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.out = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        # applying linear projections
        q = self.queries_proj(x)
        k = self.keys_proj(x)
        v = self.values_proj(x)

        # rearranging q, k, v from [batch_size, channels, seq_len] -> [batch_size, seq_len, heads, head_dim]
        q = einops.rearrange(q, "n (h d) l -> n l h d", h=self.heads)
        k = einops.rearrange(k, "n (h d) l -> n l h d", h=self.heads)
        v = einops.rearrange(v, "n (h d) l -> n l h d", h=self.heads)

        # shapes
        # query: (N, query_len, heads, head_dim)
        # keys: (N, key_len, heads, head_dim)
        # output: (N, heads, query_len, key_len)
        qk = torch.einsum("n q h d, n k h d -> n h q k", [q, k])

        # applying softmax over key dimension to calculate attention scores
        attn = torch.softmax(qk * (self.embedding_size**-0.5), dim=3)

        # shapes
        # attn: (N, heads, query_len, key_len)
        # values: (N, values_len, heads, head_dim)
        # output: (N, query_len, heads, head_dim)
        out = torch.einsum("n h q l, n l h d -> n q h d", [attn, v])

        # concatenation of heads
        out = einops.rearrange(out, "n l h d -> n (h d) l")

        return self.out(out)
