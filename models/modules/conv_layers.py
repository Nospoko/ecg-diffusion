from functools import partial
from inspect import isfunction

import torch
import einops
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv1d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    # No More Strided Convolutions or Pooling
    return nn.Sequential(
        Rearrange("b c (l p) -> b (c p) l", p=2),
        nn.Conv1d(dim * 2, default(dim_out, dim), 1),
    )


class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class WeightStandardizedConv1d(nn.Conv1d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = einops.reduce(weight, "o ... -> o 1 1", "mean")
        var = einops.reduce(weight, "o ... -> o 1 1", partial(torch.var, unbiased=False))

        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv1d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Block(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, kernel_size: int = 3, groups: int = 8):
        super().__init__()
        # calculate padding for given kernel_size
        padding = (kernel_size - 1) // 2

        self.conv = WeightStandardizedConv1d(in_dim, out_dim, kernel_size=kernel_size, padding=padding)
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=out_dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, scale_shift: tuple[torch.Tensor, torch.Tensor] = None) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)

        return x


class ResnetBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, *, kernel_size: int = 3, time_emb_dim: int = None, groups: int = 8):
        super().__init__()

        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_dim * 2)) if exists(time_emb_dim) else None

        self.block_1 = Block(in_dim, out_dim, kernel_size=kernel_size, groups=groups)
        self.block_2 = Block(out_dim, out_dim, kernel_size=kernel_size, groups=groups)
        self.res_conv = nn.Conv1d(in_dim, out_dim, kernel_size=1) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor = None):
        if exists(self.time_proj) and exists(t_emb):
            t_emb = self.time_proj(t_emb)
            t_emb = einops.rearrange(t_emb, "b c -> b c 1")
            scale_shift = t_emb.chunk(2, dim=1)

        h = self.block_1(x, scale_shift=scale_shift)
        h = self.block_2(h)

        return h + self.res_conv(x)
