"""Decision heads (TASK-006 / TASK-007).

  linear : Noul -> Linear(H, 2); Choice/Score -> Linear(H, 255) with unused options masked (spec 3.3)
  embed  : score_i = <Wq h_dec, Wk h_opt_i> / sqrt(d)       (spec 3.4, in-context choice embedding)
  mlp    : score_i = MLP(concat(h_dec, h_opt_i))            (spec 3.4 variant)
"""
from __future__ import annotations

import torch
import torch.nn as nn

HEADS = ["linear", "embed", "mlp"]
MAX_CHOICES = 255


class LinearHead(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.noul = nn.Linear(hidden, 2)
        self.choice = nn.Linear(hidden, MAX_CHOICES)

    def forward(self, pooled, opt_vecs, opt_mask, qtype: str):
        n = opt_mask.shape[1]
        logits = self.noul(pooled) if qtype == "noul" else self.choice(pooled)[:, :n]
        return logits.float().masked_fill(~opt_mask, float("-inf"))


class EmbedHead(nn.Module):
    def __init__(self, hidden: int, dim: int = 512):
        super().__init__()
        self.q = nn.Linear(hidden, dim)
        self.k = nn.Linear(hidden, dim)
        self.scale = dim ** -0.5

    def forward(self, pooled, opt_vecs, opt_mask, qtype: str):
        q = self.q(pooled)                      # (B, d)
        k = self.k(opt_vecs)                    # (B, N, d)
        logits = torch.einsum("bd,bnd->bn", q, k) * self.scale
        return logits.float().masked_fill(~opt_mask, float("-inf"))


class MLPHead(nn.Module):
    def __init__(self, hidden: int, dim: int = 512):
        super().__init__()
        self.proj_q = nn.Linear(hidden, dim)
        self.proj_k = nn.Linear(hidden, dim)
        self.mlp = nn.Sequential(nn.GELU(), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(self, pooled, opt_vecs, opt_mask, qtype: str):
        x = self.proj_q(pooled).unsqueeze(1) + self.proj_k(opt_vecs)   # (B, N, d)
        logits = self.mlp(x).squeeze(-1)
        return logits.float().masked_fill(~opt_mask, float("-inf"))


def build_head(kind: str, hidden: int, dim: int = 512) -> nn.Module:
    if kind == "linear":
        return LinearHead(hidden)
    if kind == "embed":
        return EmbedHead(hidden, dim)
    if kind == "mlp":
        return MLPHead(hidden, dim)
    raise ValueError(kind)
