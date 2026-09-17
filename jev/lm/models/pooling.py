from __future__ import annotations

import torch
import torch.nn as nn

POOLINGS = ["last", "decision", "mean", "attn"]


class AttentionPooling(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(hidden) * 0.02)
        self.key = nn.Linear(hidden, hidden, bias=False)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # h: (B, L, H), mask: (B, L) 1 = valid
        h = h.to(self.key.weight.dtype)
        k = self.key(h)
        scores = (k @ self.query.to(k.dtype)) / (h.shape[-1] ** 0.5)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        w = torch.softmax(scores.float(), dim=-1).to(h.dtype)
        return (w.unsqueeze(-1) * h).sum(1)


def pool(pooling: str, h: torch.Tensor, attention_mask: torch.Tensor, decision_pos: torch.Tensor,
         attn_pool: AttentionPooling | None = None) -> torch.Tensor:
    B = h.shape[0]
    ar = torch.arange(B, device=h.device)
    if pooling in ("last", "decision"):
        return h[ar, decision_pos]
    if pooling == "mean":
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1).clamp(min=1)
    if pooling == "attn":
        assert attn_pool is not None
        return attn_pool(h, attention_mask)
    raise ValueError(pooling)
