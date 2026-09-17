"""Losses on masked logits (B, N) vs target distributions (B, N)  (Phase 4)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

LOSSES = ["ce", "kl", "brier", "focal", "ce+brier", "kl+brier"]


def _logp(logits):
    return F.log_softmax(logits, dim=-1)


def soft_ce(logits, t, mask):
    return -(t * _logp(logits).masked_fill(~mask, 0)).sum(-1).mean()


def kl(logits, t, mask, eps=1e-8):
    logp = _logp(logits).masked_fill(~mask, 0)
    logt = torch.log(t.clamp(min=eps)).masked_fill(~mask, 0)
    return (t * (logt - logp)).sum(-1).mean()


def brier(logits, t, mask):
    p = torch.softmax(logits, dim=-1).masked_fill(~mask, 0)
    return ((p - t) ** 2).sum(-1).mean()


def focal(logits, t, mask, gamma: float = 2.0):
    logp = _logp(logits).masked_fill(~mask, 0)
    p = logp.exp()
    return -(t * (1 - p) ** gamma * logp).sum(-1).mean()


def compute_loss(name: str, logits, t, mask):
    if name == "ce":
        return soft_ce(logits, t, mask)
    if name == "kl":
        return kl(logits, t, mask)
    if name == "brier":
        return brier(logits, t, mask)
    if name == "focal":
        return focal(logits, t, mask)
    if name == "ce+brier":
        return soft_ce(logits, t, mask) + brier(logits, t, mask)
    if name == "kl+brier":
        return kl(logits, t, mask) + brier(logits, t, mask)
    raise ValueError(name)
