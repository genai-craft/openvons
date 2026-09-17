"""Temperature / vector / matrix (Dirichlet-style) scaling fitted on validation logits by NLL (TASK-010)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _logits(probs, eps=1e-8) -> torch.Tensor:
    return torch.log(torch.clamp(torch.as_tensor(np.asarray(probs), dtype=torch.float64), eps, 1))


class TemperatureScaler:
    def __init__(self):
        self.T = 1.0

    def fit(self, probs, labels, iters: int = 300) -> "TemperatureScaler":
        z = _logits(probs)
        y = torch.as_tensor(np.asarray(labels), dtype=torch.long)
        mask = torch.isfinite(z) & (z > np.log(1e-8) + 1e-6)  # masked (padded) options stay -inf
        logT = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=iters, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            zz = torch.where(mask, z / torch.exp(logT), torch.full_like(z, -1e9))
            loss = F.cross_entropy(zz, y)
            loss.backward()
            return loss

        opt.step(closure)
        self.T = float(torch.exp(logT).item())
        return self

    def transform(self, probs) -> np.ndarray:
        z = _logits(probs)
        z = torch.where(z > np.log(1e-8) + 1e-6, z / self.T, torch.full_like(z, -1e9))
        return torch.softmax(z, 1).numpy()


class VectorScaler:
    """Per-class scale + bias. Only meaningful for fixed-class tasks (same options every sample)."""

    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, probs, labels, iters: int = 300) -> "VectorScaler":
        z = _logits(probs)
        y = torch.as_tensor(np.asarray(labels), dtype=torch.long)
        k = z.shape[1]
        w = torch.ones(k, dtype=torch.float64, requires_grad=True)
        b = torch.zeros(k, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], lr=0.1, max_iter=iters, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(z * w + b, y)
            loss.backward()
            return loss

        opt.step(closure)
        self.w, self.b = w.detach(), b.detach()
        return self

    def transform(self, probs) -> np.ndarray:
        return torch.softmax(_logits(probs) * self.w + self.b, 1).numpy()
