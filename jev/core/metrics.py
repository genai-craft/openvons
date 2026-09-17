"""ECE / Brier / NLL / accuracy / macro-F1 / reliability diagram (Phase 5)."""
from __future__ import annotations

import numpy as np


def _np(x):
    return np.asarray(x, dtype=np.float64)


def accuracy(probs, labels) -> float:
    return float((np.argmax(_np(probs), 1) == _np(labels)).mean())


def macro_f1(probs, labels, n_classes: int | None = None) -> float:
    pred = np.argmax(_np(probs), 1)
    labels = _np(labels).astype(int)
    classes = range(n_classes or int(max(pred.max(), labels.max()) + 1))
    f1s = []
    for c in classes:
        tp = ((pred == c) & (labels == c)).sum()
        fp = ((pred == c) & (labels != c)).sum()
        fn = ((pred != c) & (labels == c)).sum()
        if tp + fp + fn == 0:
            continue
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def nll(probs, labels, eps=1e-12) -> float:
    p = _np(probs)
    return float(-np.log(np.clip(p[np.arange(len(p)), _np(labels).astype(int)], eps, 1)).mean())


def brier(probs, labels) -> float:
    """Multi-class Brier: mean over samples of sum_i (p_i - y_i)^2."""
    p = _np(probs)
    y = np.zeros_like(p)
    y[np.arange(len(p)), _np(labels).astype(int)] = 1
    return float(((p - y) ** 2).sum(1).mean())


def ece(probs, labels, n_bins: int = 15) -> float:
    """Top-label expected calibration error with equal-width bins."""
    p = _np(probs)
    conf = p.max(1)
    correct = (p.argmax(1) == _np(labels)).astype(np.float64)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


def reliability_table(probs, labels, n_bins: int = 10) -> list[dict]:
    p = _np(probs)
    conf = p.max(1)
    correct = (p.argmax(1) == _np(labels)).astype(np.float64)
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "count": int(m.sum()),
                     "mean_confidence": float(conf[m].mean()) if m.any() else None,
                     "accuracy": float(correct[m].mean()) if m.any() else None})
    return rows


def kl_to_teacher(probs, teacher_probs, eps=1e-8) -> float:
    p, t = np.clip(_np(probs), eps, 1), np.clip(_np(teacher_probs), eps, 1)
    return float((t * (np.log(t) - np.log(p))).sum(1).mean())


def all_metrics(probs, labels, n_classes: int | None = None, teacher_probs=None) -> dict:
    out = {"accuracy": accuracy(probs, labels), "macro_f1": macro_f1(probs, labels, n_classes),
           "ece": ece(probs, labels), "brier": brier(probs, labels), "nll": nll(probs, labels),
           "n": int(len(labels))}
    if teacher_probs is not None:
        out["kl_to_teacher"] = kl_to_teacher(probs, teacher_probs)
    return out


def latency_summary(ms: list[float]) -> dict:
    a = _np(ms)
    if len(a) == 0:
        return {}
    return {"latency_p50_ms": float(np.percentile(a, 50)), "latency_p95_ms": float(np.percentile(a, 95)),
            "latency_mean_ms": float(a.mean())}
