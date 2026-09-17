"""Isotonic regression on the top-label confidence (multi-class: rescale top prob, distribute the rest)
and Platt scaling for binary (noul) questions."""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class IsotonicTopLabel:
    def fit(self, probs, labels):
        p = np.asarray(probs)
        conf = p.max(1)
        correct = (p.argmax(1) == np.asarray(labels)).astype(float)
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(conf, correct)
        return self

    def transform(self, probs):
        p = np.asarray(probs, dtype=np.float64).copy()
        conf = p.max(1)
        new_conf = np.clip(self.iso.predict(conf), 1e-6, 1 - 1e-6)
        top = p.argmax(1)
        rest = 1 - conf
        for i in range(len(p)):
            if rest[i] > 1e-12:
                p[i] *= (1 - new_conf[i]) / rest[i]
            else:
                p[i] = (1 - new_conf[i]) / max(1, p.shape[1] - 1)
            p[i, top[i]] = new_conf[i]
        return p


class PlattBinary:
    """Platt scaling for 2-option questions (p(true))."""

    def fit(self, probs, labels):
        p = np.asarray(probs)[:, 0]
        z = np.log(np.clip(p, 1e-8, 1) / np.clip(1 - p, 1e-8, 1))
        y = (np.asarray(labels) == 0).astype(int)
        self.lr = LogisticRegression(C=1e6).fit(z[:, None], y)
        return self

    def transform(self, probs):
        p = np.asarray(probs)[:, 0]
        z = np.log(np.clip(p, 1e-8, 1) / np.clip(1 - p, 1e-8, 1))
        pt = self.lr.predict_proba(z[:, None])[:, 1]
        return np.stack([pt, 1 - pt], 1)
