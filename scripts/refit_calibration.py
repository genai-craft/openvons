"""eval_synthetic.py が保存した採点サンプルから校正 (T, β0, β1) を再推定し、方式を比較する.

    .venv/bin/python scripts/refit_calibration.py experiments/eval_synth_v2_samples_shuto.json [...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from openvons.core.none_calibration import Calibration, fit  # noqa: E402


def load(path: str):
    D = json.load(open(path))
    if "lens" not in D[0]:
        from transformers import WhisperTokenizer
        tok = WhisperTokenizer.from_pretrained("sbintuitions/kana-whisper")
        for d in D:
            d["lens"] = [len(tok.encode(c, add_special_tokens=False)) for c in d["cands"]]
    return D


def report(D, cal: Calibration, name: str):
    P = [cal.probs(np.array(d["scores"]), d["free_score"], np.array(d["lens"]), d.get("n_free")) for d in D]
    corr = np.array([p.argmax() == d["y"] for p, d in zip(P, D)]); conf = np.array([p.max() for p in P])
    bins = np.linspace(0, 1, 11); e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs(corr[m].mean() - conf[m].mean())
    by = {}
    for p, d in zip(P, D):
        by.setdefault(d["cond"], []).append(p.argmax() == d["y"])
    gate = conf >= 0.85
    print(f"{name:22} T={cal.temperature:.2f} β0={cal.none_bias:.1f} β1={cal.len_bonus:.2f} | acc {corr.mean():.3f} ECE {e:.3f} | gate.85 cov {gate.mean():.2f} prec {corr[gate].mean() if gate.any() else 0:.3f} | "
          + " ".join(f"{k} {np.mean(v):.2f}" for k, v in sorted(by.items())))
    return corr, P


if __name__ == "__main__":
    for path in sys.argv[1:]:
        D = load(path)
        print(f"== {path}: {len(D)} samples")
        S = [(np.array(d["scores"]), d["free_score"], d["y"], np.array(d["lens"]), d.get("n_free")) for d in D]
        report(D, Calibration(1.0, 3.0, 0.0, 0.0), "旧既定 (β=3)")
        report(D, Calibration(1.5, 30.0, 0.0, 0.0), "定数 β=30")
        c0 = fit(S, Calibration(1.5, 30.0, 0.0, 0.0), fit_len_bonus=False); report(D, c0, "fit: β0 のみ")
        c1 = fit(S, Calibration(), fit_len_bonus=True); corr, P = report(D, c1, "fit: β0 β1 γ")
        report(D, Calibration(), "既定 (2.5, 4, 1.4, 1.0)")
        # 短い候補の安全性: 「ハイ」(2 トークン、尤度 −5) が長い発話 (13 トークン、自由認識 −0.1) に勝たないか
        for cal, nm in [(c1, "fit"), (Calibration(), "既定")]:
            p = cal.probs(np.array([-5.0]), -0.1, np.array([2.0]), 13)
            print(f"   [{nm}] 電話の「はい」模擬: p(はい)={p[0]:.3f} p(該当なし)={p[1]:.3f}")
        miss = [(d["cond"], d["text"], d["free"], d["cands"][int(np.argmax(np.array(d['scores'])))]) for d, ok in zip(D, corr) if not ok]
        print("   misses:", miss[:6])
