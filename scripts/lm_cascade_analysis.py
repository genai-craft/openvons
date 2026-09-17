"""§11 fallback strategy の実測。

Decision Model が答え、max probability がしきい値未満の質問だけ生成 LLM に回したときの
最終 accuracy / LLM 呼び出し率 / 平均レイテンシを測る。結果は experiments/exp012_cascade_<task>.json。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openvons.core.metrics import all_metrics  # noqa: E402
from openvons.core.temperature import TemperatureScaler  # noqa: E402
from openvons.core.formats import read_jsonl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="massive_scenario_en")
    ap.add_argument("--ckpt", required=True, help="decision model checkpoint name under /data/decision_model/checkpoints")
    ap.add_argument("--llm_exp", required=True, help="experiments/<name>_probs.npy of the fallback LLM")
    ap.add_argument("--dm_latency_ms", type=float, default=16.4)
    ap.add_argument("--llm_latency_ms", type=float, default=73.6)
    a = ap.parse_args()

    test = read_jsonl(str(ROOT / "data/store/processed" / a.task / "test.jsonl"))
    valid = read_jsonl(str(ROOT / "data/store/processed" / a.task / "valid.jsonl"))
    y = np.array([s.label for s in test])
    yv = np.array([s.label for s in valid])
    ck = Path("/data/decision_model/checkpoints") / a.ckpt
    P = np.load(ck / "probs_test.npy")
    ts = TemperatureScaler().fit(np.load(ck / "probs_valid.npy"), yv)
    P = ts.transform(P)
    L = np.load(ROOT / "experiments" / f"{a.llm_exp}_probs.npy")
    assert len(P) == len(L) == len(y)

    dm_pred, dm_conf = P.argmax(1), P.max(1)
    llm_pred = L.argmax(1)
    rows = []
    for th in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01]:
        low = dm_conf < th
        pred = np.where(low, llm_pred, dm_pred)
        rows.append({"threshold": th, "llm_call_rate": float(low.mean()), "accuracy": float((pred == y).mean()),
                     "mean_latency_ms": round(a.dm_latency_ms + float(low.mean()) * a.llm_latency_ms, 1)})
    auto = []
    for th in [0.8, 0.9, 0.95]:
        for name, probs, pred in [("decision_model", P, dm_pred), ("llm", L, llm_pred)]:
            hi = probs.max(1) >= th
            auto.append({"system": name, "threshold": th, "coverage": float(hi.mean()),
                         "accuracy_on_covered": float((pred[hi] == y[hi]).mean()) if hi.any() else None})
    res = {"experiment": f"exp012_cascade_{a.task}", "dataset": a.task, "ckpt": a.ckpt, "fallback_llm": a.llm_exp,
           "dm_alone": all_metrics(P, y), "llm_alone": all_metrics(L, y), "temperature": ts.T,
           "cascade": rows, "auto_execute": auto}
    json.dump(res, open(ROOT / "experiments" / f"exp012_cascade_{a.task}.json", "w"), indent=2)
    print(f"{'threshold':>10} {'llm_calls':>10} {'accuracy':>9} {'mean_ms':>8}")
    for r in rows:
        print(f"{r['threshold']:10.2f} {r['llm_call_rate']:9.1%} {r['accuracy']:9.4f} {r['mean_latency_ms']:8.1f}")
    print()
    for r in auto:
        print(f"auto-execute {r['system']:14s} conf>={r['threshold']}: coverage {r['coverage']:.1%} accuracy {r['accuracy_on_covered']:.4f}")


if __name__ == "__main__":
    main()
