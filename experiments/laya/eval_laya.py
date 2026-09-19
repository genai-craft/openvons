"""convaiinnovations/laya (ModernBERT-large の判断モデル、生成しない) を openvons.lm と同じテストセットで評価する。

  /data/openvons/choice_spec/venvs/laya/bin/python experiments/laya/eval_laya.py --tasks massive_scenario_en,tweet_offensive,glaive_tools,massive_scenario_ja
ゼロショット (学習なし) の数字。openvons.lm の 4B 凍結 + head は各タスクで学習しているので、比較は「学習なしでどこまで出るか」として読む。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

DATA = Path("/data/decision_model/data/processed")


def ece(conf, correct, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1); e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


def to_questions(row):
    if row["type"] == "noul":
        return {"q": {"type": "noul", "instructions": row["question"]}}
    return {"q": {"type": "choice", "instructions": row["question"], "criteria": {c["id"]: c["description"] for c in row["choices"]}}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="massive_scenario_en,tweet_offensive,glaive_tools,massive_intent_en,massive_scenario_ja")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", default="experiments/laya/results.json")
    a = ap.parse_args()
    import laya
    agents = {}
    res = {}
    for task in a.tasks.split(","):
        f = DATA / task / "test.jsonl"
        if not f.exists():
            print("skip", task); continue
        rows = [json.loads(l) for l in open(f)][: a.n]
        sub = "multilingual" if task.endswith("_ja") else None
        key = sub or "en"
        if key not in agents:
            t0 = time.time()
            agents[key] = laya.load("convaiinnovations/laya", subfolder=sub) if sub else laya.load("convaiinnovations/laya")
            print("loaded", key, "%.0fs" % (time.time() - t0), flush=True)
        agent = agents[key]
        correct, conf, lat = [], [], []
        for r in rows:
            qs = to_questions(r)
            t0 = time.perf_counter()
            try:
                out = agent.predict(r["state"], qs)
            except Exception as e:  # noqa: BLE001
                print("ERR", task, str(e)[:120]); continue
            lat.append((time.perf_counter() - t0) * 1e3)
            ans = out["answers"]["q"] if "answers" in out else out["q"]
            if r["type"] == "noul":
                p = ans.get("noul", ans.get("probability", ans.get("p_true", ans.get("prob"))))
                if isinstance(p, dict):
                    p = p.get("true", p.get("yes", p.get("probability")))
                if p is None and isinstance(ans.get("probabilities"), dict):
                    p = ans["probabilities"].get("true", ans["probabilities"].get("yes"))
                pred = 1 if (p or 0) >= 0.5 else 0
                # label: 0/1 (target true = choices[0] "true")
                gold = 1 if r["label"] == 0 else 0  # choices[0]=true → label 0 means "true"
                correct.append(pred == gold); conf.append(max(p or 0, 1 - (p or 0)))
            else:
                ch = ans.get("choice", ans.get("answer", ans.get("selected")))
                probs = ans.get("probabilities") or ans.get("probs") or {}
                gold = r["choices"][r["label"]]["id"] if isinstance(r["label"], int) else r["label"]
                correct.append(ch == gold); conf.append(float(probs.get(ch, ans.get("confidence", 0.0)) or 0.0))
        c = np.array(correct, float); cf = np.array(conf, float)
        res[task] = {"n": int(len(c)), "accuracy": float(c.mean()) if len(c) else None, "ece": ece(cf, c) if len(c) else None, "latency_ms_median": float(np.median(lat)) if lat else None,
                     "checkpoint": key, "example_answer_keys": list((out["answers"]["q"] if "answers" in out else out["q"]).keys()) if rows else []}
        print(task, res[task], flush=True)
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
    print("->", a.out)


if __name__ == "__main__":
    main()
