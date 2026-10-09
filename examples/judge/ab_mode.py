"""judge の判定方式 (質問方式 / 学習 head) を同一クリップで A/B し、項目ごとにどちらが良いかを出す。

  CUDA_VISIBLE_DEVICES=4 .venv/bin/python examples/judge/ab_mode.py 30      # 項目ごと あり/なし 各 30 本
  .venv/bin/python examples/judge/ab_mode.py 30 --all                       # ラベル不良 (suspect) も含める

サーバー (:8607) が起きている必要がある。結果を見て state/judge/mode_default.json の "head" を更新すると、
mode=auto (既定) がその項目だけ head を使うようになる (state に無ければ examples/judge/mode_default.json の実測値を使う)。
"""
import json, sys, time, collections
from pathlib import Path
import urllib.request, urllib.parse

ROOT = Path(__file__).resolve().parents[2]
rows = json.load(open(ROOT / "state/judge/clips/manifest.json"))
exclude_suspect = "--all" not in sys.argv
per = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 6

# 項目ごとに problem あり/なし を per 本ずつ
by = collections.defaultdict(lambda: {0: [], 1: []})
for r in rows:
    if exclude_suspect and r.get("suspect"):
        continue
    by[r["check"]][r["label"]].append(r)
sel = [r for k in by for lab in (0, 1) for r in by[k][lab][:per]]
print(f"評価 {len(sel)} 本 ({len(by)} 項目 × あり/なし 各 {per} 本" + ("、ラベル不良は除外)" if exclude_suspect else "、全部)"))

def judge(fn, check, scene, mode):
    data = urllib.parse.urlencode({"sample": fn, "checks": check, "scene": scene, "mode": mode}).encode()
    req = urllib.request.Request("http://127.0.0.1:8607/api/judge", data=data)
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)

res = {m: {"ok": 0, "n": 0, "t": 0.0, "tp": 0, "fp": 0, "fn": 0, "tn": 0,
           "per_check": collections.defaultdict(lambda: [0, 0])} for m in ("zeroshot", "head")}
for i, r in enumerate(sel):
    for m in ("zeroshot", "head"):
        t0 = time.time()
        try:
            j = judge(r["file"], r["check"], r["scene"], m)
        except Exception as e:
            print("  err", r["file"], m, e); continue
        res[m]["t"] += time.time() - t0
        s = next((x for x in j.get("summary", []) if x["key"] == r["check"]), None)
        pred = 1 if s and s.get("action") in ("alert", "confirm") else 0
        ok = int(pred == r["label"])
        res[m]["ok"] += ok; res[m]["n"] += 1
        res[m]["per_check"][r["check"]][0] += ok; res[m]["per_check"][r["check"]][1] += 1
        key = ("tp" if pred else "fn") if r["label"] else ("fp" if pred else "tn")
        res[m][key] += 1
    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{len(sel)}", flush=True)

print(f"\n{'方式':12s} {'正解率':>8s} {'再現率':>8s} {'特異度':>8s} {'1本あたり':>10s}")
for m, lab in [("zeroshot", "質問方式"), ("head", "学習 head")]:
    d = res[m]
    rec = d["tp"] / max(d["tp"] + d["fn"], 1); spec = d["tn"] / max(d["tn"] + d["fp"], 1)
    print(f"{lab:12s} {d['ok']/max(d['n'],1):8.3f} {rec:8.3f} {spec:8.3f} {d['t']/max(d['n'],1)*1000:9.0f}ms")
print("\n項目別の正解率 (質問方式 -> 学習 head)")
for k in sorted(res["zeroshot"]["per_check"], key=lambda k: res["head"]["per_check"][k][0] - res["zeroshot"]["per_check"][k][0]):
    a = res["zeroshot"]["per_check"][k]; b = res["head"]["per_check"][k]
    d = b[0] / max(b[1], 1) - a[0] / max(a[1], 1)
    mark = "◎" if d > 0.15 else ("○" if d > 0 else ("×" if d < -0.15 else "－"))
    print(f"  {k:24s} {a[0]/max(a[1],1):.2f} -> {b[0]/max(b[1],1):.2f}  {mark}")
