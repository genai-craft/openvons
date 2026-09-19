"""v2 データ: Tool Call (既知/未知 schema split, 学習用 3000 + test 500) と repo-level Python (site-packages の実 repo)。

出力: /data/openvons/jevpick/prompts_v2.jsonl
  {sample_id, domain, split (train/test), messages, tools, meta}
  toolcall.meta = {tool_names, schema_known: bool}
  python.meta   = {repo, path, cut_line}
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

from datasets import load_dataset

OUT = Path("/data/openvons/jevpick/prompts_v2.jsonl")
SITE = Path(sys.prefix) / "lib/python3.12/site-packages"
REPOS = ["transformers", "huggingface_hub", "fastapi", "pydantic", "rich", "datasets", "anyio", "httpx",
         "jinja2", "click", "attr", "fsspec", "aiohttp", "starlette", "uvicorn", "peft", "PIL", "yaml",
         "tokenizers", "safetensors"]
rng = random.Random(0)


def parse_glaive(r):
    m = re.search(r"following functions\. Use them if required -\s*(.*)$", r["system"], re.S)
    if not m:
        return None
    raw, tools, pos, dec = m.group(1).strip(), [], 0, json.JSONDecoder()
    try:
        while pos < len(raw):
            while pos < len(raw) and raw[pos] in " \n\r\t":
                pos += 1
            if pos >= len(raw):
                break
            obj, end = dec.raw_decode(raw, pos)
            tools.append({"type": "function", "function": obj})
            pos = end
    except json.JSONDecodeError:
        return None
    if not tools:
        return None
    um = re.search(r"USER:\s*(.*?)\s*ASSISTANT:\s*(.*?)(?:<\|endoftext\|>|USER:|$)", r["chat"], re.S)
    if not um or not um.group(2).strip().startswith("<functioncall>"):
        return None
    return um.group(1).strip(), tools


def toolcall(n_train=3000, n_test=500):
    ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
    rows = []
    for i in range(len(ds)):
        p = parse_glaive(ds[i])
        if p:
            rows.append((i, *p))
        if len(rows) >= 12000:
            break
    # tool 名で split: 出現頻度の低い tool 名を「未知 schema」側へ
    from collections import Counter
    cnt = Counter(t["function"]["name"] for _, _, tools in rows for t in tools)
    names = sorted(cnt, key=lambda k: cnt[k])
    unknown = set()
    for nm in names:  # 低頻度側から集めて未知集合を作る (全体の約 1/6 のサンプルが該当するように)
        unknown.add(nm)
        if sum(cnt[u] for u in unknown) > 0.18 * sum(cnt.values()):
            break
    train, test_known, test_unknown = [], [], []
    for i, user, tools in rows:
        tn = [t["function"]["name"] for t in tools]
        rec = {"sample_id": f"tc_{i}", "domain": "toolcall", "messages": [{"role": "user", "content": user}],
               "tools": tools, "meta": {"tool_names": tn}}
        if any(t in unknown for t in tn):
            rec["meta"]["schema_known"] = False
            test_unknown.append(rec)
        elif len(train) < n_train:
            rec["split"] = "train"
            rec["meta"]["schema_known"] = True
            train.append(rec)
        else:
            rec["meta"]["schema_known"] = True
            test_known.append(rec)
    rng.shuffle(test_known); rng.shuffle(test_unknown)
    test = test_known[: n_test // 2] + test_unknown[: n_test // 2]
    for r in test:
        r["split"] = "test"
    print(f"toolcall train {len(train)} test {len(test)} (known {len(test_known[:n_test//2])}, unknown {len(test_unknown[:n_test//2])}), unknown tools {len(unknown)}")
    return train + test


def python_repo(n_train=1500, n_test=500):
    files = []
    for repo in REPOS:
        fs = [f for f in (SITE / repo).rglob("*.py") if f.stat().st_size > 1500 and "test" not in f.name]
        rng.shuffle(fs)
        files += [(repo, f) for f in fs[:200]]
    rng.shuffle(files)
    # repo 単位で train/test を分ける (§10.3)
    test_repos = {"rich", "httpx", "starlette", "aiohttp", "PIL", "click"}
    out = []
    n_tr = n_te = 0
    for repo, f in files:
        is_test = repo in test_repos
        if (is_test and n_te >= n_test) or (not is_test and n_tr >= n_train):
            continue
        src = f.read_text(errors="ignore")
        lines = src.split("\n")
        if len(lines) < 30:
            continue
        # 関数/メソッド定義の直後で切る → 本体を書かせる
        defs = [i for i, l in enumerate(lines[10:-5], 10) if re.match(r"\s*def \w+\(", l)]
        if not defs:
            continue
        cut = rng.choice(defs) + 1
        while cut < len(lines) and (lines[cut].strip().startswith(("\"\"\"", "'''")) or not lines[cut].strip()):
            cut += 1
        prefix = "\n".join(lines[:cut])
        if len(prefix) > 12000:
            prefix = prefix[-12000:]
        rel = str(f.relative_to(SITE))
        msg = (f"Continue the following Python file `{rel}`. Output only the next 15-30 lines of code that follow, "
               f"in a single ```python code block, without repeating the given code.\n\n```python\n{prefix}\n```")
        out.append({"sample_id": f"py_{repo}_{len(out)}", "domain": "python", "split": "test" if is_test else "train",
                    "messages": [{"role": "user", "content": msg}], "tools": None,
                    "meta": {"repo": repo, "path": rel, "cut_line": cut}})
        if is_test:
            n_te += 1
        else:
            n_tr += 1
    print(f"python train {n_tr} test {n_te}")
    return out


def main():
    rows = toolcall() + python_repo()
    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(len(rows), "->", OUT)


if __name__ == "__main__":
    main()
