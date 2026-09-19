"""長文脈 (目標 ~16k token) の test prompt を作る。

python : 同 repo の他ファイルを文脈として前置し、そのあとに補完対象ファイルの prefix を置く (repo-level 補完)。
toolcall: 自分の tools + 他サンプルの tools を混ぜた大きな tool カタログ (~100 tools) を与える。
出力: {DATA}/prompts_long.jsonl (split=test)
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
from openvons.jevpick.paths import DATA  # noqa: E402

import json
import random
import sys
from pathlib import Path

from transformers import AutoTokenizer

SITE = Path(sys.prefix) / "lib/python3.12/site-packages"
OUT = Path(f"{DATA}/prompts_long.jsonl")
TARGET_TOK = int(sys.argv[1]) if len(sys.argv) > 1 else 16000
N = int(sys.argv[2]) if len(sys.argv) > 2 else 40
rng = random.Random(1)


def main():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    rows = [json.loads(l) for l in open(f"{DATA}/prompts_v2.jsonl")]
    py_test = [r for r in rows if r["domain"] == "python" and r["split"] == "test"]
    tc_test = [r for r in rows if r["domain"] == "toolcall" and r["split"] == "test"]
    tc_all = [r for r in rows if r["domain"] == "toolcall"]
    out = []
    # python
    for r in py_test[:N]:
        meta = r["meta"]
        own = SITE / meta["path"]
        files = [f for f in sorted((SITE / meta["repo"]).rglob("*.py")) if f != own and f.stat().st_size > 1000]
        rng.shuffle(files)
        ctx_parts, n_tok = [], 0
        body = r["messages"][0]["content"]
        budget = TARGET_TOK - len(tok.encode(body))
        for f in files:
            src = f.read_text(errors="ignore")[:20000]
            part = f"# File: {f.relative_to(SITE)}\n{src}\n\n"
            n = len(tok.encode(part))
            if n_tok + n > budget:
                continue
            ctx_parts.append(part); n_tok += n
            if n_tok > budget * 0.95:
                break
        msg = "Here are other files from the same repository for context:\n\n```python\n" + "".join(ctx_parts) + "```\n\n" + body
        out.append({**r, "sample_id": r["sample_id"] + "_long", "messages": [{"role": "user", "content": msg}],
                    "meta": {**meta, "ctx_files": len(ctx_parts)}})
    # toolcall
    for r in tc_test[:N]:
        tools = list(r["tools"])
        names = {t["function"]["name"] for t in tools}
        n_tok = len(tok.encode(tok.apply_chat_template(r["messages"], tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=False), add_special_tokens=False))
        pool = tc_all[:]
        rng.shuffle(pool)
        for o in pool:
            for t in o["tools"]:
                if t["function"]["name"] in names:
                    continue
                n = len(tok.encode(json.dumps(t)))
                if n_tok + n > TARGET_TOK:
                    break
                tools.append(t); names.add(t["function"]["name"]); n_tok += n
            if n_tok >= TARGET_TOK * 0.97:
                break
        rng.shuffle(tools)
        out.append({**r, "sample_id": r["sample_id"] + "_long", "tools": tools, "meta": {**r["meta"], "n_tools": len(tools)}})
    with OUT.open("w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    import numpy as np
    for d in ("python", "toolcall"):
        ls = []
        for r in out:
            if r["domain"] != d:
                continue
            kw = {"tools": r["tools"]} if r["tools"] else {}
            ls.append(len(tok.encode(tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False, **kw), add_special_tokens=False)))
        print(d, "n", len(ls), "prompt tokens mean %.0f min %d max %d" % (np.mean(ls), min(ls), max(ls)))
    print("->", OUT)


if __name__ == "__main__":
    main()
