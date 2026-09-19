"""llama.cpp (GGUF Q4_K_M) 上での比較: 通常 decode / n-gram (prompt lookup) / MTP (同梱 NextN) / DFlash2 (GGUF draft)。
llama-server を構成ごとに起動し、/completion に chat template 済み prompt を投げて timings.predicted_per_second (decode 専用) を集める。
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
from openvons.jevpick.paths import DATA  # noqa: E402

import argparse
import json
import subprocess
import time

import numpy as np
import requests
from transformers import AutoTokenizer


def wait_ready(port, proc, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            if requests.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200:
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(3)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default=f"{DATA}/tools/llama.cpp/build/bin/llama-server")
    ap.add_argument("--model", required=True, help="main GGUF path")
    ap.add_argument("--draft", default="", help="DFlash2 GGUF path")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--prompts", default=f"{DATA}/prompts_v2.jsonl")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--port", type=int, default=8477)
    ap.add_argument("--gpu", default="7")
    ap.add_argument("--configs", default="none,ngram,mtp,dflash")
    ap.add_argument("--out", default="experiments/jevpick/phase4_runtime/llamacpp_Qwen3.8-27B-Q4_K_M.json")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    rows = [json.loads(l) for l in open(args.prompts)]
    test = [r for r in rows if r["domain"] == "toolcall" and r.get("split") == "test"][: args.n + 1]
    prompts = [tok.apply_chat_template(r["messages"], tools=r["tools"], tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in test]
    common = [args.server, "-m", args.model, "--port", str(args.port), "-ngl", "999", "-c", "8192", "-np", "1", "--temp", "0", "-fa", "on", "--no-warmup"]
    configs = {
        "none": [],
        "ngram": ["--spec-type", "ngram-simple", "--spec-draft-n-max", "8"],
        "mtp": ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"],
        "mtp7": ["--spec-type", "draft-mtp", "--spec-draft-n-max", "7"],
        "dflash": ["-md", args.draft, "--spec-type", "draft-dflash", "--spec-draft-n-max", "7"],
    }
    res = {}
    for name in args.configs.split(","):
        if name == "dflash" and not args.draft:
            continue
        cmd = common + configs[name]
        log = open(f"{DATA}/llamacpp_{name}.log", "w")
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env={"CUDA_VISIBLE_DEVICES": args.gpu, "PATH": "/usr/bin:/bin"})
        if not wait_ready(args.port, proc):
            res[name] = {"error": "server failed to start; see log"}
            print(name, res[name], flush=True)
            proc.kill(); continue
        recs = []
        for i, p in enumerate(prompts):
            r = requests.post(f"http://127.0.0.1:{args.port}/completion", json={"prompt": p, "n_predict": args.max_tokens, "temperature": 0, "cache_prompt": False}, timeout=600).json()
            t = r.get("timings", {})
            if i > 0:
                recs.append({"n": t.get("predicted_n"), "ms": t.get("predicted_ms"), "tps": t.get("predicted_per_second"), "pp_ms": t.get("prompt_ms"),
                             "draft_n": t.get("draft_n"), "draft_acc": t.get("draft_n_accepted"), "text": r.get("content", "")[:120]})
        proc.terminate(); proc.wait(timeout=60)
        tot_n = sum(x["n"] for x in recs); tot_ms = sum(x["ms"] for x in recs)
        out = {"n": len(recs), "decode_tok_s": 1000 * tot_n / tot_ms, "mean_tps": float(np.mean([x["tps"] for x in recs])),
               "draft_n": sum(x["draft_n"] or 0 for x in recs), "draft_accepted": sum(x["draft_acc"] or 0 for x in recs), "texts": [x["text"] for x in recs[:3]]}
        res[name] = out
        print(name, {k: v for k, v in out.items() if k != "texts"}, flush=True)
        json.dump(res, open(args.out, "w"), indent=1)
    print("->", args.out)


if __name__ == "__main__":
    main()
