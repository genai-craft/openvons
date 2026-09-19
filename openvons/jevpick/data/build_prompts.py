"""§22 最小実験用の prompt セットを作る。Python / Tool Call / Chat 各 500 件。

出力: {DATA}/prompts.jsonl
  {sample_id, domain, messages, tools}
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
from openvons.jevpick.paths import DATA  # noqa: E402

import json
import re
import sys
from pathlib import Path

from datasets import load_dataset

OUT = Path(f"{DATA}/prompts.jsonl")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 500


def python_prompts():
    ds = load_dataset("google-research-datasets/mbpp", "full")
    rows = list(ds["test"]) + list(ds["validation"]) + list(ds["train"])
    out = []
    for r in rows[:N]:
        tests = "\n".join(r["test_list"])
        msg = (
            f"{r['text']}\n\nYour code should pass these tests:\n```python\n{tests}\n```\n"
            "Return only the Python code in a single code block."
        )
        out.append({"sample_id": f"py_{r['task_id']}", "domain": "python",
                    "messages": [{"role": "user", "content": msg}], "tools": None})
    return out


def toolcall_prompts():
    ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train", streaming=True)
    out = []
    for i, r in enumerate(ds):
        sysm = r["system"]
        m = re.search(r"following functions\. Use them if required -\s*(.*)$", sysm, re.S)
        if not m:
            continue
        # 複数関数が連結されていることがある → JSON を順に切り出す
        raw = m.group(1).strip()
        tools = []
        dec = json.JSONDecoder()
        pos = 0
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
            continue
        if not tools:
            continue
        chat = r["chat"]
        um = re.search(r"USER:\s*(.*?)\s*ASSISTANT:\s*(.*?)(?:<\|endoftext\|>|USER:|$)", chat, re.S)
        if not um:
            continue
        user, asst = um.group(1).strip(), um.group(2).strip()
        if not asst.startswith("<functioncall>"):
            continue  # 最初の応答が tool call になるものだけ
        out.append({"sample_id": f"tc_{i}", "domain": "toolcall",
                    "messages": [{"role": "user", "content": user}], "tools": tools})
        if len(out) >= N:
            break
    return out


def chat_prompts():
    out = []
    half = N // 2
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft", streaming=True)
    for i, r in enumerate(ds):
        first = r["messages"][0]
        if first["role"] != "user" or len(first["content"]) > 2000:
            continue
        out.append({"sample_id": f"chat_en_{i}", "domain": "chat",
                    "messages": [{"role": "user", "content": first["content"]}], "tools": None})
        if len(out) >= half:
            break
    ds = load_dataset("kunishou/databricks-dolly-15k-ja", split="train")
    k = 0
    for i, r in enumerate(ds):
        msg = r["instruction"] + (("\n\n" + r["input"]) if r.get("input") else "")
        if len(msg) > 2000:
            continue
        out.append({"sample_id": f"chat_ja_{i}", "domain": "chat",
                    "messages": [{"role": "user", "content": msg}], "tools": None})
        k += 1
        if k >= N - half:
            break
    return out


def main():
    rows = python_prompts() + toolcall_prompts() + chat_prompts()
    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(Counter(r["domain"] for r in rows), "->", OUT)


if __name__ == "__main__":
    main()
