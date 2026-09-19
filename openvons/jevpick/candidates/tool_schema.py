"""Source D: tool schema 駆動候補 (§6.4) + Source E の tool 版 macro/copy (§6.5)。

prompt に含まれる tools (OpenAI 形式) から、Qwen3 の tool_call 出力形式に沿った断片を展開し
SuffixIndex に入れる。展開は sample ごとに 1 回。値 slot は user 発話中の数値・引用・大文字語から copy する。
"""
from __future__ import annotations

import json
import re

from .ngram import SuffixIndex

SEP = 151643


def _values_from_text(text: str) -> list[str]:
    vals = re.findall(r"\d+(?:\.\d+)?", text)
    vals += re.findall(r"[\"'“]([^\"'”]{1,60})[\"'”]", text)
    vals += re.findall(r"\b[A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)*\b", text)  # 固有名詞らしきもの
    seen, out = set(), []
    for v in vals:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out[:24]


def expand_tool_templates_xml(tools: list[dict], user_text: str = "") -> list[str]:
    """Qwen3.5 系の形式: <tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n...</function>\n</tool_call>"""
    frags = ["<tool_call>\n<function=", "\n</parameter>\n</function>\n</tool_call>", "\n</parameter>\n<parameter=", "</function>\n</tool_call>"]
    values = _values_from_text(user_text)
    for t in tools:
        fn = t.get("function", t)
        name = fn.get("name", "")
        params = (fn.get("parameters") or {}).get("properties") or {}
        req = (fn.get("parameters") or {}).get("required") or list(params)
        head = f"<tool_call>\n<function={name}>\n"
        frags.append(head)
        keys = list(params)
        for i, k in enumerate(keys):
            frags.append(f"<parameter={k}>\n")
            frags.append(f"\n</parameter>\n<parameter={k}>\n")
            if i == 0:
                frags.append(head + f"<parameter={k}>\n")
            for ev in (params[k] or {}).get("enum", [])[:8]:
                frags.append(f"<parameter={k}>\n{ev}\n</parameter>\n")
            for v in values[:12]:
                frags.append(f"<parameter={k}>\n{v}\n</parameter>\n")
        if req:
            frags.append(head + "".join(f"<parameter={k}>\n" for k in req[:1]))
            frags.append("".join(f"<parameter={k}>\n\n</parameter>\n" for k in req) + "</function>\n</tool_call>")
    return frags


def expand_tool_templates(tools: list[dict], user_text: str = "", fmt: str = "json") -> list[str]:
    if fmt == "xml":
        return expand_tool_templates_xml(tools, user_text)
    if fmt == "both":
        return expand_tool_templates_xml(tools, user_text) + expand_tool_templates(tools, user_text, "json")
    frags = ["<tool_call>\n{\"name\": \"", "\"}}\n</tool_call>", "}}\n</tool_call>", "\"}\n</tool_call>"]
    values = _values_from_text(user_text)
    for t in tools:
        fn = t.get("function", t)
        name = fn.get("name", "")
        params = (fn.get("parameters") or {}).get("properties") or {}
        req = (fn.get("parameters") or {}).get("required") or list(params)
        head = f"<tool_call>\n{{\"name\": \"{name}\", \"arguments\": {{"
        frags.append(head)
        keys = list(params)
        for i, k in enumerate(keys):
            typ = (params[k] or {}).get("type", "string")
            frags.append(f"\"{k}\": " + ("\"" if typ == "string" else ""))
            frags.append(f", \"{k}\": " + ("\"" if typ == "string" else ""))
            if i == 0:
                frags.append(head + f"\"{k}\": " + ("\"" if typ == "string" else ""))
            if i + 1 < len(keys):
                nk = keys[i + 1]
                ntyp = (params[nk] or {}).get("type", "string")
                # 現在 key の値を閉じて次 key へ
                close = "\", " if typ == "string" else ", "
                frags.append(close + f"\"{nk}\": " + ("\"" if ntyp == "string" else ""))
            # enum は値まで展開
            for ev in (params[k] or {}).get("enum", [])[:8]:
                frags.append(f"\"{k}\": \"{ev}\"")
            # copy slot: user 発話の値を埋める
            for v in values[:12]:
                if typ == "string":
                    frags.append(f"\"{k}\": \"{v}\"")
                elif re.fullmatch(r"\d+(?:\.\d+)?", v):
                    frags.append(f"\"{k}\": {v}")
        # 必須引数を順に並べた骨格 (値は空)
        if req:
            skel = head + ", ".join(f"\"{k}\": " + ("\"\"" if (params.get(k) or {}).get("type", "string") == "string" else "0") for k in req) + "}}\n</tool_call>"
            frags.append(skel)
    return frags


def detect_format(prompt_text: str) -> str:
    """chat template の system 部分から tool_call の形式を判定する。"""
    return "xml" if "<function=" in prompt_text or "<parameter=" in prompt_text else "json"


def build_schema_index(tok, tools: list[dict], user_text: str = "", max_n: int = 6, fmt: str = "json") -> SuffixIndex:
    idx = SuffixIndex(max_n=max_n, max_positions=256)
    for s in expand_tool_templates(tools, user_text, fmt):
        idx.extend(tok.encode(s, add_special_tokens=False))
        idx.extend([SEP])
    return idx


def tools_from_prompt_text(text: str) -> list[dict]:
    """Qwen3 chat template の <tools>...</tools> から tool 定義を取り出す。"""
    m = re.search(r"<tools>\n(.*?)\n</tools>", text, re.S)
    if not m:
        return []
    out = []
    for line in m.group(1).split("\n"):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def user_text_from_prompt_text(text: str) -> str:
    m = re.findall(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", text, re.S)
    return m[-1] if m else ""
