"""Source E: macro + copy (§6.5) の Python 版。

prompt (と直近出力) に現れる identifier を slot に埋めた定型断片を展開し、SuffixIndex に入れる。
展開は identifier 集合が変わったとき (既定: 32 token ごと) にやり直す。
"""
from __future__ import annotations

import re

from .ngram import SuffixIndex

SEP = 151643
IDENT = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")
KEYWORDS = {"def", "class", "return", "import", "from", "None", "True", "False", "self", "for", "while", "with",
            "not", "and", "the", "python", "code", "lines", "given", "file", "following", "Continue", "Output",
            "only", "next", "that", "follow", "single", "block", "without", "repeating", "elif", "else", "try",
            "except", "raise", "pass", "yield", "lambda", "assert", "async", "await", "del", "global", "nonlocal"}

TEMPLATES = [
    "if {X} is None:\n            return", "if {X} is None:\n        return", "if {X} is None:\n            raise",
    "if {X} is None:\n        raise", "if not {X}:\n            return", "if not {X}:\n        return",
    "if {X} is not None:\n", "        return {X}\n", "            return {X}\n", "        self.{X} = {X}\n",
    "        self.{X} = {Y}\n", "        return self.{X}\n", "        return self._{X}\n", "self.{X}(", "self._{X}(",
    "        {X} = self.{Y}(", "        {X} = {Y}(", "        for {X} in {Y}:\n", "        for {X} in self.{Y}:\n",
    "            {X}.append(", "        {X}.append(", "        {X} = []\n", "        {X} = {}\n",
    "        return {X}(", "        raise {X}(", "isinstance({X}, ", "len({X})", "        if {X}:\n",
    "        if {X} in {Y}:\n", "        if {X} not in {Y}:\n", "{X}={X}", ", {X}={X}", "{X}: {Y}", "        {X} = {Y}\n",
    "        del self.{X}\n", "        return {X}.{Y}(", "        {X}.{Y}(", "self.{X}.{Y}(", "        with {X}(",
    "    def {X}(self", "    def _{X}(self", "    def {X}(self, {Y}", "    @property\n    def {X}(self):\n        return self._{X}\n",
]


def identifiers(text: str, limit: int = 40) -> list[str]:
    seen, out = set(), []
    for m in reversed(IDENT.findall(text)):  # 直近を優先
        w = m
        if w in KEYWORDS or w in seen:
            continue
        seen.add(w)
        out.append(w)
        if len(out) >= limit:
            break
    return out


def expand_macros(idents: list[str], max_pairs: int = 12) -> list[str]:
    out = []
    top = idents[:max_pairs]
    for t in TEMPLATES:
        if "{Y}" in t:
            for x in top:
                for y in top:
                    if x != y:
                        out.append(t.replace("{X}", x).replace("{Y}", y))
        else:
            for x in idents:
                out.append(t.replace("{X}", x))
    return out


def build_macro_index(tok, text: str, max_n: int = 6) -> SuffixIndex:
    idx = SuffixIndex(max_n=max_n, max_positions=256)
    for s in expand_macros(identifiers(text)):
        idx.extend(tok.encode(s, add_special_tokens=False))
        idx.extend([SEP])
    return idx
