"""Source B: repository retrieval (§6.2) の最小版。

同一 repo (site-packages 配下の package) の全 .py を target tokenizer で token 化し、1 本の SuffixIndex に入れる。
ファイル境界 (file_ranges) を持ち、補完対象ファイル自身の出現は検索時に除外する (leakage 防止)。
文字検索ではなく token 列の suffix 一致 (§6.2)。identifier slot 正規化検索は未実装。
"""
from __future__ import annotations

from pathlib import Path

from .ngram import SuffixIndex

SEP = 151643


class RepoIndex(SuffixIndex):
    def __init__(self, max_n=6, max_positions=128):
        super().__init__(max_n, max_positions)
        self.file_ranges: dict[str, tuple[int, int]] = {}

    def add_file(self, rel: str, tokens):
        s = len(self.seq)
        self.extend(tokens)
        self.extend([SEP])
        self.file_ranges[rel] = (s, len(self.seq))

    def occurrences_excluding(self, suffix, rel: str | None):
        occ = self.occurrences(suffix)
        if rel and rel in self.file_ranges:
            s, e = self.file_ranges[rel]
            occ = [(n, p) for n, p in occ if not (s < p <= e)]
        return occ


def build_repo_index(tok, root: str, max_files: int = 150, max_chars: int = 30000, max_n: int = 6) -> RepoIndex:
    idx = RepoIndex(max_n=max_n, max_positions=128)
    rootp = Path(root)
    files = sorted(rootp.rglob("*.py"), key=lambda f: -f.stat().st_size)[:max_files]
    for f in files:
        try:
            src = f.read_text(errors="ignore")[:max_chars]
        except OSError:
            continue
        idx.add_file(str(f.relative_to(rootp.parent)), tok.encode(src, add_special_tokens=False))
    return idx
