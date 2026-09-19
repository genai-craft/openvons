"""Source A: suffix 一致 n-gram 候補 (§6.1)。

SuffixIndex は token 列に対し「末尾 n token が一致する過去の出現位置」を返す。
prompt / 直近出力 / 静的 template corpus / 外部 corpus のいずれにも同じ機構を使う。
prior = (一致 suffix 長 n, 出現回数, 直近性)。
"""
from __future__ import annotations

from collections import defaultdict

from .base import Candidate


class SuffixIndex:
    def __init__(self, max_n: int = 6, max_positions: int = 128):
        self.max_n = max_n
        self.max_positions = max_positions
        self.seq: list[int] = []
        # (n, ngram) -> ngram が終わる位置 p (= 続きが seq[p] から始まる)
        self.index: dict[tuple, list[int]] = defaultdict(list)

    def extend(self, tokens):
        seq, index, max_n = self.seq, self.index, self.max_n
        for t in tokens:
            seq.append(t)
            p = len(seq)
            for n in range(1, min(max_n, p) + 1):
                index[(n, tuple(seq[p - n : p]))].append(p)

    def occurrences(self, suffix, exclude_from: int | None = None) -> list[tuple[int, int]]:
        """suffix に一致する過去の出現を (n, p) で返す。長い n を優先し、同じ p は最長 n のみ。
        exclude_from: この位置以降で始まる続きは (自分自身なので) 除外。"""
        seen: dict[int, int] = {}
        for n in range(min(self.max_n, len(suffix)), 0, -1):
            pos = self.index.get((n, tuple(suffix[len(suffix) - n :])))
            if not pos:
                continue
            cnt = 0
            for p in reversed(pos):
                if exclude_from is not None and p >= exclude_from:
                    continue
                if p not in seen:
                    seen[p] = n
                    cnt += 1
                    if cnt >= self.max_positions:
                        break
            if len(seen) >= self.max_positions:
                break
        return [(n, p) for p, n in seen.items()]

    def blocks(self, occ, block_len: int, source: str, limit: int | None = None) -> list[Candidate]:
        """出現位置から block_len token の候補を作り、重複統合する。"""
        seq = self.seq
        found: dict[tuple, list] = {}
        for n, p in occ:
            if limit is not None:
                p_end = min(p + block_len, limit)
            else:
                p_end = p + block_len
            blk = tuple(seq[p:p_end])
            if not blk:
                continue
            e = found.get(blk)
            if e is None:
                found[blk] = [n, 1, p]
            else:
                if n > e[0]:
                    e[0], e[1] = n, 1
                elif n == e[0]:
                    e[1] += 1
                if p > e[2]:
                    e[2] = p
        return [Candidate(blk, source, (n, c, p), metadata={"p": p}) for blk, (n, c, p) in found.items()]


def rank(cands: list[Candidate], k: int) -> list[Candidate]:
    return sorted(cands, key=lambda c: c.prior_score, reverse=True)[:k]
