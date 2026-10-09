"""komimi (genai-craft/komimi、小さな Conformer-CTC、Apache-2.0) を kana 入力にする backend.

KanaASR (kana-whisper) と同じ口 (encode / transcribe / tokenize / score_tokens) を持つので、Recognizer をそのまま使える。
違いは採点の中身だけ:
  - kana-whisper: decoder の教師強制 log p(候補 + EOT | 音声)。候補ごとに decoder を回す (GPU)
  - komimi:       CTC forward log p(候補 | 音声) = blank を挟む全アラインメントの和。音声側の (T', 1025) 行列を 1 回作れば、
                  候補の採点は行列の上の動的計画法だけ (CPU で候補 64 個 1 ms 級)。同じ計算をブラウザの WebAssembly でも回せる
CTC は「候補で説明できない音」を blank で埋めるしかないので、短い候補 (「はい」) が長い発話に勝つことはほぼ無い
(kana-whisper の EOT が安い問題が構造的に起きない)。

モデルは KOMIMI_HOME (既定 ~/dev/komimi) の models/*.kmm、共有ライブラリは csrc/libkomimi.so。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

from .asr_types import Encoded, Transcript

KOMIMI_HOME = Path(os.environ.get("KOMIMI_HOME", Path.home() / "dev" / "komimi"))


def _komimi_engine():
    try:
        from komimi.engine import Engine
    except ImportError:
        sys.path.insert(0, str(KOMIMI_HOME))
        from komimi.engine import Engine
    return Engine


class KomimiASR:
    MIN_SEC = 0.6

    def __init__(self, kmm: str | Path):
        p = Path(kmm)
        if not p.is_absolute() and not p.exists():
            p = KOMIMI_HOME / "models" / p
        self.engine = _komimi_engine()(p)
        self.path = str(p)
        self.eot = self.engine.blank          # 互換のため (内容トークンは全部 < blank)

    def encode(self, wav: np.ndarray) -> Encoded:
        t0 = time.perf_counter()
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.mean(1)
        if len(wav) < int(self.MIN_SEC * 16000):
            wav = np.concatenate([wav, np.zeros(int(self.MIN_SEC * 16000) - len(wav), dtype=np.float32)])
        lp = self.engine.logprobs(wav)
        return Encoded(lp, len(wav) / 16000, (time.perf_counter() - t0) * 1000)

    def transcribe(self, enc: Encoded, max_new_tokens: int = 64) -> Transcript:
        t0 = time.perf_counter()
        ids = self.engine.greedy(enc.hidden)
        kana = self.engine.decode(ids).replace(" ", "")
        lp = float(self.engine.ctc_score(enc.hidden, [ids])[0]) if ids else float(enc.hidden[:, self.engine.blank].sum())
        return Transcript(kana, lp, len(ids), (time.perf_counter() - t0) * 1000, ids)

    def tokenize(self, kana: str) -> list[int]:
        return self.engine.tokenize(kana)

    def tokenize_many(self, kanas: list[str]) -> list[list[int]]:
        return [self.engine.tokenize(k) for k in kanas]

    def score(self, enc: Encoded, cands: list[str], batch_size: int = 0) -> tuple[np.ndarray, np.ndarray]:
        return self.score_tokens(enc, self.tokenize_many(cands))

    def score_tokens(self, enc: Encoded, token_lists: list[list[int]], batch_size: int = 0) -> tuple[np.ndarray, np.ndarray]:
        if not token_lists:
            return np.zeros(0), np.zeros(0, dtype=int)
        s = self.engine.ctc_score(enc.hidden, [list(t) for t in token_lists])
        return np.asarray(s, dtype=np.float64), np.array([len(t) for t in token_lists], dtype=np.int64)

    def warmup(self) -> None:
        enc = self.encode(np.zeros(16000, dtype=np.float32))
        self.transcribe(enc); self.score(enc, ["テスト"])
