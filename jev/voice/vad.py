"""Silero VAD による発話区間の切り出し (ストリーミング).

ブラウザから 16kHz PCM が細切れで届く前提。512 サンプル (32ms) ごとに発話確率を出し、
発話開始で録音を始め (直前 300ms のプリロール込み)、無音が end_silence_ms 続いたら 1 発話として返す。
コマンドは短いので max_utt_sec で強制的に切る。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

SR = 16000
CHUNK = 512


@dataclass
class VadConfig:
    threshold: float = 0.5
    end_silence_ms: int = 450
    min_speech_ms: int = 250
    preroll_ms: int = 300
    max_utt_sec: float = 8.0


class StreamingVad:
    def __init__(self, cfg: VadConfig | None = None):
        from silero_vad import load_silero_vad
        self.cfg = cfg or VadConfig()
        self.model = load_silero_vad()
        self.reset()

    def reset(self) -> None:
        self.model.reset_states()
        self._buf = np.zeros(0, dtype=np.float32)
        self._pre: list[np.ndarray] = []
        self._utt: list[np.ndarray] = []
        self._speaking = False
        self._silence = 0.0
        self._speech_ms = 0.0
        self.level = 0.0

    @property
    def speaking(self) -> bool:
        return self._speaking

    def feed(self, pcm: np.ndarray) -> list[np.ndarray]:
        """音声を追加し、確定した発話 (0 個以上) を返す。"""
        out: list[np.ndarray] = []
        self._buf = np.concatenate([self._buf, np.asarray(pcm, dtype=np.float32)])
        pre_n = int(self.cfg.preroll_ms / 1000 * SR / CHUNK) + 1
        while len(self._buf) >= CHUNK:
            chunk, self._buf = self._buf[:CHUNK], self._buf[CHUNK:]
            with torch.no_grad():
                p = float(self.model(torch.from_numpy(chunk), SR).item())
            self.level = float(np.sqrt((chunk ** 2).mean()))
            ms = CHUNK / SR * 1000
            if p >= self.cfg.threshold:
                if not self._speaking:
                    self._speaking = True
                    self._utt = list(self._pre)
                    self._speech_ms = 0.0
                self._utt.append(chunk)
                self._silence = 0.0
                self._speech_ms += ms
            else:
                if self._speaking:
                    self._utt.append(chunk)
                    self._silence += ms
                    if self._silence >= self.cfg.end_silence_ms:
                        out.extend(self._finish())
                self._pre.append(chunk)
                if len(self._pre) > pre_n:
                    self._pre.pop(0)
            if self._speaking and sum(len(c) for c in self._utt) >= self.cfg.max_utt_sec * SR:
                out.extend(self._finish())
        return out

    def _finish(self) -> list[np.ndarray]:
        utt = np.concatenate(self._utt) if self._utt else np.zeros(0, np.float32)
        self._speaking = False
        self._utt = []
        self._silence = 0.0
        if self._speech_ms < self.cfg.min_speech_ms:
            return []
        return [utt]

    def flush(self) -> list[np.ndarray]:
        return self._finish() if self._speaking else []
