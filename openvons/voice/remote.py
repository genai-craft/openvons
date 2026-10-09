"""共有 ASR サーバー (openvons.voice.asr_server) の client。KanaASR / KomimiASR と同じ口なので Recognizer にそのまま渡せる.

    asr = RemoteASR("http://127.0.0.1:8630", "komimi-v12a")
    rec = Recognizer(asr)

1 発話あたりの往復: encode (encoder + 自由認識) → tokenize (候補、キャッシュに無いものだけ) → score の 3 回。
encoder の出力はサーバー側に handle で残り、この client は torch を読まない (デモのプロセスは GPU を使わない)。
"""
from __future__ import annotations

import threading
from collections import OrderedDict

import httpx
import numpy as np

from .asr_types import Encoded, Transcript


class RemoteASR:
    def __init__(self, url: str, engine: str, timeout: float = 60.0):
        self.url = url.rstrip("/")
        self.engine = engine
        self.http = httpx.Client(base_url=self.url, timeout=timeout)
        self._tok: OrderedDict[str, list[int]] = OrderedDict()
        self._tr: OrderedDict[str, Transcript] = OrderedDict()
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- 口 (KanaASR と同じ)
    def encode(self, wav: np.ndarray) -> Encoded:
        x = np.ascontiguousarray(np.asarray(wav, dtype=np.float32).reshape(-1))
        r = self.http.post("/v1/encode", params={"engine": self.engine}, content=x.tobytes(),
                           headers={"Content-Type": "application/octet-stream"})
        r.raise_for_status()
        j = r.json()
        tr = Transcript(j["kana"], float(j["logprob"]), int(j["n_tokens"]), float(j["ms_transcribe"]), j["tokens"])
        with self._lock:
            self._tr[j["handle"]] = tr
            while len(self._tr) > 256:
                self._tr.popitem(last=False)
        return Encoded(j["handle"], float(j["seconds"]), float(j["ms_encode"]))

    def transcribe(self, enc: Encoded, max_new_tokens: int = 64) -> Transcript:
        with self._lock:
            tr = self._tr.get(enc.hidden)
        if tr is None:
            raise KeyError("transcript for this handle is gone (encode again)")
        return tr

    def tokenize(self, kana: str) -> list[int]:
        return self.tokenize_many([kana])[0]

    def tokenize_many(self, kanas: list[str]) -> list[list[int]]:
        with self._lock:
            miss = [k for k in dict.fromkeys(kanas) if k not in self._tok]
        if miss:
            r = self.http.post("/v1/tokenize", json={"engine": self.engine, "texts": miss})
            r.raise_for_status()
            with self._lock:
                for k, ids in zip(miss, r.json()["ids"]):
                    self._tok[k] = ids
                while len(self._tok) > 200_000:
                    self._tok.popitem(last=False)
        with self._lock:
            return [list(self._tok[k]) for k in kanas]

    def score(self, enc: Encoded, cands: list[str], batch_size: int = 0) -> tuple[np.ndarray, np.ndarray]:
        return self.score_tokens(enc, self.tokenize_many(cands))

    def score_tokens(self, enc: Encoded, token_lists: list[list[int]], batch_size: int = 0) -> tuple[np.ndarray, np.ndarray]:
        if not token_lists:
            return np.zeros(0), np.zeros(0, dtype=int)
        r = self.http.post("/v1/score", json={"handle": enc.hidden, "token_lists": [list(map(int, t)) for t in token_lists]})
        r.raise_for_status()
        j = r.json()
        return np.asarray(j["scores"], dtype=np.float64), np.asarray(j["counts"], dtype=np.int64)

    # ---------------------------------------------------------------- 補助
    def warmup(self) -> None:
        self.encode(np.zeros(16000, dtype=np.float32))


def list_engines(url: str, timeout: float = 5.0) -> list[dict]:
    r = httpx.get(url.rstrip("/") + "/v1/engines", timeout=timeout)
    r.raise_for_status()
    return r.json()["engines"]
