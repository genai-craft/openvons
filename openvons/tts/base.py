from __future__ import annotations

import fcntl
import hashlib
import io
import os
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR = 16000
LOCK_PATH = os.environ.get("JEV_TTS_LOCK", "/tmp/jev_tts.lock")


def to16k(a: np.ndarray, sr: int) -> np.ndarray:
    if a.ndim > 1:
        a = a.mean(1)
    if sr != SR:
        from math import gcd
        g = gcd(sr, SR)
        a = resample_poly(a, SR // g, sr // g)
    return a.astype(np.float32)


class TTSBackend:
    """共通部分: キャッシュ、プロセス横断ロック、16kHz 化。サブクラスは _request(text, seed) -> (bytes wav)."""

    name = "base"

    def __init__(self, cache_dir: str | Path | None = None, timeout: float = 120.0):
        self.client = httpx.Client(timeout=timeout)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def ok(self) -> bool:
        raise NotImplementedError

    def _request(self, text: str, seed: int) -> bytes:
        raise NotImplementedError

    def synth(self, text: str, seed: int = 0) -> np.ndarray:
        key = None
        if self.cache_dir:
            key = self.cache_dir / (hashlib.sha1(f"{self.name}|{text}|{seed}".encode()).hexdigest() + ".wav")
            if key.exists():
                a, sr = sf.read(key)
                return to16k(a, sr)
        # 同時リクエストを直列化 (HTTP ワーカーによっては同時に投げると別テキストの音声が返る)
        with open(LOCK_PATH, "w") as lockf:
            fcntl.flock(lockf, fcntl.LOCK_EX)
            try:
                data = self._request(text, seed)
            finally:
                fcntl.flock(lockf, fcntl.LOCK_UN)
        a, sr = sf.read(io.BytesIO(data))
        wav = to16k(a, sr)
        if key:
            sf.write(key, wav, SR)
        return wav


def get_backend(url: str, cache_dir: str | Path | None = None) -> TTSBackend:
    u = urlparse(url)
    q = {k: v[0] for k, v in parse_qs(u.query).items()}
    if u.scheme == "voicevox":
        from .voicevox import VoicevoxTTS
        return VoicevoxTTS(f"http://{u.netloc}", speakers=[int(x) for x in q.get("speaker", "3,2,8,10,9,11").split(",")], cache_dir=cache_dir)
    if u.scheme == "irodori":
        from .irodori import IrodoriTTS
        return IrodoriTTS(f"http://{u.netloc}", cache_dir=cache_dir)
    if u.scheme == "openai":
        from .openai_compat import OpenAICompatTTS
        return OpenAICompatTTS(f"http://{u.netloc}{u.path}", model=q.get("model", "tts-1"), voices=q.get("voice", "alloy,nova,onyx").split(","), api_key=q.get("api_key") or os.environ.get("OPENAI_API_KEY", "x"), cache_dir=cache_dir)
    if u.scheme in ("http", "https"):     # 後方互換: 素の URL は Irodori ワーカー扱い
        from .irodori import IrodoriTTS
        return IrodoriTTS(url, cache_dir=cache_dir)
    raise ValueError(f"unknown TTS url: {url}")
