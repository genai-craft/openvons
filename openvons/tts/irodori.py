"""Irodori TTS の HTTP ワーカー (Aunvox deploy/tts/irodori/tts_server.py、POST /tts {text, seed})."""
from __future__ import annotations

from .base import TTSBackend


class IrodoriTTS(TTSBackend):
    name = "irodori"

    def __init__(self, base_url: str, **kw):
        super().__init__(**kw)
        self.base_url = base_url.rstrip("/")

    def ok(self) -> bool:
        try:
            return bool(self.client.get(self.base_url + "/health").json().get("ok"))
        except Exception:
            return False

    def _request(self, text: str, seed: int) -> bytes:
        r = self.client.post(self.base_url + "/tts", json={"text": text, "seed": seed})
        r.raise_for_status()
        return r.content
