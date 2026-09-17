"""OpenAI 互換 POST /v1/audio/speech {model, input, voice, response_format: wav}."""
from __future__ import annotations

from .base import TTSBackend


class OpenAICompatTTS(TTSBackend):
    name = "openai"

    def __init__(self, base_url: str, model: str, voices: list[str], api_key: str, **kw):
        super().__init__(**kw)
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.voices = voices
        self.api_key = api_key

    def ok(self) -> bool:
        try:
            return self.client.get(self.base_url + "/models", headers={"Authorization": f"Bearer {self.api_key}"}).status_code == 200
        except Exception:
            return False

    def _request(self, text: str, seed: int) -> bytes:
        r = self.client.post(self.base_url + "/audio/speech", headers={"Authorization": f"Bearer {self.api_key}"},
                             json={"model": self.model, "input": text, "voice": self.voices[seed % len(self.voices)], "response_format": "wav"})
        r.raise_for_status()
        return r.content
