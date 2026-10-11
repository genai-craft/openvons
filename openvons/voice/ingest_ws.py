"""hayamimi 互換の WebSocket 取り込み口 (/ingest)。hayamimi (oboroge0、MIT) の規約に合わせ、同じクライアント (スマホ・ESP32・
scripts/ws_mic_client.py) がそのまま使える:
  接続 → JSON テキスト 1 枚 {"sr": 16000, "format": "pcm_s16le", "channels": 1} → 以後は pcm_s16le のバイナリフレーム
  返り: {"type":"session_start"} / {"type":"partial","text"} (発話中 0.5 s ごと) / {"type":"final","text","lang","speaker","latency_ms","tier","audio_s"}
区切りはエネルギー (−45 dBFS、20 ms 枠): 声が 0.3 s 以上 → 無音 0.35 s で確定。最大 12 s で強制確定。hayamimi の実測に倣い、
声の立ち上がりより前の実音声 0.8 s を発話に前置する (komimi の出だし対策と同じ考え方)。エンジンは ?engine=komimi-v12a (既定) など。
  python -m openvons.voice.asr_server --port 8630 --preload komimi-v12a   # /ingest も同じプロセスで受ける
"""
from __future__ import annotations
import json
import time
import numpy as np
from fastapi import WebSocket

SR = 16000
THR_DB = -45.0; FRAME = SR // 50          # 20 ms
MIN_SPEECH = 0.3; END_SIL = 0.35; MAX_SPEECH = 12.0; PARTIAL_EVERY = 0.5; PREROLL = 0.8

def _to16k(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == SR: return x
    try:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, SR); return resample_poly(x, SR // g, sr // g).astype(np.float32)
    except Exception:
        n = int(len(x) * SR / sr); return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)

def register(app, get_engine):
    @app.websocket("/ingest")
    async def ingest(ws: WebSocket, engine: str = "komimi-v12a"):
        import anyio
        await ws.accept()
        try:
            asr = get_engine(engine)
        except KeyError as e:
            await ws.send_json({"type": "warning", "code": "engine_unknown", "message": str(e)}); await ws.close(); return
        await ws.send_json({"type": "session_start"})
        sr, ch = SR, 1; ring = np.zeros(0, np.float32); utt = np.zeros(0, np.float32)
        in_speech = False; sil = 0.0; spoken = 0.0; last_partial = 0.0; t_end = 0.0

        def run(x):
            enc = asr.encode(x); tr = asr.transcribe(enc); return tr.kana

        async def finalize(reason):
            nonlocal utt, in_speech, sil, spoken, last_partial
            x = utt
            utt = np.zeros(0, np.float32); in_speech = False; sil = 0.0; spoken = 0.0; last_partial = 0.0
            if len(x) < int(MIN_SPEECH * SR): return
            t0 = time.perf_counter(); text = await anyio.to_thread.run_sync(run, x)
            await ws.send_json({"type": "final", "text": text, "lang": "ja", "speaker": "S1", "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                                "tier": engine, "audio_s": round(len(x) / SR, 2), "reason": reason})

        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect": break
            if msg.get("text") is not None:
                try:
                    hdr = json.loads(msg["text"]); sr = int(hdr.get("sr", SR)); ch = int(hdr.get("channels", 1))
                except Exception:
                    await ws.send_json({"type": "warning", "code": "bad_header", "message": msg["text"][:80]})
                continue
            data = msg.get("bytes")
            if not data: continue
            x = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            if ch > 1: x = x[: len(x) // ch * ch].reshape(-1, ch).mean(1)
            x = _to16k(x, sr)
            n = len(x) // FRAME
            for i in range(n):
                f = x[i * FRAME:(i + 1) * FRAME]; db = 10 * np.log10(float((f * f).mean()) + 1e-12)
                voiced = db > THR_DB
                if not in_speech:
                    ring = np.concatenate([ring, f])[-int(PREROLL * SR):]
                    if voiced:
                        in_speech = True; utt = ring.copy(); spoken = 0.0; sil = 0.0; last_partial = 0.0
                else:
                    utt = np.concatenate([utt, f])
                    if voiced: spoken += FRAME / SR; sil = 0.0
                    else: sil += FRAME / SR
                    dur = len(utt) / SR
                    if spoken >= MIN_SPEECH and sil >= END_SIL:
                        await finalize("silence")
                    elif dur >= MAX_SPEECH:
                        await finalize("max_speech")
                    elif spoken >= MIN_SPEECH and dur - last_partial >= PARTIAL_EVERY:
                        last_partial = dur; text = await anyio.to_thread.run_sync(run, utt[-int(MAX_SPEECH * SR):])
                        await ws.send_json({"type": "partial", "text": text})
        if in_speech:
            await finalize("close")
