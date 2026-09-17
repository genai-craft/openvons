"""kana-whisper (sbintuitions/kana-whisper, MIT) による自由認識と候補一括採点.

whisper-large-v3-turbo 系 (encoder 32 層 / decoder 4 層)。encoder を 1 回だけ回し、
decoder を候補数ぶんバッチで回す。64 候補で約 40ms (RTX PRO 6000, fp16)。

score() が返すのは log p(候補カナ列 | 音声) の合計 (トークン単位の対数尤度の和)。
候補どうしの比較は同じ音声に対する尤度比なので、そのまま softmax にかければ
「この音声はどの候補か」の分布になる (校正は calibration.py)。
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

import numpy as np
import torch
from transformers.modeling_outputs import BaseModelOutput

DEFAULT_MODEL = "sbintuitions/kana-whisper"
SR = 16000


@dataclass
class Transcript:
    kana: str
    logprob: float          # 生成列の対数尤度の合計 (prefix と EOT を除く内容トークン)
    n_tokens: int
    ms: float
    tokens: list[int] | None = None


@dataclass
class Encoded:
    hidden: torch.Tensor    # (1, T, d)
    seconds: float
    ms: float


class KanaASR:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None, dtype: torch.dtype = torch.float16):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            dtype = torch.float32
        self.device = torch.device(device)
        self.dtype = dtype
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name, dtype=dtype).to(self.device).eval()
        self.model.generation_config.forced_decoder_ids = None
        tok = self.processor.tokenizer
        self.tok = tok
        self.prefix = tok.convert_tokens_to_ids(["<|startoftranscript|>", "<|ja|>", "<|transcribe|>", "<|notimestamps|>"])
        self.eot = tok.eos_token_id
        self._lock = threading.Lock()   # GPU を触るのは 1 スレッドずつ
        # 句読点・記号トークンは生成でも採点でも抑制する。kana-whisper は「フナバシミナミイチ、ノボリ」のように
        # 読点を挟むことがあり、候補側 (読点なし) と採点位置がずれて尤度が 10 nats 落ちる事故を防ぐ。
        # 生成時に抑制するだけでは足りない (採点時にその位置の確率質量が読点に残る) ので、両方に同じマスクを掛ける。
        self.suppress_ids = self._build_suppress_ids()
        # 蒸留した小型モデル (openvons.voice.distill) は非カナトークンの抑制リストを同梱している
        import json as _json, os as _os
        info = _os.path.join(model_name, "distill_info.json") if _os.path.isdir(model_name) else None
        if info and _os.path.exists(info):
            extra = set(_json.load(open(info)).get("suppress_tokens_kana_only", []))
            self.suppress_ids = sorted(set(self.suppress_ids) | extra - {self.eot})
        self._suppress_tensor = torch.tensor(self.suppress_ids, device=self.device, dtype=torch.long)
        self.warmup()

    _PUNCT_CHARS = set("、。，．,.!?！？「」『』・~〜〔〕（）()[]【】\"'’‘-–—:;/… \u3000")

    def _build_suppress_ids(self) -> list[int]:
        ids = set(self.model.generation_config.suppress_tokens or [])
        vocab_size = self.model.config.vocab_size
        for tid in range(vocab_size):
            if tid >= self.tok.eos_token_id and tid != self.eot:
                continue
            txt = self.tok.decode([tid], skip_special_tokens=True)
            if txt and all(ch in self._PUNCT_CHARS for ch in txt):
                ids.add(tid)
        ids.discard(self.eot)
        return sorted(ids)

    # ------------------------------------------------------------------ 前処理
    def features(self, wav: np.ndarray) -> torch.Tensor:
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.mean(1)
        f = self.processor(wav, sampling_rate=SR, return_tensors="pt").input_features
        return f.to(self.device, self.dtype)

    MIN_SEC = 1.0    # これより短い音声は末尾を無音で伸ばす (短い単語で反復ハルシネーションが出るため)

    @torch.no_grad()
    def encode(self, wav: np.ndarray) -> Encoded:
        t0 = time.perf_counter()
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.mean(1)
        if len(wav) < int(self.MIN_SEC * SR):
            wav = np.concatenate([wav, np.zeros(int(self.MIN_SEC * SR) - len(wav), dtype=np.float32)])
        with self._lock:
            enc = self.model.model.encoder(self.features(wav)).last_hidden_state
            self._sync()
        return Encoded(enc, len(wav) / SR, (time.perf_counter() - t0) * 1000)

    # ------------------------------------------------------------------ 自由認識
    @torch.no_grad()
    def transcribe(self, enc: Encoded, max_new_tokens: int = 64) -> Transcript:
        t0 = time.perf_counter()
        with self._lock:
            out = self.model.generate(
                encoder_outputs=BaseModelOutput(last_hidden_state=enc.hidden),
                max_new_tokens=max_new_tokens,
                num_beams=1,
                do_sample=False,
                suppress_tokens=self.suppress_ids,
                language="ja",
                task="transcribe",
            )
            self._sync()
        seq = out[0].tolist()
        # transformers の版によって返り値に prefix (<|startoftranscript|>...) が含まれたり含まれなかったりする。
        # 特殊トークンは全部 id >= EOT なので、それで内容トークンだけを残す
        content = [t for t in seq if t < self.eot]
        kana = self.tok.decode(content, skip_special_tokens=True).strip()
        # 対数尤度は候補と同じ関数・同じマスクで採点する (整合性が命)
        lp, n = self.score_tokens(enc, [content])
        return Transcript(kana, float(lp[0]), int(n[0]), (time.perf_counter() - t0) * 1000, content)

    # ------------------------------------------------------------------ 候補採点
    def tokenize(self, kana: str) -> list[int]:
        return self.tok.encode(kana, add_special_tokens=False)

    @torch.no_grad()
    def score(self, enc: Encoded, cands: list[str], batch_size: int = 96) -> tuple[np.ndarray, np.ndarray]:
        """各候補カナ列の log p(候補|音声) の合計と内容トークン数。"""
        return self.score_tokens(enc, [self.tokenize(c) for c in cands], batch_size)

    @torch.no_grad()
    def score_tokens(self, enc: Encoded, token_lists: list[list[int]], batch_size: int = 96) -> tuple[np.ndarray, np.ndarray]:
        """内容トークン列ごとに log p(列 + EOT | 音声) の合計を返す。EOT を含めるので
        「音声より短い候補」は EOT が出にくい分だけ自然に減点される。句読点トークンは -inf にマスクして
        自由認識と同じ分布で採点する。"""
        if not token_lists:
            return np.zeros(0), np.zeros(0, dtype=int)
        seqs = [self.prefix + list(t) + [self.eot] for t in token_lists]
        totals = np.zeros(len(seqs), dtype=np.float64)
        counts = np.zeros(len(seqs), dtype=np.int64)
        with self._lock:
            for s in range(0, len(seqs), batch_size):
                chunk = seqs[s:s + batch_size]
                L = max(len(x) for x in chunk)
                ids = torch.full((len(chunk), L), self.eot, dtype=torch.long, device=self.device)
                mask = torch.zeros((len(chunk), L), dtype=torch.bool, device=self.device)
                for i, x in enumerate(chunk):
                    ids[i, :len(x)] = torch.tensor(x, device=self.device)
                    mask[i, :len(x)] = True
                hid = enc.hidden.expand(len(chunk), -1, -1)
                logits = self.model(encoder_outputs=(hid,), decoder_input_ids=ids[:, :-1]).logits.float()
                logits.index_fill_(-1, self._suppress_tensor, float("-inf"))
                lp = torch.log_softmax(logits, -1).gather(-1, ids[:, 1:, None])[..., 0]
                m = mask[:, 1:].clone()
                m[:, : len(self.prefix) - 1] = False      # 強制 prefix の位置は採点しない
                lp = torch.where(m, lp, torch.zeros_like(lp))
                totals[s:s + len(chunk)] = lp.sum(1).cpu().numpy()
                counts[s:s + len(chunk)] = m.sum(1).cpu().numpy()
            self._sync()
        return totals, counts

    # ------------------------------------------------------------------ 補助
    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def warmup(self) -> None:
        enc = self.encode(np.zeros(SR, dtype=np.float32))
        self.transcribe(enc)
        self.score(enc, ["テスト"])

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())
