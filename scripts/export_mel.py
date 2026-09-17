"""Whisper の前処理 (log-mel) を ONNX に書き出す。端末は生の 16kHz 波形を渡すだけでよくなる。

    .venv/bin/python scripts/export_mel.py --out /data/openjev/models/ondevice/kana-small-4l/onnx/mel.onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


class LogMel(torch.nn.Module):
    """16kHz の波形 (1, N) -> log-mel (1, n_mels, 3000)。transformers の WhisperFeatureExtractor と同じ定義。"""

    def __init__(self, mel_filters: np.ndarray, n_fft: int = 400, hop: int = 160, n_samples: int = 480000):
        super().__init__()
        self.n_fft, self.hop, self.n_samples = n_fft, hop, n_samples
        win = torch.hann_window(n_fft)
        # torch.stft は複素数を返すため ONNX に書き出せない。DFT 行列との積に展開する (n_fft=400 なので十分軽い)
        k = torch.arange(n_fft // 2 + 1).unsqueeze(1)           # (201, 1)
        n = torch.arange(n_fft).unsqueeze(0)                    # (1, 400)
        ang = 2 * torch.pi * k * n / n_fft
        self.register_buffer("cos_w", (torch.cos(ang) * win).unsqueeze(1).contiguous())    # (201, 1, 400) conv1d の重み
        self.register_buffer("sin_w", (-torch.sin(ang) * win).unsqueeze(1).contiguous())
        self.register_buffer("filters", torch.from_numpy(mel_filters).float())

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = audio[:, : self.n_samples]
        pad = self.n_samples - x.shape[1]
        x = torch.nn.functional.pad(x, (0, pad)) if pad > 0 else x
        x = torch.nn.functional.pad(x, (self.n_fft // 2, self.n_fft // 2), mode="reflect")   # center=True と同じ
        # unfold は ONNX に出せないので、窓付き DFT を conv1d の重みとして畳み込む
        x = x.unsqueeze(1)                                        # (B, 1, L)
        re = torch.nn.functional.conv1d(x, self.cos_w, stride=self.hop)   # (B, 201, T)
        im = torch.nn.functional.conv1d(x, self.sin_w, stride=self.hop)
        mag = (re ** 2 + im ** 2)[:, :, :-1]                      # 最後のフレームを落とす
        mel = self.filters @ mag
        log = torch.clamp(mel, min=1e-10).log10()
        log = torch.maximum(log, log.amax(dim=(-2, -1), keepdim=True) - 8.0)
        return (log + 4.0) / 4.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/openjev/distill/kana-whisper-small-4l")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from transformers import WhisperFeatureExtractor
    fe = WhisperFeatureExtractor.from_pretrained(args.model)
    m = LogMel(np.asarray(fe.mel_filters).T if np.asarray(fe.mel_filters).shape[0] != fe.feature_size else np.asarray(fe.mel_filters)).eval()
    audio = torch.from_numpy(np.random.randn(1, 32000).astype(np.float32) * 0.05)
    with torch.no_grad():
        got = m(audio)
        ref = torch.from_numpy(fe(audio[0].numpy(), sampling_rate=16000, return_tensors="np").input_features)
    diff = (got - ref).abs().max().item()
    print(f"mel shape {tuple(got.shape)}  transformers との最大差 {diff:.5f}")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(m, (audio,), str(out), input_names=["audio"], output_names=["mel"],
                      dynamic_axes={"audio": {0: "batch", 1: "samples"}}, opset_version=17, dynamo=False)
    print("saved", out, f"{out.stat().st_size/1e3:.0f} KB")


if __name__ == "__main__":
    main()
