"""視覚エンコーダ単体の Decision Model (spec §16 固定業務専用モデル)。

言語モデルを一切使わない。VLM から視覚エンコーダだけを取り出し、パッチ特徴をプーリングして
質問ごとの小さな head を載せる。質問は事前登録した固定のものに限られるが、
パラメータ・レイテンシともに VLM 版の 1/10 前後で済み、エッジ機器に載る。

  image -> vision tower (凍結) -> pooling -> {質問key: Linear(H, n_options)} -> softmax
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

os.environ.setdefault("HF_HOME", "/data/decision_model/hf_home")


@dataclass
class VisionDecisionConfig:
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct"
    pooling: str = "meanmax"       # mean | meanmax | attn
    dtype: str = "bfloat16"
    questions: dict | None = None  # {key: {"question":..., "choices":[{"id","description"}]}}


class _AttnPool(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.q = nn.Parameter(torch.randn(hidden) * 0.02)
        self.k = nn.Linear(hidden, hidden, bias=False)

    def forward(self, h):                      # h: (B, P, H)
        s = (self.k(h) @ self.q) / h.shape[-1] ** 0.5
        w = torch.softmax(s.float(), -1).to(h.dtype)
        return (w.unsqueeze(-1) * h).sum(1)


def load_vision_tower(model_name: str, dtype=torch.bfloat16, device="cuda"):
    """VLM チェックポイントから視覚エンコーダと画像プロセッサだけを取り出す。"""
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(model_name)
    full = AutoModelForImageTextToText.from_pretrained(model_name, dtype=dtype)
    tower = full.model.visual
    tower.to(device).eval()
    del full.model.language_model
    if hasattr(full, "lm_head"):
        del full.lm_head
    del full
    torch.cuda.empty_cache()
    return proc, tower


class VisionDecisionModel(nn.Module):
    def __init__(self, cfg: VisionDecisionConfig, device="cuda"):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.proc, self.tower = load_vision_tower(cfg.model_name, getattr(torch, cfg.dtype), device)
        H = self.tower.config.hidden_size
        self.hidden_size = H
        self.pool = _AttnPool(H).to(device).float() if cfg.pooling == "attn" else None
        D = H * 2 if cfg.pooling == "meanmax" else H
        self.feat_dim = D
        # 生のパッチ特徴はノルムが数百あり共通成分が支配的なので、head の前に LayerNorm を入れる
        self.norm = nn.LayerNorm(D).to(device).float()
        self.heads = nn.ModuleDict()
        self.temperature = {}
        for key, q in (cfg.questions or {}).items():
            self.heads[key] = nn.Linear(D, len(q["choices"]))
            self.temperature[key] = 1.0
        self.heads.to(device).float()

    # ---------------------------------------------------------------- features
    @torch.no_grad()
    def patch_features(self, images: list) -> torch.Tensor:
        """(B, D) にプールした視覚特徴。画像ごとにサイズ (= パッチ数) が違ってもよい。"""
        inp = self.proc(images=images, text=["x"] * len(images), return_tensors="pt").to(self.device)
        grid = inp["image_grid_thw"]
        o = self.tower(inp["pixel_values"], grid_thw=grid)
        h = (o.last_hidden_state if hasattr(o, "last_hidden_state") else o[0]).float()   # (総パッチ数, H)
        counts = [int(g.prod().item()) for g in grid]                    # 画像ごとの出力パッチ数 (last_hidden_state は merge 前)
        assert sum(counts) == h.shape[0], (sum(counts), h.shape)
        out = []
        for part in torch.split(h, counts):
            p = part.unsqueeze(0)                                        # (1, P_i, H)
            if self.cfg.pooling == "attn":
                out.append(self.pool(p)[0])
            elif self.cfg.pooling == "meanmax":
                out.append(torch.cat([p.mean(1), p.amax(1)], -1)[0])
            else:
                out.append(p.mean(1)[0])
        return torch.stack(out)

    def logits(self, feats: torch.Tensor, key: str) -> torch.Tensor:
        return self.heads[key](self.norm(feats.float()))

    @torch.no_grad()
    def decide(self, images: list, keys: list[str] | None = None) -> dict[str, torch.Tensor]:
        f = self.patch_features(images)
        keys = keys or list(self.heads)
        return {k: torch.softmax(self.logits(f, k) / self.temperature.get(k, 1.0), -1) for k in keys}

    # ---------------------------------------------------------------- save / load
    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save({"heads": self.heads.state_dict(), "norm": self.norm.state_dict(),
                    "pool": self.pool.state_dict() if self.pool is not None else None,
                    "temperature": self.temperature}, os.path.join(path, "head.pt"))
        json.dump(asdict(self.cfg), open(os.path.join(path, "config.json"), "w"), indent=2, ensure_ascii=False)

    @staticmethod
    def from_checkpoint(path: str, device="cuda") -> "VisionDecisionModel":
        cfg = VisionDecisionConfig(**json.load(open(os.path.join(path, "config.json"))))
        m = VisionDecisionModel(cfg, device)
        sd = torch.load(os.path.join(path, "head.pt"), map_location=device)
        m.heads.load_state_dict(sd["heads"])
        if sd.get("norm"): m.norm.load_state_dict(sd["norm"])
        if m.pool is not None and sd.get("pool"):
            m.pool.load_state_dict(sd["pool"])
        m.temperature = sd.get("temperature", {})
        return m

    def n_trainable(self) -> int:
        n = sum(p.numel() for p in self.heads.parameters()) + sum(p.numel() for p in self.norm.parameters())
        return n + (sum(p.numel() for p in self.pool.parameters()) if self.pool is not None else 0)

    def n_frozen(self) -> int:
        return sum(p.numel() for p in self.tower.parameters())
