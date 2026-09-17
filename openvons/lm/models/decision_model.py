from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn

from openvons.core.primitives import Question
from .backbone import load_backbone
from .encoding import DecisionTokenizer, Encoded
from .heads import build_head
from .pooling import AttentionPooling, pool


@dataclass
class DecisionModelConfig:
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    pooling: str = "last"          # last | decision | mean | attn
    head: str = "embed"            # linear | embed | mlp
    head_dim: int = 512
    dtype: str = "bfloat16"
    attn_impl: str = "sdpa"
    max_state_tokens: int = 1024


class DecisionModel(nn.Module):
    def __init__(self, cfg: DecisionModelConfig, device="cuda"):
        super().__init__()
        self.cfg = cfg
        self.device = device
        dtype = getattr(torch, cfg.dtype)
        tok, self.backbone = load_backbone(cfg.model_name, dtype, cfg.attn_impl, device)
        self.dtok = DecisionTokenizer(tok, use_decision_token=(cfg.pooling == "decision"), max_state_tokens=cfg.max_state_tokens)
        if cfg.pooling == "decision":
            emb = self.backbone.get_input_embeddings()
            if self.dtok.decision_id >= emb.weight.shape[0]:
                self.backbone.resize_token_embeddings(len(tok))
                emb = self.backbone.get_input_embeddings()
            with torch.no_grad():
                emb.weight[self.dtok.decision_id] = emb.weight[: self.dtok.decision_id].mean(0)
        H = self.backbone.config.hidden_size
        self.hidden_size = H
        self.head = build_head(cfg.head, H, cfg.head_dim).to(device)
        self.attn_pool = AttentionPooling(H).to(device) if cfg.pooling == "attn" else None
        self.temperature = 1.0
        self.head_dtype = torch.float32

    # ------------------------------------------------------------ features
    def features(self, batch: dict[str, torch.Tensor], **bb_kwargs) -> dict[str, torch.Tensor]:
        """Run the backbone and gather (pooled query vector, option vectors)."""
        out = self.backbone(input_ids=batch["input_ids"], attention_mask=bb_kwargs.pop("attention_mask", batch["attention_mask"]), **bb_kwargs)
        h = out.last_hidden_state                                       # (B, L, H)
        pooled = pool(self.cfg.pooling, h, batch["attention_mask"], batch["decision_pos"], self.attn_pool)
        B = h.shape[0]
        opt = h[torch.arange(B, device=h.device).unsqueeze(1), batch["option_pos"]]   # (B, N, H)
        return {"pooled": pooled, "opt_vecs": opt, "opt_mask": batch["option_mask"]}

    def logits_from_features(self, feats: dict[str, torch.Tensor], qtype: str) -> torch.Tensor:
        return self.head(feats["pooled"].to(self.head_dtype), feats["opt_vecs"].to(self.head_dtype), feats["opt_mask"], qtype)

    def forward(self, batch: dict[str, torch.Tensor], qtype: str) -> torch.Tensor:
        return self.logits_from_features(self.features(batch), qtype)

    # ------------------------------------------------------------ inference helpers
    @torch.no_grad()
    def probs(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits / self.temperature, dim=-1)

    def trainable_head_parameters(self):
        ps = list(self.head.parameters())
        if self.attn_pool is not None:
            ps += list(self.attn_pool.parameters())
        return ps

    # ------------------------------------------------------------ save / load (head + config only)
    def save_head(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save({"head": self.head.state_dict(),
                    "attn_pool": self.attn_pool.state_dict() if self.attn_pool is not None else None,
                    "temperature": self.temperature}, os.path.join(path, "head.pt"))
        json.dump(asdict(self.cfg), open(os.path.join(path, "config.json"), "w"), indent=2)

    def load_head(self, path: str):
        sd = torch.load(os.path.join(path, "head.pt"), map_location=self.device)
        self.head.load_state_dict(sd["head"])
        if self.attn_pool is not None and sd.get("attn_pool"):
            self.attn_pool.load_state_dict(sd["attn_pool"])
        self.temperature = sd.get("temperature", 1.0)

    @staticmethod
    def from_checkpoint(path: str, device="cuda") -> "DecisionModel":
        cfg = DecisionModelConfig(**json.load(open(os.path.join(path, "config.json"))))
        m = DecisionModel(cfg, device)
        if os.path.exists(os.path.join(path, "adapter_config.json")):
            from peft import PeftModel
            m.backbone = PeftModel.from_pretrained(m.backbone, path).merge_and_unload()
        m.load_head(path)
        return m
