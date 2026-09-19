"""Qwen3.5 系 checkpoint 同梱の MTP (multi-token prediction) head を HF 上で動かし、候補 source にする。

構造 (vLLM qwen3_5_mtp.py と同じ):
  x = fc(concat(norm_e(embed(tok_{i+1})), norm_h(h_i)))  → 1 層 full-attention decoder layer → norm → target の lm_head
  で tok_{i+2} を予測する。h_i は target 最終層 (post-norm) の hidden。
複数 token は自分の出力 hidden と予測 token を次の入力にして chain する (vLLM の num_speculative_tokens>1 と同じ)。
MTP 層は自分用の KV cache を持ち、prefix 全体に対して一度 prefill しておく。
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
from torch import nn
from safetensors import safe_open
from transformers import DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5RMSNorm


class Qwen35MTP(nn.Module):
    def __init__(self, target, model_dir: str):
        super().__init__()
        cfg = copy.deepcopy(getattr(target.config, "text_config", None) or target.config)
        cfg.layer_types = ["full_attention"]
        cfg.num_hidden_layers = 1
        H = cfg.hidden_size
        self.layer = Qwen3_5DecoderLayer(cfg, 0)
        self.fc = nn.Linear(2 * H, H, bias=False)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(H, eps=cfg.rms_norm_eps)
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(H, eps=cfg.rms_norm_eps)
        self.norm = Qwen3_5RMSNorm(H, eps=cfg.rms_norm_eps)
        self._load(model_dir)
        self.embed = target.get_input_embeddings()
        self.lm_head = target.lm_head
        self.rotary = target.model.rotary_emb
        self.to(dtype=torch.bfloat16)

    def _load(self, model_dir: str):
        d = Path(model_dir)
        idx = json.load(open(d / "model.safetensors.index.json"))["weight_map"]
        keys = [k for k in idx if k.startswith("mtp.")]
        sd = {}
        by_file: dict[str, list[str]] = {}
        for k in keys:
            by_file.setdefault(idx[k], []).append(k)
        for f, ks in by_file.items():
            with safe_open(d / f, "pt") as sf:
                for k in ks:
                    sd[k[len("mtp."):].replace("layers.0.", "layer.", 1)] = sf.get_tensor(k)
        missing, unexpected = self.load_state_dict(sd, strict=False)
        missing = [m for m in missing if not m.startswith(("embed", "lm_head", "rotary"))]
        if missing or unexpected:
            raise RuntimeError(f"MTP weight mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")

    def new_cache(self):
        return DynamicCache()

    @torch.inference_mode()
    def step(self, hidden: torch.Tensor, tok_ids: torch.Tensor, positions: torch.Tensor, cache: DynamicCache):
        """hidden (1,n,H) = h_i, tok_ids (1,n) = tok_{i+1}, positions (1,n) = i+1。戻り値 (g (1,n,H), logits (1,n,V))。"""
        e = self.pre_fc_norm_embedding(self.embed(tok_ids))
        h = self.pre_fc_norm_hidden(hidden.to(e.dtype))
        x = self.fc(torch.cat([e, h], -1))
        pos3 = positions[None].expand(3, positions.shape[0], positions.shape[1])
        cos, sin = self.rotary(x, pos3)
        n = x.shape[1]
        past = cache.get_seq_length()
        q_pos = torch.arange(past, past + n, device=x.device)[:, None]
        k_pos = torch.arange(past + n, device=x.device)[None, :]
        mask = (k_pos <= q_pos)[None, None]  # (1,1,n,kv) bool
        out = self.layer(x, position_embeddings=(cos, sin), attention_mask=mask, past_key_values=cache)
        g = self.norm(out)
        return g, self.lm_head(g)

    @torch.inference_mode()
    def prefill(self, hidden_all: torch.Tensor, full_ids: list[int], upto: int, cache: DynamicCache):
        """index 1..upto の cache を作る: pair (h_{i-1}, tok_i), i = 1..upto。"""
        if upto < 1:
            return
        dev = hidden_all.device
        toks = torch.tensor([full_ids[1 : upto + 1]], device=dev)
        pos = torch.arange(1, upto + 1, device=dev)[None]
        self.step(hidden_all[:, :upto], toks, pos, cache)

    @torch.inference_mode()
    def draft(self, h_prev: torch.Tensor, x: int, s: int, k: int, cache: DynamicCache, topk: int = 4):
        """位置 s の確定 token x と h_{s-1} から k token を chain で draft。cache は index ≤ s-1 まで入っている前提。
        戻り値 toks (k, topk), lps (k, topk): col0 = greedy chain、col1.. は各 step の top-2..。step 1 の KV は残し、chain 分は crop する。"""
        dev = h_prev.device
        toks = torch.zeros(k, topk, dtype=torch.long, device=dev)
        lps = torch.zeros(k, topk, device=dev)
        h, t = h_prev, x
        for j in range(k):
            g, logits = self.step(h, torch.tensor([[t]], device=dev), torch.tensor([[s + j]], device=dev), cache)
            lp = torch.log_softmax(logits[0, -1].float(), -1)
            top = torch.topk(lp, topk)
            toks[j], lps[j] = top.indices, top.values
            t = int(top.indices[0])
            h = g
        if k > 1:
            cache.crop(-(k - 1))
        return toks, lps
