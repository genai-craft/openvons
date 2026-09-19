"""Greedy lossless block verifier (§9.1) の最小実装。HF transformers + DynamicCache、batch=1。

verify_step: cache には x_t より前の全 token が入っている状態で [x_t, d_1..d_L] を一括 forward し、
argmax と候補を順に比較、一致 prefix を受理、位置 a の target token を採用、KV を crop する。
"""
from __future__ import annotations

import torch
from transformers import DynamicCache


class GreedyVerifier:
    def __init__(self, model, device: str):
        self.model = model
        self.device = device
        self.cache = DynamicCache()

    @torch.no_grad()
    def prefill(self, prompt_ids: list[int]) -> int:
        """prompt の最後の token 以外を cache に入れ、最後の token を返す (次 step の x_t)。"""
        self.cache = DynamicCache()
        ids = torch.tensor([prompt_ids[:-1]], device=self.device)
        self.model(input_ids=ids, past_key_values=self.cache, use_cache=True)
        return prompt_ids[-1]

    @torch.no_grad()
    def step(self, x_t: int, draft: tuple[int, ...]) -> tuple[list[int], int]:
        """戻り値: (新たに確定した token 列 (受理 prefix + 訂正 token), 受理長 a)。draft=() なら通常 decode。"""
        base = self.cache.get_seq_length()
        toks = [x_t, *draft]
        ids = torch.tensor([toks], device=self.device)
        pos = torch.arange(base, base + len(toks), device=self.device)[None]
        out = self.model(input_ids=ids, past_key_values=self.cache, position_ids=pos, use_cache=True)
        y = out.logits[0].argmax(-1).tolist()  # y[i] = toks[i] の次の token
        a = 0
        while a < len(draft) and draft[a] == y[a]:
            a += 1
        new = list(draft[:a]) + [y[a]]
        self.cache.crop(-(len(draft) - a)) if a < len(draft) else None
        return new, a
