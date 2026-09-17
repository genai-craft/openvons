"""Qwen3.8 系 (線形アテンション + フルアテンションのハイブリッド) の KV cache を batch 方向に複製する。

transformers の `LinearAttentionLayer` には `batch_repeat_interleave` が無く、
state を 1 回だけ prefill して複数質問に共有する方式 (spec Phase 6 v2) がそのままでは使えない。
線形アテンション層の状態は per-token の KV ではなく漸化的な state (conv_states / recurrent_states) なので、
batch 次元に複製するだけで同じことができる。
"""
from __future__ import annotations

import torch


def _repeat_tensor(t: torch.Tensor | None, n: int) -> torch.Tensor | None:
    return None if t is None else t.repeat_interleave(n, dim=0)


def repeat_cache_(cache, n: int) -> None:
    """past_key_values を in-place で n 倍に複製する (ハイブリッド構造に対応)。"""
    if n == 1:
        return
    for layer in cache.layers:
        # フルアテンション部分 (DynamicLayer 由来の keys / values)
        for attr in ("keys", "values"):
            v = getattr(layer, attr, None)
            if isinstance(v, torch.Tensor):
                setattr(layer, attr, _repeat_tensor(v, n))
            elif isinstance(v, (list, tuple)):
                setattr(layer, attr, type(v)(_repeat_tensor(x, n) for x in v))
        # 線形アテンション部分 (conv_states / recurrent_states は state_idx -> Tensor の dict)
        for attr in ("conv_states", "recurrent_states"):
            d = getattr(layer, attr, None)
            if isinstance(d, dict):
                for k, v in d.items():
                    if isinstance(v, torch.Tensor):
                        d[k] = _repeat_tensor(v, n)


def supports_native_repeat(cache) -> bool:
    return all(hasattr(l, "batch_repeat_interleave") for l in getattr(cache, "layers", []))


def question_position_ids(model, state_inputs, n_questions: int, q_len: int, device=None) -> torch.Tensor:
    """state を prefill した後に続く質問トークンの position_ids を返す。

    M-RoPE のモデルでは画像トークンが位置番号を共有するため、state の「トークン数」と
    「位置番号」が一致しない (例: 940 トークンの state が位置 99 で終わる)。
    ここを素朴に S + arange(L) にすると rope がずれ、共有 forward の結果が naive と一致しなくなる。
    """
    mm = getattr(model, "backbone", None) or (model.model if hasattr(model, "model") else model)
    device = device or state_inputs["input_ids"].device
    pid = mm.compute_3d_position_ids(
        input_ids=state_inputs["input_ids"], inputs_embeds=None,
        image_grid_thw=state_inputs.get("image_grid_thw"), video_grid_thw=state_inputs.get("video_grid_thw"),
        mm_token_type_ids=state_inputs.get("mm_token_type_ids"),
        attention_mask=state_inputs.get("attention_mask"),
    )
    if pid is None:                                   # テキストのみ: 素直な連番
        S = state_inputs["input_ids"].shape[1]
        base = S + torch.arange(q_len, device=device)
        return base.unsqueeze(0).expand(n_questions, -1)
    last = int(pid.reshape(pid.shape[0], -1)[:, -1].max())      # state 末尾の位置番号
    base = last + 1 + torch.arange(q_len, device=device)        # (L,)
    n_rope = pid.shape[0]                                       # 通常 3 (t, h, w)
    return base.view(1, 1, q_len).expand(n_rope, n_questions, q_len).contiguous()
