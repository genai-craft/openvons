"""小型 VLM を使った Decision Model。

テキスト版と同じ構造 (選択肢行末ベクトル × Decision 位置ベクトル) を画像入力に拡張したもの。
質問と選択肢を自然言語で受け取れるので、学習時に無かった質問・選択肢にもある程度対応できる。

  [画像トークン | State テキスト] ... 共有
  Question: ...\nOptions:\nA. id: description\n ... \nDecision:
  選択肢行末の hidden = 選択肢ベクトル / Decision 位置の hidden = 問い合わせベクトル
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from openvons.core.formats import option_labels
from openvons.core.primitives import Question
from openvons.lm.models.heads import build_head
from openvons.lm.models.hybrid_cache import question_position_ids, repeat_cache_

os.environ.setdefault("HF_HOME", "/data/decision_model/hf_home")


@dataclass
class VLMDecisionConfig:
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct"
    head: str = "embed"
    head_dim: int = 512
    dtype: str = "bfloat16"
    state_text: str = "State:\nAn image.\n"


class VLMDecisionModel(nn.Module):
    def __init__(self, cfg: VLMDecisionConfig, device="cuda"):
        super().__init__()
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.cfg = cfg
        self.device = device
        self.proc = AutoProcessor.from_pretrained(cfg.model_name)
        self.tok = self.proc.tokenizer
        full = AutoModelForImageTextToText.from_pretrained(cfg.model_name, dtype=getattr(torch, cfg.dtype))
        self.lm_head = full.lm_head                      # zero-shot 比較用に残す (学習では使わない)
        self.backbone = full.model                       # visual + language_model
        self.backbone.to(device).eval()
        self.lm_head.to(device).eval()
        H = self.backbone.language_model.config.hidden_size
        self.hidden_size = H
        self.head = build_head(cfg.head, H, cfg.head_dim).to(device).float()
        self.temperature = 1.0
        self._cache: dict[str, list[int]] = {}

    # ---------------------------------------------------------------- encoding
    def _enc(self, s: str) -> list[int]:
        r = self._cache.get(s)
        if r is None:
            r = self.tok.encode(s, add_special_tokens=False)
            self._cache[s] = r
        return list(r)

    def state_inputs(self, image, text: str | None = None):
        msgs = [{"role": "user", "content": [{"type": "image", "image": image},
                                             {"type": "text", "text": text or self.cfg.state_text}]}]
        return self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                             return_dict=True, return_tensors="pt").to(self.device)

    def question_ids(self, q: Question) -> tuple[list[int], list[int], int]:
        ids = self._enc(f"Question: {q.text}\nOptions:\n")
        pos, nl = [], self._enc("\n")
        for lab, o in zip(option_labels(q.n), q.options):
            ids += self._enc(f"{lab}. {o.label()}")
            pos.append(len(ids) - 1)
            ids += nl
        ids += self._enc("Decision:")
        return ids, pos, len(ids) - 1

    # ---------------------------------------------------------------- forward
    @torch.no_grad()
    def features(self, image, questions: list[Question], state_text: str | None = None) -> dict[str, torch.Tensor]:
        """画像を 1 回だけ読み、全質問ぶんの (問い合わせベクトル, 選択肢ベクトル) を返す。"""
        inp = self.state_inputs(image, state_text)
        S = inp["input_ids"].shape[1]
        n = len(questions)
        cache = self.backbone(**inp, use_cache=True).past_key_values
        repeat_cache_(cache, n)
        encs = [self.question_ids(q) for q in questions]
        L = max(len(e[0]) for e in encs)
        N = max(len(e[1]) for e in encs)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        qi = torch.full((n, L), pad, dtype=torch.long)
        opos = torch.zeros((n, N), dtype=torch.long)
        omask = torch.zeros((n, N), dtype=torch.bool)
        dpos = torch.zeros(n, dtype=torch.long)
        for i, (ids, pos, dp) in enumerate(encs):
            qi[i, : len(ids)] = torch.tensor(ids)
            opos[i, : len(pos)] = torch.tensor(pos)
            omask[i, : len(pos)] = True
            dpos[i] = dp
        qi, opos, omask, dpos = (t.to(self.device) for t in (qi, opos, omask, dpos))
        h = self.backbone.language_model(
            input_ids=qi,
            attention_mask=torch.ones(n, S + L, dtype=torch.long, device=self.device),
            past_key_values=cache,
            cache_position=S + torch.arange(L, device=self.device),
            position_ids=question_position_ids(self, inp, n, L),
            use_cache=True,
        ).last_hidden_state
        ar = torch.arange(n, device=self.device)
        return {"pooled": h[ar, dpos], "opt_vecs": h[ar.unsqueeze(1), opos], "opt_mask": omask,
                "state_tokens": S, "hidden": h, "encs": encs}

    def logits_from_features(self, feats, qtype: str = "choice") -> torch.Tensor:
        return self.head(feats["pooled"].float(), feats["opt_vecs"].float(), feats["opt_mask"], qtype)

    @torch.no_grad()
    def decide(self, image, questions: list[Question], state_text: str | None = None) -> list[list[float]]:
        f = self.features(image, questions, state_text)
        lg = self.logits_from_features(f, questions[0].type)
        p = torch.softmax(lg / self.temperature, -1)
        return [p[i, : q.n].tolist() for i, q in enumerate(questions)]

    @torch.no_grad()
    def decide_zeroshot(self, image, questions: list[Question], state_text: str | None = None) -> list[list[float]]:
        """学習前の比較用: LM head 経由で選択肢ラベルトークンの確率を読む。"""
        f = self.features(image, questions, state_text)
        out = []
        for i, q in enumerate(questions):
            dp = f["encs"][i][2]
            lg = self.lm_head(f["hidden"][i, dp].to(self.lm_head.weight.dtype))
            ids = [self.tok.encode(" " + c, add_special_tokens=False)[0] for c in option_labels(q.n)]
            out.append(torch.softmax(lg[ids].float(), -1).tolist())
        return out

    # ---------------------------------------------------------------- save / load
    def save_head(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save({"head": self.head.state_dict(), "temperature": self.temperature}, os.path.join(path, "head.pt"))
        json.dump(asdict(self.cfg), open(os.path.join(path, "config.json"), "w"), indent=2, ensure_ascii=False)

    @staticmethod
    def from_checkpoint(path: str, device="cuda") -> "VLMDecisionModel":
        cfg = VLMDecisionConfig(**json.load(open(os.path.join(path, "config.json"))))
        m = VLMDecisionModel(cfg, device)
        sd = torch.load(os.path.join(path, "head.pt"), map_location=device)
        m.head.load_state_dict(sd["head"])
        m.temperature = sd.get("temperature", 1.0)
        return m
