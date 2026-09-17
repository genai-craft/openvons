"""TASK-005: Qwen backbone -> hidden states (no LM head used)."""
from __future__ import annotations

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("HF_HOME", "/data/decision_model/hf_home")


def load_backbone(model_name: str, dtype=torch.bfloat16, attn_impl: str = "sdpa", device="cuda"):
    tok = AutoTokenizer.from_pretrained(model_name)
    lm = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, attn_implementation=attn_impl)
    backbone = lm.model            # decoder stack without lm_head
    del lm.lm_head                 # H1: the LM head is not needed for decisions
    backbone.to(device)
    backbone.eval()
    return tok, backbone
