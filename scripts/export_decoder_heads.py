"""デコーダを「端末で必要な答えだけ返す」2 つの ONNX に包み直す.

    .venv/bin/python scripts/export_decoder_heads.py --model /data/openjev/models/ondevice/kana-small-4l

素の decoder_model.onnx は (B, T, 語彙 51866) のロジットをそのまま返す。端末アプリでは
これが メソッドチャネル越しに丸ごと Dart に渡り、候補 20 件で 1300 万要素 → OutOfMemory で落ちた。
必要なのは「次の 1 トークン」と「候補ごとの対数尤度の和」だけなので、グラフの中で潰して返す:

  decoder_head.onnx  input_ids(B,T) + encoder_hidden_states(1,S,D) + targets(B,T) + mask(B,T)
                     -> scores(B)   候補ごとの sum log p(target|音声)
                        next_id(B)  次の 1 トークン (カナ以外の抑制はグラフに焼き込み済み)

生成と採点で同じ重みを使うので 1 つのグラフにまとめる (2 つに分けると端末に同じ重みが 2 回載る)。
生成のときは targets と mask に 0 を渡して next_id だけ読み、採点のときは scores だけ読む。
encoder の出力はグラフの中で B に広げる (端末側で複製すると 20 倍の転送になる)。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


class DecoderHead(torch.nn.Module):
    """採点 (scores) と次トークン (next_id) を 1 回の forward でまとめて返す。"""

    def __init__(self, model, suppress: list[int], vocab: int):
        super().__init__()
        self.decoder = model.model.decoder
        # 注意: proj_out は埋め込みと重みを共有しているが、定数畳み込みが転置済みの複製を作るため
        # ONNX には 51865x768 が 2 回入る (fp32 +160MB / int8 +40MB)。
        # do_constant_folding=False にすると重複は消えるが、MatMul の B が定数でなくなり
        # 動的量子化が効かなくなる (312MB のまま)。量子化後の小ささを取って畳み込みは有効のままにする
        bias = torch.zeros(vocab)
        if suppress:
            bias[torch.tensor(suppress, dtype=torch.long)] = -1e4
        self.register_buffer("suppress_bias", bias)

    def forward(self, input_ids: torch.Tensor, encoder_hidden_states: torch.Tensor,
                targets: torch.Tensor, mask: torch.Tensor):
        b = input_ids.shape[0]
        enc = encoder_hidden_states.expand(b, -1, -1)
        h = self.decoder(input_ids=input_ids, encoder_hidden_states=enc).last_hidden_state
        logits = torch.nn.functional.linear(h, self.decoder.embed_tokens.weight)
        logp = torch.log_softmax(logits, dim=-1)
        picked = torch.gather(logp, 2, targets.unsqueeze(-1)).squeeze(-1)
        scores = (picked * mask).sum(-1)
        next_id = (logits[:, -1] + self.suppress_bias).argmax(-1)
        return scores, next_id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/openjev/models/ondevice/kana-small-4l")
    ap.add_argument("--out", default="/data/openjev/models/ondevice/kana-small-4l/onnx", help="書き出し先 (端末に配る場所)")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()
    d = Path(args.model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from transformers import WhisperForConditionalGeneration
    model = WhisperForConditionalGeneration.from_pretrained(d).eval()
    cfg = model.config
    info_p = d / "distill_info.json"
    if not info_p.exists():
        info_p = out.parent / "distill_info.json"
    suppress = json.loads(info_p.read_text()).get("suppress_tokens_kana_only", []) if info_p.exists() else []
    vocab = int(model.proj_out.weight.shape[0])
    dim, layers = cfg.d_model, cfg.decoder_layers
    frames = cfg.max_source_positions        # 1500
    print(f"vocab {vocab} d_model {dim} decoder_layers {layers} frames {frames} suppress {len(suppress)}")

    enc = torch.randn(1, frames, dim)
    b, t = 3, 6
    ids_b = torch.randint(0, 1000, (b, t), dtype=torch.long)
    tgt = torch.randint(0, 1000, (b, t), dtype=torch.long)
    msk = torch.ones(b, t)
    net = DecoderHead(model, suppress, vocab)
    path = out / "decoder_head.onnx"
    # 旧 (TorchScript) エクスポータは系列長を書き出し時の例の値に焼いてしまい、
    # 長さの違う入力で Reshape が落ちる。dynamo=True なら記号次元のまま出る
    dim_b = torch.export.Dim("b", min=1, max=64)
    dim_t = torch.export.Dim("t", min=2, max=128)
    with torch.no_grad():
        torch.onnx.export(
            net, (ids_b, enc, tgt, msk), str(path),
            input_names=["input_ids", "encoder_hidden_states", "targets", "mask"],
            output_names=["scores", "next_id"],
            dynamic_shapes={"input_ids": {0: dim_b, 1: dim_t}, "encoder_hidden_states": None,
                            "targets": {0: dim_b, 1: dim_t}, "mask": {0: dim_b, 1: dim_t}},
            opset_version=args.opset, dynamo=True)

    # 数値の突き合わせ (ONNX Runtime と PyTorch)
    import numpy as np
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    g_s, g_n = sess.run(None, {"input_ids": ids_b.numpy(), "encoder_hidden_states": enc.numpy(),
                               "targets": tgt.numpy(), "mask": msk.numpy()})
    with torch.no_grad():
        w_s, w_n = net(ids_b, enc, tgt, msk)
    print("scores:", np.round(g_s, 3), "max diff", float(np.abs(g_s - w_s.numpy()).max()))
    print("next_id:", g_n, "expected", w_n.numpy(), "OK" if (g_n == w_n.numpy()).all() else "MISMATCH")
    for tt in (4, 9, 20):        # 系列長が変わっても動くこと (旧エクスポータはここで落ちた)
        a = np.random.randint(0, 1000, (2, tt)).astype(np.int64)
        sess.run(None, {"input_ids": a, "encoder_hidden_states": enc.numpy(), "targets": np.zeros_like(a),
                        "mask": np.zeros(a.shape, dtype=np.float32)})
    print("dynamic length: OK (T=4,9,20)")
    total = path.stat().st_size + sum(f.stat().st_size for f in out.glob(path.name + ".data"))
    print(f"{path.name}: {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
