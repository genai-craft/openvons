"""SigLIP2 の画像側だけを ONNX に書き出す (端末に置くのはこれだけ).

    .venv/bin/python scripts/export_vision_encoder.py --out /data/openjev/models/ondevice/siglip2-base-img

テキスト側はサーバーで先に計算して choices.json に入れてある (scripts/build_vision_choices.py)。
端末は「画像を埋め込む → 選択肢の埋め込みと内積 → 校正 → 判断」だけを行う。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


class ImageTower(torch.nn.Module):
    """画像 → 正規化済み埋め込み。前処理 (224x224 リサイズと正規化) は端末側で行う。"""

    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, pixel_values):
        f = self.m.get_image_features(pixel_values=pixel_values)
        f = f.pooler_output if hasattr(f, "pooler_output") else f
        return f / f.norm(dim=-1, keepdim=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/siglip2-base-patch16-224")
    ap.add_argument("--out", default="/data/openjev/models/ondevice/siglip2-base-img")
    args = ap.parse_args()
    from transformers import AutoModel, AutoProcessor
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model = AutoModel.from_pretrained(args.model).eval()
    proc = AutoProcessor.from_pretrained(args.model)
    tower = ImageTower(model).eval()
    ip = proc.image_processor
    size = ip.size.get("height", 224) if isinstance(ip.size, dict) else 224
    dummy = torch.randn(1, 3, size, size)
    with torch.no_grad():
        ref = tower(dummy)
    # 新しい dynamo 版は出力名が内部ノードと衝突して壊れるので、従来の exporter を使う
    torch.onnx.export(tower, (dummy,), str(out / "image_encoder.onnx"), input_names=["pixel_values"], output_names=["image_embeds"],
                      dynamic_axes={"pixel_values": {0: "batch"}, "image_embeds": {0: "batch"}}, opset_version=17, dynamo=False)
    meta = {"model": args.model, "size": size, "mean": list(ip.image_mean), "std": list(ip.image_std),
            "dim": int(ref.shape[-1]), "rescale": 1 / 255}
    (out / "preprocess.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    # 端末向けに int8 も (MatMul のみ。Conv を量子化すると onnxruntime-web/mobile で落ちる場合がある)
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(str(out / "image_encoder.onnx"), str(out / "image_encoder_quantized.onnx"),
                     weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=["MatMul"])
    fp32 = (out / "image_encoder.onnx").stat().st_size / 1e6
    q8 = (out / "image_encoder_quantized.onnx").stat().st_size / 1e6
    print(f"{args.model}: {size}x{size}, dim {meta['dim']}  fp32 {fp32:.0f}MB / int8 {q8:.0f}MB -> {out}")


if __name__ == "__main__":
    main()
