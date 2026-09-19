"""Intel CPU / iGPU (OpenVINO) で openvons の端末モデルが動くか・何 ms かを測る。13900HK (Raptor Lake-P, UHD Graphics) 上で実行。

  venv/bin/python bench_ov.py --models ~/ov/models
ONNX (kana-whisper 蒸留 encoder/decoder、SigLIP2 image encoder) を OpenVINO で CPU と GPU にコンパイルし、動的軸を既定形状で固定して時間を測る。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import openvino as ov

DEFAULT_DIMS = {"batch_size": 1, "batch": 1, "N": 1, "num_frames": 3000, "feature_size": 80, "sequence_length": 8, "encoder_sequence_length": 1500,
                "decoder_sequence_length": 8, "channels": 3, "height": 224, "width": 224, "past_sequence_length": 0, "seq": 8, "T": 3000}


def fixed_shape(port):
    ps = port.get_partial_shape(); dims = []
    for i, d in enumerate(ps):
        if d.is_static:
            dims.append(d.get_length())
        else:
            name = ps[i].to_string()
            dims.append(DEFAULT_DIMS.get(name.strip("[]?"), 1 if i == 0 else 8))
    return dims


def bench(core, path: Path, device: str, iters: int = 10):
    m = core.read_model(str(path))
    shapes = {}
    for inp in m.inputs:
        if inp.get_partial_shape().is_dynamic:
            shapes[inp.get_any_name()] = ov.PartialShape(fixed_shape(inp))
    if shapes:
        m.reshape(shapes)
    t0 = time.perf_counter()
    cm = core.compile_model(m, device)
    tc = time.perf_counter() - t0
    req = cm.create_infer_request()
    feeds = {}
    for inp in cm.inputs:
        shp = list(inp.get_shape()); et = inp.get_element_type().to_string()
        if "int" in et:
            feeds[inp.get_any_name()] = np.zeros(shp, dtype=np.int64 if "64" in et else np.int32)
        else:
            feeds[inp.get_any_name()] = np.random.randn(*shp).astype(np.float16 if "f16" in et else np.float32)
    for _ in range(3):
        req.infer(feeds)
    ts = []
    for _ in range(iters):
        t = time.perf_counter(); req.infer(feeds); ts.append((time.perf_counter() - t) * 1e3)
    return {"device": device, "compile_s": round(tc, 1), "median_ms": round(float(np.median(ts)), 1), "inputs": {k: list(v.shape) for k, v in feeds.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(Path.home() / "ov/models"))
    ap.add_argument("--devices", default="CPU,GPU")
    ap.add_argument("--out", default=str(Path.home() / "ov/bench.json"))
    a = ap.parse_args()
    core = ov.Core()
    print("devices:", core.available_devices, {d: core.get_property(d, "FULL_DEVICE_NAME") for d in core.available_devices}, flush=True)
    res = {}
    for f in sorted(Path(a.models).rglob("*.onnx")):
        if f.name.endswith(".onnx.data") or f.stat().st_size < 1_000_000:
            continue
        for d in a.devices.split(","):
            if d not in core.available_devices:
                continue
            try:
                r = bench(core, f, d)
                print(f"{f.parent.name}/{f.name:40s} {d:4s} compile {r['compile_s']}s  infer {r['median_ms']} ms  {r['inputs']}", flush=True)
                res[f"{f.parent.name}/{f.name}|{d}"] = r
            except Exception as e:  # noqa: BLE001
                print(f"{f.parent.name}/{f.name:40s} {d:4s} ERROR {str(e)[:160]}", flush=True)
                res[f"{f.parent.name}/{f.name}|{d}"] = {"error": str(e)[:300]}
    json.dump(res, open(a.out, "w"), indent=1)
    print("->", a.out)


if __name__ == "__main__":
    main()
