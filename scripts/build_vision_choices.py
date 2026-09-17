"""カメラの「状態」を当てる質問セットを作る (SigLIP2 のテキスト側をサーバーで先に計算しておく).

    .venv/bin/python scripts/build_vision_choices.py --out /data/openjev/models/ondevice/vision-choices

端末には**画像エンコーダだけ**を置き、選択肢のテキスト埋め込みはこのスクリプトが JSON にして配る。
質問を足したり言い換えたりしても、アプリの更新は要らない (JSON を配り直すだけ)。

検出 (物体の箱) ではなく、場面の**状態**を有限の選択肢で答えさせるのが狙い:
  片付いているか / 照明が点いているか / 混んでいるか / 路面が濡れているか / 扉が開いているか / 画面が正常か …
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

#: 質問 = 有限の選択肢。prompt は SigLIP の学習分布に寄せた短い説明文 (日本語も可)
QUESTIONS = [
    {"key": "tidy", "title": "片付き具合", "choices": [
        {"id": "tidy", "label": "片付いている", "prompt": "a tidy, organized desk or room"},
        {"id": "messy", "label": "散らかっている", "prompt": "a messy, cluttered desk or room with scattered objects"}]},
    {"key": "light", "title": "照明", "choices": [
        {"id": "on", "label": "点いている", "prompt": "a room with the lights turned on, brightly lit"},
        {"id": "off", "label": "消えている", "prompt": "a dark room with the lights turned off"}]},
    {"key": "crowd", "title": "混み具合", "choices": [
        {"id": "empty", "label": "無人", "prompt": "an empty place with no people"},
        {"id": "few", "label": "数人", "prompt": "a place with a few people"},
        {"id": "crowded", "label": "混雑", "prompt": "a crowded place packed with many people"}]},
    {"key": "wet", "title": "路面・床", "choices": [
        {"id": "dry", "label": "乾いている", "prompt": "a dry road or floor surface"},
        {"id": "wet", "label": "濡れている", "prompt": "a wet road or floor surface with puddles or reflections from rain"},
        {"id": "snow", "label": "雪", "prompt": "a road or ground covered with snow"}]},
    {"key": "door", "title": "扉", "choices": [
        {"id": "open", "label": "開いている", "prompt": "a door that is open"},
        {"id": "closed", "label": "閉まっている", "prompt": "a door that is closed"}]},
    {"key": "screen", "title": "画面の状態", "choices": [
        {"id": "normal", "label": "正常に表示", "prompt": "a computer screen showing a normal working application"},
        {"id": "error", "label": "エラー表示", "prompt": "a computer screen showing an error message or a warning dialog"},
        {"id": "off", "label": "消灯・真っ暗", "prompt": "a computer screen that is turned off or black"}]},
    {"key": "weather", "title": "天気 (屋外)", "choices": [
        {"id": "sunny", "label": "晴れ", "prompt": "an outdoor scene on a sunny day with blue sky"},
        {"id": "cloudy", "label": "曇り", "prompt": "an outdoor scene on a cloudy overcast day"},
        {"id": "rain", "label": "雨", "prompt": "an outdoor scene in the rain, wet and grey"},
        {"id": "night", "label": "夜", "prompt": "an outdoor scene at night, dark"}]},
    {"key": "work", "title": "作業の状態", "choices": [
        {"id": "running", "label": "稼働中", "prompt": "machinery or equipment that is running and in operation"},
        {"id": "idle", "label": "停止中", "prompt": "machinery or equipment that is stopped and idle"},
        {"id": "maintenance", "label": "点検・作業中", "prompt": "a worker doing maintenance or repair on equipment"}]},
    {"key": "safety", "title": "安全", "choices": [
        {"id": "helmet", "label": "ヘルメット着用", "prompt": "a worker wearing a safety helmet and high visibility vest"},
        {"id": "nohelmet", "label": "着用なし", "prompt": "a person without a safety helmet"},
        {"id": "noperson", "label": "人がいない", "prompt": "a scene with no people at all"}]},
    {"key": "shelf", "title": "棚・在庫", "choices": [
        {"id": "full", "label": "十分ある", "prompt": "store shelves fully stocked with products"},
        {"id": "low", "label": "少ない", "prompt": "store shelves that are nearly empty, low stock"},
        {"id": "none", "label": "棚が写っていない", "prompt": "a scene that does not show any shelves"}]},
]


#: 河川ライブカメラの監視で実際に見たい「状態」。検出 (物の箱) ではなく、当直が目で確かめている項目を選択肢にした。
#: 監視カメラは半分の時間が夜なので、選択肢ごとに昼と夜の言い方を持たせて平均する (prompts)
QUESTIONS_KASEN = [
    {"key": "flood", "title": "水の広がり", "choices": [
        {"id": "normal", "label": "平常 (河原が見えている)", "prompts": [
            "a river in normal condition with dry riverbank and gravel visible",
            "a calm narrow river between wide dry banks",
            "a night view of a river with dry banks"]},
        {"id": "high", "label": "増水 (河原が水に浸かる)", "prompts": [
            "a swollen river with high muddy water covering the riverbank",
            "a river in flood with fast brown water near the top of the bank"]},
        {"id": "flood", "label": "一面が水", "prompts": [
            "a flooded area completely covered by water, overflowing river",
            "farmland and roads submerged under flood water"]},
        {"id": "unknown", "label": "暗くて判別できない", "prompts": [
            "a dark night camera view where the water level cannot be judged",
            "a green infrared night camera image of grass and bushes",
            "a night image too dark to tell where the water is"]}]},
    {"key": "turbid", "title": "水の濁り", "choices": [
        {"id": "clear", "label": "澄んでいる", "prompts": [
            "a river with clear blue green water", "clean transparent river water"]},
        {"id": "muddy", "label": "濁っている", "prompts": [
            "a river with brown muddy turbid water", "chocolate brown flood water"]},
        {"id": "nowater", "label": "水面が見えない", "prompts": [
            "a scene without any water surface", "a view of grass and road with no river visible",
            "a dark night scene where the water cannot be seen"]}]},
    {"key": "weather", "title": "天気", "choices": [
        {"id": "sunny", "label": "晴れ", "prompts": [
            "an outdoor scene on a sunny day with blue sky and shadows", "bright sunshine outdoors"]},
        {"id": "cloudy", "label": "曇り", "prompts": [
            "an outdoor scene on a cloudy overcast grey day", "flat grey sky over a landscape"]},
        {"id": "rain", "label": "雨", "prompts": [
            "an outdoor scene during rain, wet surfaces and raindrops", "heavy rain falling outdoors"]},
        {"id": "snow", "label": "雪", "prompts": ["an outdoor scene covered with snow", "snow falling over a landscape"]}]},
    {"key": "time", "title": "明るさ", "choices": [
        {"id": "day", "label": "昼", "prompts": ["an outdoor scene in bright daylight", "a daytime landscape photo"]},
        {"id": "dusk", "label": "薄暮", "prompts": ["an outdoor scene at dusk or dawn, dim orange light", "twilight over a river"]},
        {"id": "night", "label": "夜", "prompts": [
            "an outdoor scene at night, dark with artificial lights",
            "a dark night surveillance image with street lights",
            "a night time camera view, mostly black with a few lights"]}]},
    {"key": "visibility", "title": "見通し", "choices": [
        {"id": "clear", "label": "良好", "prompts": [
            "a clear sharp outdoor view with good visibility",
            "a sharp night camera view where the scene is recognisable"]},
        {"id": "fog", "label": "霧・もや", "prompts": [
            "an outdoor scene covered in fog or haze with poor visibility", "thick fog hiding the background"]},
        {"id": "drops", "label": "レンズに水滴・汚れ", "prompts": [
            "a camera image blurred by water droplets on the lens",
            "a dirty smeared camera lens blurring the whole picture"]}]},
    {"key": "signal", "title": "映像の状態", "choices": [
        {"id": "ok", "label": "正常", "prompts": [
            "a normal outdoor surveillance camera image in daylight",
            "a normal night surveillance camera image, dark but recognisable",
            "a usable security camera picture of an outdoor scene"]},
        {"id": "black", "label": "真っ黒・無信号", "prompts": [
            "a completely black image with no signal", "a blank empty frame, no picture"]},
        {"id": "noise", "label": "乱れ・ノイズ", "prompts": [
            "a broken video image with static noise and colour bands",
            "a glitched distorted video frame with torn lines"]}]},
    {"key": "debris", "title": "漂流物", "choices": [
        {"id": "none", "label": "見当たらない", "prompts": [
            "a river surface with nothing floating on it", "an empty water surface"]},
        {"id": "debris", "label": "流木・ごみが見える", "prompts": [
            "driftwood, logs and debris floating on a river", "branches and rubbish carried by flood water"]},
        {"id": "unknown", "label": "暗くて判別できない", "prompts": [
            "a dark night camera view where the water surface cannot be inspected",
            "a green infrared night camera image with no visible water surface"]}]},
    {"key": "people", "title": "人・車", "choices": [
        {"id": "none", "label": "いない", "prompts": [
            "an outdoor scene with no people and no vehicles", "an empty riverside with nobody around"]},
        {"id": "people", "label": "人がいる", "prompts": ["an outdoor scene with people walking", "people standing near a river"]},
        {"id": "vehicle", "label": "車がいる", "prompts": ["an outdoor scene with cars or trucks", "vehicles driving on a road"]}]},
    {"key": "structure", "title": "写っているもの", "choices": [
        {"id": "bridge", "label": "橋", "prompts": ["a bridge over a river", "a long road bridge crossing water"]},
        {"id": "weir", "label": "堰・水門", "prompts": ["a weir, floodgate or sluice on a river", "concrete flood gates on a waterway"]},
        {"id": "bank", "label": "堤防・河川敷", "prompts": [
            "an embankment and open riverbank without structures", "grass covered flood plain beside a river"]}]},
]

QUESTION_SETS = {"general": ("室内外の汎用", QUESTIONS), "kasen": ("河川カメラの監視", QUESTIONS_KASEN)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/siglip2-base-patch16-224")
    ap.add_argument("--out", default="/data/openjev/models/ondevice/vision-choices")
    ap.add_argument("--set", default="general", choices=sorted(QUESTION_SETS))
    args = ap.parse_args()
    title, questions = QUESTION_SETS[args.set]
    from transformers import AutoModel, AutoProcessor
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model = AutoModel.from_pretrained(args.model).eval()
    proc = AutoProcessor.from_pretrained(args.model)
    # 1 つの選択肢に複数の言い方を持たせ、正規化した埋め込みの平均を使う (prompt ensembling)。
    # 夜の監視カメラ画像で「乱れ・ノイズ」に倒れるなど、言い方 1 つだと外す場面が減る。
    prompts, index = [], []
    for q in questions:
        for c in q["choices"]:
            for pr in (c.get("prompts") or [c["prompt"]]):
                index.append((q["key"], c["id"])); prompts.append(pr)
    with torch.no_grad():
        tin = proc(text=prompts, padding="max_length", max_length=64, return_tensors="pt")
        emb = model.get_text_features(**tin)
        emb = emb.pooler_output if hasattr(emb, "pooler_output") else (emb.last_hidden_state if hasattr(emb, "last_hidden_state") else emb)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    acc: dict[str, dict[str, list[list[float]]]] = {}
    for (qk, cid), v in zip(index, emb.tolist()):
        acc.setdefault(qk, {}).setdefault(cid, []).append(v)
    by_key: dict[str, dict[str, list[float]]] = {}
    for qk, per_choice in acc.items():
        for cid, vs in per_choice.items():
            t = torch.tensor(vs).mean(0)
            t = t / t.norm()
            by_key.setdefault(qk, {})[cid] = [round(x, 5) for x in t.tolist()]
    doc = {"model": args.model, "set": args.set, "title": title, "logit_scale": float(model.logit_scale.exp()), "logit_bias": float(getattr(model, "logit_bias", torch.tensor(0.0))),
           "dim": emb.shape[1], "questions": [{**q, "choices": [{k: v for k, v in c.items() if k != "prompts"} for c in q["choices"]],
                          "embeddings": by_key[q["key"]]} for q in questions]}
    name = "choices.json" if args.set == "general" else f"choices_{args.set}.json"
    (out / name).write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    print(f"[{args.set}] questions {len(questions)}  choices {len(prompts)}  dim {emb.shape[1]}  -> {out/name}")


if __name__ == "__main__":
    main()
