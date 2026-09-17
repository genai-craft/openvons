"""全国の道路カメラ風カタログを作る (デモ用の現実的な合成データ).

実在の階層 (地方整備局 -> 国道事務所 -> 都道府県 -> 国道番号) に、日本郵便の郵便番号データ
(読み付きの町域名、著作権主張なし) から採った地名を載せて「<地名><番号><上り|下り>」型の
カメラ名を作る。国交省にはカメラ名の全国統一データが無いため (docs/research.md)、
名前の形式は首都国道事務所系の「船橋南1上り」型に揃える。

出力: data/cameras.json (lexicon 形式), data/hierarchy.json (UI の登録画面用)
"""
from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import jaconv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from openvons.voice import kana as K  # noqa: E402

KEN_ALL = Path("/data/sashizu/data/kenall/ken_all.csv")

# 地方整備局 -> 事務所 -> (都道府県, 国道番号リスト)。事務所名は実在の名称に寄せている。
HIERARCHY: dict[str, dict[str, tuple[list[str], list[int]]]] = {
    "北海道開発局": {
        "札幌開発建設部": (["北海道"], [5, 12, 36, 230, 231, 274]),
        "函館開発建設部": (["北海道"], [5, 227, 228, 278]),
        "旭川開発建設部": (["北海道"], [12, 39, 40, 237]),
        "帯広開発建設部": (["北海道"], [38, 236, 241, 274]),
    },
    "東北地方整備局": {
        "青森河川国道事務所": (["青森県"], [4, 7, 45, 101]),
        "岩手河川国道事務所": (["岩手県"], [4, 45, 46, 106]),
        "仙台河川国道事務所": (["宮城県"], [4, 6, 45, 48, 286]),
        "秋田河川国道事務所": (["秋田県"], [7, 13, 46]),
        "山形河川国道事務所": (["山形県"], [13, 47, 48, 112]),
        "福島河川国道事務所": (["福島県"], [4, 13, 49, 115]),
    },
    "関東地方整備局": {
        "首都国道事務所": (["千葉県"], [14, 357, 6]),
        "千葉国道事務所": (["千葉県"], [16, 51, 126, 127, 128, 409]),
        "東京国道事務所": (["東京都"], [1, 4, 6, 14, 15, 17, 20, 246, 357]),
        "相武国道事務所": (["東京都"], [16, 20, 129]),
        "川崎国道事務所": (["神奈川県"], [1, 15, 246, 357, 409]),
        "横浜国道事務所": (["神奈川県"], [1, 16, 129, 134, 246]),
        "大宮国道事務所": (["埼玉県"], [16, 17, 254, 463]),
        "北首都国道事務所": (["埼玉県"], [4, 16, 298]),
        "常総国道事務所": (["茨城県"], [6, 50, 51, 468]),
        "常陸河川国道事務所": (["茨城県"], [6, 50, 51, 349]),
        "宇都宮国道事務所": (["栃木県"], [4, 50, 119, 121]),
        "高崎河川国道事務所": (["群馬県"], [17, 18, 50, 354]),
        "甲府河川国道事務所": (["山梨県"], [20, 52, 138, 139]),
        "長野国道事務所": (["長野県"], [18, 19, 20, 158]),
    },
    "北陸地方整備局": {
        "新潟国道事務所": (["新潟県"], [7, 8, 49, 116]),
        "長岡国道事務所": (["新潟県"], [8, 17, 116, 351]),
        "富山河川国道事務所": (["富山県"], [8, 41, 156, 160]),
        "金沢河川国道事務所": (["石川県"], [8, 157, 159, 249]),
    },
    "中部地方整備局": {
        "名古屋国道事務所": (["愛知県"], [1, 19, 22, 23, 41, 302]),
        "静岡国道事務所": (["静岡県"], [1, 52, 139, 150]),
        "浜松河川国道事務所": (["静岡県"], [1, 42, 152, 257]),
        "岐阜国道事務所": (["岐阜県"], [21, 41, 156, 258]),
        "三重河川国道事務所": (["三重県"], [1, 23, 42, 258]),
        "飯田国道事務所": (["長野県"], [19, 153, 256]),
    },
    "近畿地方整備局": {
        "大阪国道事務所": (["大阪府"], [1, 2, 25, 26, 43, 163, 171, 176]),
        "京都国道事務所": (["京都府"], [1, 9, 24, 171]),
        "兵庫国道事務所": (["兵庫県"], [2, 28, 43, 171, 175]),
        "奈良国道事務所": (["奈良県"], [24, 25, 165, 169]),
        "滋賀国道事務所": (["滋賀県"], [1, 8, 161]),
        "和歌山河川国道事務所": (["和歌山県"], [24, 26, 42]),
    },
    "中国地方整備局": {
        "広島国道事務所": (["広島県"], [2, 31, 54, 185]),
        "岡山国道事務所": (["岡山県"], [2, 30, 53, 180]),
        "山口河川国道事務所": (["山口県"], [2, 9, 190, 191]),
        "鳥取河川国道事務所": (["鳥取県"], [9, 29, 53]),
        "松江国道事務所": (["島根県"], [9, 54, 431]),
    },
    "四国地方整備局": {
        "徳島河川国道事務所": (["徳島県"], [11, 32, 55, 192]),
        "香川河川国道事務所": (["香川県"], [11, 32, 193]),
        "松山河川国道事務所": (["愛媛県"], [11, 33, 56, 196]),
        "土佐国道事務所": (["高知県"], [32, 33, 55, 56]),
    },
    "九州地方整備局": {
        "福岡国道事務所": (["福岡県"], [3, 201, 202, 208]),
        "北九州国道事務所": (["福岡県"], [3, 10, 199, 200]),
        "佐賀国道事務所": (["佐賀県"], [34, 35, 203, 207]),
        "長崎河川国道事務所": (["長崎県"], [34, 57, 202, 206]),
        "熊本河川国道事務所": (["熊本県"], [3, 57, 208, 266]),
        "大分河川国道事務所": (["大分県"], [10, 57, 210]),
        "宮崎河川国道事務所": (["宮崎県"], [10, 220, 268]),
        "鹿児島国道事務所": (["鹿児島県"], [3, 10, 220, 225]),
    },
    "沖縄総合事務局": {
        "南部国道事務所": (["沖縄県"], [58, 329, 330, 331]),
        "北部国道事務所": (["沖縄県"], [58, 329, 449]),
    },
}

# ユーザー例の「船橋南1上り」型。首都国道事務所 (国道14号・357号) の実在っぽい地点を人手で
HANDMADE = [
    ("首都国道事務所", "千葉県", 14, "船橋南", "フナバシミナミ"), ("首都国道事務所", "千葉県", 14, "船橋北", "フナバシキタ"),
    ("首都国道事務所", "千葉県", 14, "谷津", "ヤツ"), ("首都国道事務所", "千葉県", 14, "津田沼", "ツダヌマ"),
    ("首都国道事務所", "千葉県", 14, "幕張", "マクハリ"), ("首都国道事務所", "千葉県", 14, "検見川", "ケミガワ"),
    ("首都国道事務所", "千葉県", 14, "市川", "イチカワ"), ("首都国道事務所", "千葉県", 14, "本八幡", "モトヤワタ"),
    ("首都国道事務所", "千葉県", 357, "習志野", "ナラシノ"), ("首都国道事務所", "千葉県", 357, "浦安", "ウラヤス"),
    ("首都国道事務所", "千葉県", 357, "千鳥町", "チドリチョー"), ("首都国道事務所", "千葉県", 357, "湾岸市川", "ワンガンイチカワ"),
    ("首都国道事務所", "千葉県", 357, "美浜", "ミハマ"), ("首都国道事務所", "千葉県", 357, "稲毛海岸", "イナゲカイガン"),
    ("首都国道事務所", "千葉県", 6, "松戸", "マツド"), ("首都国道事務所", "千葉県", 6, "柏", "カシワ"), ("首都国道事務所", "千葉県", 6, "我孫子", "アビコ"),
]

BAD_TOWN_WORDS = ("以下に掲載がない場合", "の次に番地", "一円", "その他", "地階", "階層不明", "（")


def load_towns() -> dict[str, list[tuple[str, str, str]]]:
    """都道府県 -> [(市区町村, 町域, 町域カナ ASR 形)] (重複除去)。"""
    out: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    with open(KEN_ALL, encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            pref, city, town = row[6], row[7], row[8]
            kana = row[5]
            if any(b in town for b in BAD_TOWN_WORDS) or not town:
                continue
            town = town.split("（")[0]
            kana = kana.split("(")[0]
            if len(town) > 6 or len(town) < 2:
                continue
            key = (pref, town)
            if key in seen:
                continue
            seen.add(key)
            kana = K.normalize(jaconv.h2z(kana, kana=True))
            if not kana:
                continue
            out[pref].append((city, town, kana))
    return out


def main(n_per_route: int = 6, seed: int = 7) -> None:
    rng = random.Random(seed)
    towns = load_towns()
    entities: list[dict] = []
    hier: dict[str, dict[str, dict]] = {}
    used_names: set[str] = set()

    def add(bureau: str, office: str, pref: str, route: int, place: str, reading: str, idx: int, direction: str, city: str = ""):
        label = f"{place}{idx}{direction}"
        if label in used_names:
            return
        used_names.add(label)
        eid = f"cam_{len(entities):05d}"
        dir_kana = "ノボリ" if direction == "上り" else "クダリ"
        readings = [K.normalize(reading + K.digits_to_kana(idx) + dir_kana)]
        entities.append({
            "id": eid, "label": label, "kind": "camera",
            "readings": readings,
            "aliases": [f"{place}{idx}", place] if idx == 1 else [f"{place}{idx}"],
            "attrs": {"bureau": bureau, "office": office, "pref": pref, "route": route, "place": place,
                      "place_reading": reading, "index": idx, "direction": direction, "city": city,
                      "route_label": f"国道{route}号"},
        })

    for bureau, offices in HIERARCHY.items():
        hier[bureau] = {}
        for office, (prefs, routes) in offices.items():
            hier[bureau][office] = {"prefs": prefs, "routes": routes, "n_cameras": 0}
            for route in routes:
                pool = [t for p in prefs for t in towns.get(p, [])]
                if not pool:
                    continue
                picks = rng.sample(pool, min(n_per_route, len(pool)))
                for city, town, kana in picks:
                    idx = rng.choice([1, 1, 1, 2, 2, 3])
                    for direction in ("上り", "下り"):
                        add(bureau, office, prefs[0], route, town, kana, idx, direction, city)
    for office, pref, route, place, reading in HANDMADE:
        for idx in (1, 2):
            for direction in ("上り", "下り"):
                add("関東地方整備局", office, pref, route, place, reading, idx, direction, "")
    for e in entities:
        hier[e["attrs"]["bureau"]][e["attrs"]["office"]]["n_cameras"] += 1

    out_dir = ROOT / "examples" / "road_cameras"
    (out_dir / "cameras.json").write_text(json.dumps(entities, ensure_ascii=False, indent=0), encoding="utf-8")
    (out_dir / "hierarchy.json").write_text(json.dumps(hier, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"cameras: {len(entities)}  bureaus: {len(hier)}  offices: {sum(len(v) for v in hier.values())}")


if __name__ == "__main__":
    main()
