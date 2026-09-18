"""河川ライブカメラを声で見る (kasen): 国土交通省 関東地方整備局の河川ライブカメラを、地点名で選び、上流・下流へ移る.

出典: 関東地方整備局ウェブサイト (https://www.ktr.mlit.go.jp/river/bousai/river_bousai00000080.html) の各事務所ライブカメラ。
公共データ利用規約 (PDL1.0) に基づき、カメラ一覧を加工して作成。画像は表示時にその都度取得 (10 分キャッシュ) し、保存しない。

状態:
  MAP      地図全体。受理: 地点名 (表示)、河川名で絞る、ヘルプ
  CAMERA   1 地点を表示中。受理: 上流 / 下流 / 別の地点 / 更新 / 拡大 / 縮小 / 一覧に戻る / お気に入り (要確認)
  CONFIRM  確認待ち。受理: はい / いいえ
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from openvons.voice.engine import Decision
from openvons.voice.grammar import Grammar, Intent, code_hypotheses, example_of, number_hypotheses
from openvons.voice.lexicon import Entity, Lexicon, Scope
from openvons.voice.state import StateDef, StateMachine

CONFIRM_TIMEOUT_SEC = 8.0
ZOOM_STEP = 1.5
DEFAULT_ZOOM = 1.0     # 画像は元の解像度で表示する (引き伸ばすと荒れる)。拡大はホイールか「寄って」で
MAX_ZOOM = 6.0

#: 地図を動かす意図。方向 8 つ × 量 3 段を組み合わせて作る (1 つずつ書くと 24 個になるため)。
#: 量は画面の何割動かすかで、端末側が地図に渡す。
MAP_DIRS = [
    ("up", "上", ["北"]), ("down", "下", ["南"]), ("left", "左", ["西"]), ("right", "右", ["東"]),
    ("upleft", "左上", []), ("upright", "右上", []), ("downleft", "左下", []), ("downright", "右下", []),
]
MAP_MAGS = [("small", ["ちょっと", "少し"], 0.25), ("normal", [""], 0.6), ("large", ["大きく", "ぐっと", "もっと"], 1.2)]


def _pan_intents() -> list["Intent"]:
    out = []
    for dkey, dword, alts in MAP_DIRS:
        words = [dword, *alts]
        for mkey, mwords, amount in MAP_MAGS:
            pats = []
            for m in mwords:
                for w in words:
                    if m:
                        # 「ちょっと右」だけでも通す (5 モーラあるので雑音には強い)
                        pats.append(f"{m}{w}[に|へ][動かして|ずらして|移動して|寄せて]")
                    else:
                        # 量を言わないときは動詞を必須にする (「右」単独は短すぎて雑音に弱い)
                        pats.append(f"{w}(に|へ)(動かして|ずらして|移動して|寄せて)")
                        pats.append(f"{w}(の方|のほう)[に|へ][動かして|ずらして]")
            out.append(Intent(f"pan_{dkey}_{mkey}", pats, params={"dir": dkey, "amount": amount},
                              description=f"地図を{mwords[0]}{dword}へ"))
    return out


INTENTS = [
    Intent("select_camera", ["{camera}[を](表示|出して|見せて|映して|お願い)", "{camera}[に](切り替え|切り替えて)", "{camera}"], description="地点のカメラを表示"),
    Intent("upstream", ["(上流|一つ上流|上流側)[へ|に][移って|行って|見せて]", "上流のカメラ", "ひとつ上流"], description="上流のカメラへ"),
    Intent("downstream", ["(下流|一つ下流|下流側)[へ|に][移って|行って|見せて]", "下流のカメラ", "ひとつ下流"], description="下流のカメラへ"),
    Intent("refresh", ["(更新|最新)[して|の画像]", "リロード", "今の画像"], description="画像を更新"),
    Intent("zoom_in", ["[もっと](寄って|拡大|ズームイン)", "拡大して"], description="画像を拡大"),
    Intent("zoom_out", ["[もっと](引いて|縮小|ズームアウト)", "縮小して", "全体を見せて"], description="画像を縮小"),
    Intent("back", ["(一覧|全体|地図|元の画面|ホーム|最初|トップ)[に|へ](戻って|戻る|戻して)", "戻る", "閉じて",
                    "ホーム[へ|に]", "地図[へ|に]"], description="地図に戻る (いつでも使えます)"),
    Intent("favorite", ["(この地点|ここ|このカメラ)[を](登録|お気に入り|保存)[して]", "お気に入り登録"], risk="high", description="お気に入りに登録 (要確認)"),
    # 確認への返事は、それ自体を確認し直さない (「いいえ でよろしいですか」と聞き返す無限ループになる)
    Intent("yes", ["はい", "そうです", "お願いします", "OK", "実行"], description="確認: はい",
           allow_embed=False, confirmable=False, positive=True),
    Intent("no", ["いいえ", "違います", "キャンセル", "やめて", "取り消し"], description="確認: いいえ",
           allow_embed=False, confirmable=False, positive=False),
    Intent("help", ["ヘルプ", "何ができる", "コマンド一覧"], description="使えるコマンド"),
] + _pan_intents()

#: どの状態でも受け付ける意図。確認待ちで行き止まりにならないように、戻ると助けはいつでも通す
GLOBAL_INTENTS = ["back", "help"]
PAN_INTENTS = [f"pan_{d}_{m}" for d, _, _ in MAP_DIRS for m, _, _ in MAP_MAGS]

STATES = {
    # 地図では地点名に加えて、地図そのものを動かせる (拡大縮小と 8 方向 × 3 段)
    "MAP": StateDef("MAP", ["select_camera", "zoom_in", "zoom_out", *PAN_INTENTS, *GLOBAL_INTENTS],
                    "地図全体。地点名を言うとそのカメラを表示します。拡大縮小と上下左右の移動も"),
    "CAMERA": StateDef("CAMERA", ["upstream", "downstream", "refresh", "zoom_in", "zoom_out", "favorite", "select_camera", *GLOBAL_INTENTS],
                       "カメラ表示中。上流 / 下流・更新・拡大縮小・戻る"),
    # 確認待ちでも「戻る」と「ヘルプ」は通す。はい / いいえ しか受け付けないと行き止まりに感じる
    "CONFIRM": StateDef("CONFIRM", ["yes", "no", *GLOBAL_INTENTS], "確認待ち。はい / いいえ (「戻る」で取り消し)"),
}

CALIBRATION_STATES = [
    ("CAMERA", ["upstream", "downstream", "refresh", "zoom_in", "zoom_out", "back", "favorite", "help"], [
        ("上流へ", "upstream"), ("上流のカメラ", "upstream"), ("下流へ", "downstream"), ("一つ下流", "downstream"), ("更新して", "refresh"),
        ("もっと寄って", "zoom_in"), ("もっと引いて", "zoom_out"), ("地図に戻って", "back"), ("戻る", "back"), ("この地点を登録して", "favorite"),
        ("はい", None), ("少々お待ちください", None), ("はい、お世話になっております", None), ("了解しました", None), ("ちょっと待ってね", None), ("水位はどうかな", None),
    ]),
    ("CONFIRM", ["yes", "no"], [
        ("はい", "yes"), ("そうです", "yes"), ("お願いします", "yes"), ("いいえ", "no"), ("違います", "no"), ("キャンセル", "no"),
        ("はい、お世話になっております", None), ("はいはい、大丈夫です", None), ("この地点を登録して", None), ("地図に戻って", None), ("いいえ、こちらこそ", None), ("そうですね、確認します", None),
    ]),
]

SLOT = "camera"
SELECT_INTENT = "select_camera"
DATA_FILE = "cameras.json"
TITLE = "河川ライブカメラを声で見る"
TTS_HINTS = {"下流": "カリュー", "上流": "ジョーリュー", "水位": "すいい", "排水機場": "はいすいきじょう", "樋管": "ひかん"}
ATTRIBUTION = "出典：関東地方整備局ウェブサイト (https://www.ktr.mlit.go.jp/river/bousai/river_bousai00000080.html) を加工して作成"


def default_scopes(lex: Lexicon) -> list[Scope]:
    rivers = lex.values_of("river")
    offices = lex.values_of("office")
    out = [Scope("tone", "利根川 (上流河川事務所)", filters={"office": ["利根川上流河川事務所"]})]
    for rid, name, rs in [("arakawa", "荒川 (上流・下流)", ["荒川", "入間川", "越辺川", "都幾川", "高麗川", "小畔川"]), ("hitachi", "久慈川・那珂川", ["久慈川", "那珂川"]),
                          ("kasumi", "霞ヶ浦 (西浦・北浦・常陸川)", ["西浦", "北浦", "常陸川", "北利根川"]), ("watarase", "渡良瀬川", ["渡良瀬川", "思川", "巴波川"])]:
        rs = [r for r in rs if r in rivers]
        if rs:
            out.append(Scope(rid, name, filters={"river": rs}))
    out.append(Scope("all", "関東地整 全カメラ", filters={"office": offices}))
    return out


@dataclass
class View:
    zoom: float = 1.0
    favorites: list[str] = field(default_factory=list)
    refreshed_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"zoom": self.zoom, "favorites": list(self.favorites), "refreshed_at": self.refreshed_at}


class KasenApp:
    """1 セッション分。他のデモと同じ契約。"""

    def __init__(self, lexicon: Lexicon, scope: Scope):
        self.lexicon = lexicon
        self.scope = scope
        self.grammar = Grammar(INTENTS)
        self.sm = StateMachine(STATES, "MAP", {"camera": None, "pending": None})
        self.view = View()
        self.log: list[dict[str, Any]] = []
        self._cs_cache: dict[str, Any] = {}

    def entities(self) -> list[Entity]:
        return self.lexicon.in_scope(self.scope)

    def set_scope(self, scope: Scope) -> None:
        self.scope = scope; self._cs_cache.clear(); self.sm.goto("MAP", camera=None, pending=None); self.view = View()

    def invalidate(self) -> None:
        self._cs_cache.clear()

    def command_set(self):
        st = self.sm.state
        if st not in self._cs_cache:
            cs = self.grammar.compile(self.sm.allowed_intents(), self.entities(), st)
            # 名前を全部覚えなくて済むように、画面に出ている整理番号でも選べるようにする。
            # 主役はコード (C06 = シーゼロロク。音が長く雑談とぶつかりにくい)。「6 番」も通す
            if SELECT_INTENT in self.sm.allowed_intents():
                from openvons.voice.grammar import CommandSet
                coded = self.coded_entities()
                cs = CommandSet(
                    cs.hyps
                    + code_hypotheses(SELECT_INTENT, SLOT, coded)
                    + number_hypotheses(SELECT_INTENT, SLOT, [e for _, e in coded]), st)
            self._cs_cache[st] = cs
        return self._cs_cache[st]

    def coded_entities(self) -> list[tuple[str, Entity]]:
        """画面に出る並び (河川ごと上流→下流) と同じ順に C01, C02, ... を振る。
        番号は範囲 (スコープ) ごとに振り直す。担当範囲が変われば番号も変わる。"""
        ents = sorted(self.entities(), key=lambda e: (e.attrs.get("river", ""), e.attrs.get("order") or 0))
        return [(f"C{i:02d}", e) for i, e in enumerate(ents, 1)]

    def allowed_commands(self) -> list[dict[str, str]]:
        out = []
        pan_shown = False
        for name in self.sm.allowed_intents():
            if name.startswith("pan_"):
                # 24 個を 1 行にまとめる (方向 8 × 量 3 をそのまま並べても読めない)
                if not pan_shown:
                    pan_shown = True
                    out.append({"intent": "pan", "example": "ちょっと右に動かして",
                                "description": "地図を動かす (上下左右・斜め / ちょっと・大きく)", "risk": "low"})
                continue
            it = self.grammar.intents[name]
            ex = example_of(it.patterns[0], {SLOT: "<地点名>"})
            out.append({"intent": name, "example": ex.replace("{camera}", "<地点名>"), "description": it.description, "risk": it.risk})
        return out

    def expire_confirm(self) -> bool:
        if self.sm.state == "CONFIRM" and time.time() - self.sm.context.get("confirm_at", 0) > CONFIRM_TIMEOUT_SEC:
            self.sm.back(); return True
        return False

    def apply(self, d: Decision) -> dict[str, Any]:
        ev: dict[str, Any] = {"type": "result", "decision": d.to_dict(), "applied": None, "speech": None}
        if d.action == "reject":
            ev["speech"] = "聞き取れませんでした"
        elif d.action == "confirm":
            h = d.top.hypothesis
            self.sm.goto("CONFIRM", pending={"intent": h.intent, "slots": h.slots, "params": h.params, "text": h.text}, confirm_at=time.time())
            ev["speech"] = f"{h.text} ですか？"
        elif d.action == "execute":
            h = d.top.hypothesis
            ev["applied"] = self._execute(h.intent, h.slots, h.params, h.text)
            ev["speech"] = ev["applied"].get("speech")
        ev["state"] = self.snapshot()
        self.log.append({"free": d.free_kana, "action": d.action, "top": d.top.to_dict() if d.top else None, "state": self.sm.state})
        return ev

    def _neighbor(self, cid: str, step: int) -> Entity | None:
        """同じ河川で order が step だけ違う地点 (order は上流→下流の順)。範囲外の地点も許す。"""
        e = self.lexicon.get(cid)
        river, order = e.attrs.get("river"), e.attrs.get("order")
        if order is None:
            return None
        cands = [x for x in self.lexicon if x.attrs.get("river") == river and x.attrs.get("order") is not None]
        cands.sort(key=lambda x: x.attrs["order"])
        idx = next((i for i, x in enumerate(cands) if x.id == cid), None)
        if idx is None or not (0 <= idx + step < len(cands)):
            return None
        return cands[idx + step]

    def _execute(self, intent: str, slots: dict[str, str], params: dict[str, Any], text: str) -> dict[str, Any]:
        pending = self.sm.context.get("pending")
        if self.sm.state == "CONFIRM":
            if intent == "yes" and pending:
                self.sm.back()
                return self._execute(pending["intent"], pending["slots"], pending["params"], pending["text"])
            self.sm.back()
            if intent in ("back", "help"):       # 確認待ちからの逃げ道
                self.sm.context["pending"] = None
                return self._execute(intent, slots, params, text)
            return {"intent": intent, "speech": "取り消しました"}
        # 地図の移動と、地図表示中の拡大縮小は、カメラを選んでいなくても効く
        if intent.startswith("pan_"):
            return {"intent": intent, "map": {"pan": params.get("dir"), "amount": params.get("amount", 0.6)},
                    "speech": text}
        if self.sm.state == "MAP" and intent in ("zoom_in", "zoom_out"):
            return {"intent": intent, "map": {"zoom": 1 if intent == "zoom_in" else -1},
                    "speech": "拡大します" if intent == "zoom_in" else "縮小します"}

        cid = self.sm.context.get("camera")
        if intent == "select_camera":
            e = self.lexicon.get(slots[SLOT]); self.view.zoom = DEFAULT_ZOOM; self.view.refreshed_at = time.time()
            self.sm.goto("CAMERA", camera=e.id, pending=None)
            return {"intent": intent, "camera": e.id, "speech": f"{e.label}、{e.attrs.get('river', '')} です"}
        if intent == "back":
            self.sm.goto("MAP", camera=None, pending=None); self.view.zoom = 1.0
            return {"intent": intent, "speech": "地図に戻ります"}
        if intent == "help":
            return {"intent": intent, "speech": "、".join(c["example"] for c in self.allowed_commands()[:5])}
        if cid is None:
            return {"intent": intent, "speech": "地点を選んでください"}
        if intent in ("upstream", "downstream"):
            nb = self._neighbor(cid, -1 if intent == "upstream" else 1)
            if nb is None:
                return {"intent": intent, "speech": "この先にカメラはありません"}
            self.sm.context["camera"] = nb.id; self.view.zoom = DEFAULT_ZOOM; self.view.refreshed_at = time.time()
            return {"intent": intent, "camera": nb.id, "speech": f"{nb.label}"}
        if intent == "refresh":
            self.view.refreshed_at = time.time(); return {"intent": intent, "camera": cid, "speech": "画像を更新します"}
        if intent == "zoom_in": self.view.zoom = min(self.view.zoom * ZOOM_STEP, MAX_ZOOM)
        elif intent == "zoom_out": self.view.zoom = max(self.view.zoom / ZOOM_STEP, 1.0)
        elif intent == "favorite":
            if cid not in self.view.favorites:
                self.view.favorites.append(cid)
            return {"intent": intent, "speech": f"{self.lexicon.get(cid).label} を登録しました"}
        return {"intent": intent, "camera": cid, "speech": text}

    def snapshot(self) -> dict[str, Any]:
        s = self.sm.snapshot()
        cid = s["context"].get("camera")
        e = self.lexicon.get(cid) if cid else None
        s["camera"] = {"id": cid, "label": e.label, "attrs": e.attrs} if e else None
        s["view"] = self.view.to_dict()
        s["allowed_commands"] = self.allowed_commands()
        s["n_hypotheses"] = len(self.command_set())
        s["n_cameras"] = len(self.entities())
        s["scope"] = {"id": self.scope.id, "name": self.scope.name}
        s["codes"] = {e.id: code for code, e in self.coded_entities()}
        s["attribution"] = ATTRIBUTION
        return s


# ---------------------------------------------------------------- アプリ固有 API: ライブ画像のプロキシ (10 分キャッシュ、保存しない)
def register_routes(app, G, state_dir):
    import httpx
    from fastapi import HTTPException
    from fastapi.responses import Response
    cache: dict[str, tuple[float, bytes, str]] = {}
    client = httpx.Client(timeout=20.0, headers={"User-Agent": "Mozilla/5.0 (openvons kasen demo; +https://github.com/genai-craft/openvons)"}, follow_redirects=True)

    @app.get("/api/image/{cid}")
    def image(cid: str, force: int = 0):
        e = G["lexicon"].get(cid)
        if e is None or not e.attrs.get("image_url"):
            raise HTTPException(404, "no image for this camera")
        url = e.attrs["image_url"]
        now = time.time()
        hit = cache.get(cid)
        if hit and not force and now - hit[0] < 600:
            return Response(hit[1], media_type=hit[2], headers={"Cache-Control": "public, max-age=300", "X-Source": url})
        try:
            r = client.get(url, headers={"Referer": e.attrs.get("page_url") or url})
            r.raise_for_status()
        except Exception as ex:  # noqa: BLE001
            if hit:
                return Response(hit[1], media_type=hit[2], headers={"X-Stale": "1"})
            raise HTTPException(502, f"upstream image failed: {ex}")
        ctype = r.headers.get("content-type", "image/jpeg").split(";")[0]
        cache[cid] = (now, r.content, ctype)
        return Response(r.content, media_type=ctype, headers={"Cache-Control": "public, max-age=300", "X-Source": url})


AppClass = KasenApp
