"""駅名で動く路線図ビューア (一般向けデモ): 意図・状態・仮想ビューの操作.

状態:
  MAP      路線図全体。受理: 駅名 (寄る)、路線名で絞る、ヘルプ
  STATION  1 駅を選択中。受理: 次の駅 / 前の駅 / 別の駅名 / 路線を変えて / もっと寄って / 引いて / 一覧 (全体) に戻る / この駅を登録 (お気に入り、要確認)
  CONFIRM  確認待ち。受理: はい / いいえ

道路カメラのデモと同じ状態機械・同じ認識器。差し替えたのは実体 (駅) と意図の一覧だけ。
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from jev.voice.engine import Decision
from jev.voice.grammar import Grammar, Intent
from jev.voice.lexicon import Entity, Lexicon, Scope
from jev.voice.state import StateDef, StateMachine

CONFIRM_TIMEOUT_SEC = 8.0
ZOOM_STEP = 1.6

INTENTS = [
    Intent("select_station", ["{station}[を](表示|出して|見せて|お願い)", "{station}(まで|に行きたい|へ)", "{station}[に](寄って|ズーム)", "{station}"], description="駅に寄る"),
    Intent("next_station", ["(次|つぎ)[の駅]", "一つ先", "先に進んで"], description="次の駅へ"),
    Intent("prev_station", ["(前|まえ|一つ前)[の駅]", "一つ戻って", "手前"], description="前の駅へ"),
    Intent("switch_line", ["(路線|線)[を](変えて|切り替え|切り替えて)", "別の路線", "乗り換え"], description="乗換路線に切り替え"),
    Intent("zoom_in", ["[もっと](寄って|寄せて|拡大|ズームイン)", "拡大して"], description="地図を拡大"),
    Intent("zoom_out", ["[もっと](引いて|縮小|ズームアウト|広く)", "縮小して", "全体を見せて"], description="地図を縮小"),
    Intent("back", ["(全体|路線図|元の画面|一覧)[に](戻って|戻る|戻して)", "戻る", "閉じて"], description="路線図全体に戻る"),
    Intent("favorite", ["(この駅|ここ)[を](登録|お気に入り|保存)[して]", "お気に入り登録"], risk="high", description="お気に入りに登録 (要確認)"),
    Intent("yes", ["はい", "そうです", "お願いします", "OK", "実行"], description="確認: はい", allow_embed=False),
    Intent("no", ["いいえ", "違います", "キャンセル", "やめて", "取り消し"], description="確認: いいえ", allow_embed=False),
    Intent("help", ["ヘルプ", "何ができる", "コマンド一覧"], description="使えるコマンド"),
]

STATES = {
    "MAP": StateDef("MAP", ["select_station", "help"], "路線図全体。駅名を言うと寄ります"),
    "STATION": StateDef("STATION", ["next_station", "prev_station", "switch_line", "zoom_in", "zoom_out", "back", "favorite", "select_station", "help"],
                        "駅を選択中。次/前の駅・路線切替・拡大縮小・戻る"),
    "CONFIRM": StateDef("CONFIRM", ["yes", "no"], "確認待ち。はい / いいえ"),
}

#: 事前学習の校正に使う他状態の発話 (状態名, 受理意図, [(発話, 正解意図 | None=該当なし)])
CALIBRATION_STATES = [
    ("STATION", ["next_station", "prev_station", "switch_line", "zoom_in", "zoom_out", "back", "favorite", "help"], [
        ("次の駅", "next_station"), ("一つ先", "next_station"), ("前の駅", "prev_station"), ("一つ戻って", "prev_station"), ("路線を変えて", "switch_line"),
        ("もっと寄って", "zoom_in"), ("もっと引いて", "zoom_out"), ("全体に戻って", "back"), ("戻る", "back"), ("この駅を登録して", "favorite"),
        ("はい", None), ("少々お待ちください", None), ("はい、お世話になっております", None), ("了解しました", None), ("ちょっと待ってね", None), ("何時に着くかな", None),
    ]),
    ("CONFIRM", ["yes", "no"], [
        ("はい", "yes"), ("そうです", "yes"), ("お願いします", "yes"), ("いいえ", "no"), ("違います", "no"), ("キャンセル", "no"),
        ("はい、お世話になっております", None), ("はいはい、大丈夫です", None), ("この駅を登録して", None), ("全体に戻って", None), ("いいえ、こちらこそ", None), ("そうですね、確認します", None),
    ]),
]

SLOT = "station"
SELECT_INTENT = "select_station"
DATA_FILE = "stations.json"
TITLE = "駅名で動く路線図"


def default_scopes(lex: Lexicon) -> list[Scope]:
    return [
        Scope("yamanote", "JR 山手線", filters={"line_codes": ["11302"]}),
        Scope("tokyo_jr", "JR東日本 首都圏 (山手・中央・京浜東北・総武・埼京)", filters={"line_codes": ["11302", "11312", "11332", "11314", "11313"]}),
        Scope("osaka", "大阪環状線 + 御堂筋線", filters={"line_codes": ["11623", "99618"]}),
    ]


@dataclass
class View:
    zoom: float = 1.0
    line_code: str | None = None
    favorites: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"zoom": self.zoom, "line_code": self.line_code, "favorites": list(self.favorites)}


class StationApp:
    """1 セッション分。CameraApp と同じ契約 (command_set / apply / snapshot / set_scope / invalidate / expire_confirm)。"""

    def __init__(self, lexicon: Lexicon, scope: Scope):
        self.lexicon = lexicon
        self.scope = scope
        self.grammar = Grammar(INTENTS)
        self.sm = StateMachine(STATES, "MAP", {"station": None, "pending": None})
        self.view = View()
        self.log: list[dict[str, Any]] = []
        self._cs_cache: dict[str, Any] = {}

    # Scope.matches はスカラー属性の一致しか見ないので、リスト属性 (line_codes) は自前で判定する
    def entities(self) -> list[Entity]:
        out = []
        for e in self.lexicon:
            if e.id in self.scope.exclude_ids:
                continue
            if e.id in self.scope.ids:
                out.append(e); continue
            ok = bool(self.scope.filters)
            for k, allowed in self.scope.filters.items():
                v = e.attrs.get(k)
                if isinstance(v, list):
                    if not any(x in allowed for x in v):
                        ok = False; break
                elif v not in allowed:
                    ok = False; break
            if ok:
                out.append(e)
        return out

    def set_scope(self, scope: Scope) -> None:
        self.scope = scope
        self._cs_cache.clear()
        self.sm.goto("MAP", station=None, pending=None)
        self.view = View()

    def invalidate(self) -> None:
        self._cs_cache.clear()

    def command_set(self):
        st = self.sm.state
        if st not in self._cs_cache:
            self._cs_cache[st] = self.grammar.compile(self.sm.allowed_intents(), self.entities(), st)
        return self._cs_cache[st]

    def allowed_commands(self) -> list[dict[str, str]]:
        out = []
        for name in self.sm.allowed_intents():
            it = self.grammar.intents[name]
            ex = it.patterns[0].replace("[", "").replace("]", "")
            ex = ex.split("(")[0] + (ex.split("(")[1].split("|")[0] + ex.split(")")[1] if "(" in ex else "")
            out.append({"intent": name, "example": ex.replace("{station}", "<駅名>"), "description": it.description, "risk": it.risk})
        return out

    def expire_confirm(self) -> bool:
        if self.sm.state == "CONFIRM" and time.time() - self.sm.context.get("confirm_at", 0) > CONFIRM_TIMEOUT_SEC:
            self.sm.back(); return True
        return False

    # ------------------------------------------------------------ 適用
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

    def _neighbor(self, sid: str, step: int) -> Entity | None:
        e = self.lexicon.get(sid)
        pos = [p for p in e.attrs["positions"] if p["line_code"] == self.view.line_code] or e.attrs["positions"]
        p = pos[0]
        self.view.line_code = p["line_code"]
        target_index = p["index"] + step
        for cand in self.entities():
            for q in cand.attrs["positions"]:
                if q["line_code"] == p["line_code"] and q["index"] == target_index:
                    return cand
        # 範囲外の駅も許す (路線の続き)
        for cand in self.lexicon:
            for q in cand.attrs["positions"]:
                if q["line_code"] == p["line_code"] and q["index"] == target_index:
                    return cand
        return None

    def _execute(self, intent: str, slots: dict[str, str], params: dict[str, Any], text: str) -> dict[str, Any]:
        pending = self.sm.context.get("pending")
        if self.sm.state == "CONFIRM":
            if intent == "yes" and pending:
                self.sm.back()
                return self._execute(pending["intent"], pending["slots"], pending["params"], pending["text"])
            self.sm.back()
            return {"intent": intent, "speech": "取り消しました"}
        sid = self.sm.context.get("station")
        if intent == "select_station":
            e = self.lexicon.get(slots[SLOT])
            lines = [p["line_code"] for p in e.attrs["positions"]]
            if self.view.line_code not in lines:
                self.view.line_code = lines[0]
            self.view.zoom = max(self.view.zoom, 3.0)
            self.sm.goto("STATION", station=e.id, pending=None)
            return {"intent": intent, "station": e.id, "speech": f"{e.label} です。{e.attrs['route_label']}"}
        if intent == "back":
            self.sm.goto("MAP", station=None, pending=None); self.view.zoom = 1.0
            return {"intent": intent, "speech": "路線図に戻ります"}
        if intent == "help":
            return {"intent": intent, "speech": "、".join(c["example"] for c in self.allowed_commands()[:5])}
        if sid is None:
            return {"intent": intent, "speech": "駅を選んでください"}
        if intent in ("next_station", "prev_station"):
            nb = self._neighbor(sid, 1 if intent == "next_station" else -1)
            if nb is None:
                return {"intent": intent, "speech": "終点です"}
            self.sm.context["station"] = nb.id
            return {"intent": intent, "station": nb.id, "speech": f"{nb.label}"}
        if intent == "switch_line":
            e = self.lexicon.get(sid)
            lines = [p["line_code"] for p in e.attrs["positions"]]
            if len(lines) < 2:
                return {"intent": intent, "speech": "乗換路線はありません"}
            i = lines.index(self.view.line_code) if self.view.line_code in lines else -1
            self.view.line_code = lines[(i + 1) % len(lines)]
            name = next(p["line"] for p in e.attrs["positions"] if p["line_code"] == self.view.line_code)
            return {"intent": intent, "speech": f"{name} に切り替えました"}
        if intent == "zoom_in": self.view.zoom = min(self.view.zoom * ZOOM_STEP, 20.0)
        elif intent == "zoom_out": self.view.zoom = max(self.view.zoom / ZOOM_STEP, 1.0)
        elif intent == "favorite":
            if sid not in self.view.favorites:
                self.view.favorites.append(sid)
            return {"intent": intent, "speech": f"{self.lexicon.get(sid).label} を登録しました"}
        return {"intent": intent, "station": sid, "speech": text}

    def snapshot(self) -> dict[str, Any]:
        s = self.sm.snapshot()
        sid = s["context"].get("station")
        e = self.lexicon.get(sid) if sid else None
        s["camera"] = {"id": sid, "label": e.label, "attrs": e.attrs} if e else None    # UI 互換のキー名
        s["station"] = s["camera"]
        s["view"] = self.view.to_dict()
        s["allowed_commands"] = self.allowed_commands()
        s["n_hypotheses"] = len(self.command_set())
        s["n_cameras"] = len(self.entities())
        s["scope"] = {"id": self.scope.id, "name": self.scope.name}
        return s


AppClass = StationApp
