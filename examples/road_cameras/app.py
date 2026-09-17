"""カメラ監視デモのドメイン: 意図の定義・状態機械・仮想カメラの操作.

状態:
  WALL     担当範囲のカメラ一覧。受理: カメラ名 (拡大表示)、ヘルプ
  FOCUS    1 台を拡大表示中。受理: PTZ (右/左/上/下/寄る/引く/ホーム)、別カメラ名、戻る、プリセット保存
  CONFIRM  確認待ち。受理: はい / いいえ だけ (選択肢 2 つ = ほぼ間違えない)

仮想カメラは pan/tilt (度) と zoom (倍率) を持つだけ。実機なら ONVIF 等に差し替える。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from openvons.voice.engine import Decision
from openvons.voice.grammar import Grammar, Intent
from openvons.voice.lexicon import Entity, Lexicon, Scope
from openvons.voice.state import StateDef, StateMachine

CONFIRM_TIMEOUT_SEC = 8.0     # 確認待ちはこの秒数で自動キャンセル (電話相手への「はい」を拾わないため)
PAN_STEP = 15.0
TILT_STEP = 10.0
ZOOM_STEP = 1.5

INTENTS = [
    Intent("select_camera", ["{camera}[を](表示|出して|見せて|映して|お願い)", "{camera}[に](切り替え|切り替えて)", "{camera}"],
           description="カメラを拡大表示"),
    Intent("pan_right", ["[もっと]右[に](向けて|振って|回して|旋回)", "右旋回", "右"], params={"dir": "right"}, description="右に旋回"),
    Intent("pan_left", ["[もっと]左[に](向けて|振って|回して|旋回)", "左旋回", "左"], params={"dir": "left"}, description="左に旋回"),
    Intent("tilt_up", ["[もっと]上[に](向けて|振って|向いて)", "上"], params={"dir": "up"}, description="上に向ける"),
    Intent("tilt_down", ["[もっと]下[に](向けて|振って|向いて)", "下"], params={"dir": "down"}, description="下に向ける"),
    Intent("zoom_in", ["[もっと](寄って|寄せて|拡大|ズームイン|アップ)", "拡大して", "ズームイン"], params={"dir": "in"}, description="ズームイン"),
    Intent("zoom_out", ["[もっと](引いて|引き|縮小|ズームアウト|広く)", "縮小して", "ズームアウト", "全体を見せて"], params={"dir": "out"}, description="ズームアウト"),
    Intent("home", ["(ホーム|初期位置|正面)[に](戻して|戻って)", "ホームポジション"], description="初期位置に戻す"),
    Intent("back", ["(一覧|全体|元の画面)[に](戻って|戻る|戻して)", "戻る", "閉じて"], description="一覧に戻る"),
    Intent("save_preset", ["(この位置|ここ)[を](保存|プリセット保存|登録)[して]", "プリセット保存"], risk="high", description="プリセット保存 (要確認)"),
    Intent("yes", ["はい", "そうです", "お願いします", "OK", "実行"], description="確認: はい", allow_embed=False),
    Intent("no", ["いいえ", "違います", "キャンセル", "やめて", "取り消し"], description="確認: いいえ", allow_embed=False),
    Intent("help", ["ヘルプ", "何ができる", "コマンド一覧"], description="使えるコマンド"),
]

#: 事前学習の校正に使う他状態の発話 (状態名, 受理意図, [(発話, 正解意図 | None=該当なし)])
CALIBRATION_STATES = [
    ("FOCUS", STATES_FOCUS_INTENTS := ["pan_right", "pan_left", "tilt_up", "tilt_down", "zoom_in", "zoom_out", "home", "back", "save_preset", "help"], [
        ("もっと右に向けて", "pan_right"), ("右に振って", "pan_right"), ("左に向けて", "pan_left"), ("上に向いて", "tilt_up"), ("もっと下に向けて", "tilt_down"),
        ("もっと寄って", "zoom_in"), ("ズームイン", "zoom_in"), ("もっと引いて", "zoom_out"), ("ズームアウト", "zoom_out"), ("ホームに戻して", "home"),
        ("一覧に戻る", "back"), ("戻る", "back"), ("この位置を保存して", "save_preset"),
        ("はい", None), ("少々お待ちください", None), ("はい、お世話になっております", None), ("了解しました", None), ("ちょっと待ってね", None), ("大丈夫です", None),
    ]),
    ("CONFIRM", ["yes", "no"], [
        ("はい", "yes"), ("そうです", "yes"), ("お願いします", "yes"), ("いいえ", "no"), ("違います", "no"), ("キャンセル", "no"),
        ("はい、お世話になっております", None), ("はいはい、大丈夫です", None), ("この位置を保存して", None), ("一覧に戻る", None), ("いいえ、こちらこそ", None), ("そうですね、確認します", None),
    ]),
]

SLOT = "camera"
SELECT_INTENT = "select_camera"
DATA_FILE = "cameras.json"
TITLE = "道路カメラ 音声コマンド"
AppClass = None   # 末尾で束縛


def default_scopes(lex):
    return [Scope("shuto", "首都国道事務所 (千葉 R14/R357/R6)", filters={"office": ["首都国道事務所"]}),
            Scope("kanto_chiba", "千葉県 全域", filters={"pref": ["千葉県"]})]


STATES = {
    "WALL": StateDef("WALL", ["select_camera", "help"], "一覧表示中。カメラ名を言うと拡大します"),
    "FOCUS": StateDef("FOCUS", ["pan_right", "pan_left", "tilt_up", "tilt_down", "zoom_in", "zoom_out", "home", "back", "save_preset", "select_camera", "help"],
                      "拡大表示中。旋回・ズーム・別カメラ・戻る"),
    "CONFIRM": StateDef("CONFIRM", ["yes", "no"], "確認待ち。はい / いいえ"),
}


@dataclass
class Ptz:
    pan: float = 0.0
    tilt: float = 0.0
    zoom: float = 1.0
    presets: list[dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"pan": self.pan, "tilt": self.tilt, "zoom": self.zoom, "presets": list(self.presets)}


class CameraApp:
    """1 セッション分のアプリ状態。recognize の結果 (Decision) を受けて状態を進める。"""

    def __init__(self, lexicon: Lexicon, scope: Scope):
        self.lexicon = lexicon
        self.scope = scope
        self.grammar = Grammar(INTENTS)
        self.sm = StateMachine(STATES, "WALL", {"camera": None, "pending": None})
        self.ptz: dict[str, Ptz] = {}
        self.log: list[dict[str, Any]] = []
        self._cs_cache: dict[str, Any] = {}

    # ------------------------------------------------------------ コマンド集合
    def entities(self) -> list[Entity]:
        return self.lexicon.in_scope(self.scope)

    def set_scope(self, scope: Scope) -> None:
        self.scope = scope
        self._cs_cache.clear()
        self.sm.goto("WALL", camera=None, pending=None)

    def invalidate(self) -> None:
        self._cs_cache.clear()

    def command_set(self):
        st = self.sm.state
        if st not in self._cs_cache:
            self._cs_cache[st] = self.grammar.compile(self.sm.allowed_intents(), self.entities(), st, self.sm.current.extra_patterns)
        return self._cs_cache[st]

    def allowed_commands(self) -> list[dict[str, str]]:
        """UI 用: 今の状態で言えること。"""
        out = []
        for name in self.sm.allowed_intents():
            it = self.grammar.intents[name]
            ex = it.patterns[0].replace("[", "").replace("]", "")
            ex = ex.split("(")[0] + (ex.split("(")[1].split("|")[0] + ex.split(")")[1] if "(" in ex else "")
            out.append({"intent": name, "example": ex.replace("{camera}", "<カメラ名>"), "description": it.description, "risk": it.risk})
        return out

    # ------------------------------------------------------------ 適用
    def expire_confirm(self) -> bool:
        """確認待ちがタイムアウトしていれば元の状態へ戻す。発話処理の前に呼ぶ。"""
        import time
        if self.sm.state == "CONFIRM" and time.time() - self.sm.context.get("confirm_at", 0) > CONFIRM_TIMEOUT_SEC:
            self.sm.back()
            return True
        return False

    def apply(self, d: Decision) -> dict[str, Any]:
        """Decision を状態に適用し、UI へのイベントを返す。"""
        ev: dict[str, Any] = {"type": "result", "decision": d.to_dict(), "applied": None, "speech": None}
        if d.action == "none":
            ev["speech"] = None
        elif d.action == "reject":
            ev["speech"] = "聞き取れませんでした"
        elif d.action == "confirm":
            h = d.top.hypothesis
            import time
            self.sm.goto("CONFIRM", pending={"intent": h.intent, "slots": h.slots, "params": h.params, "text": h.text}, confirm_at=time.time())
            ev["speech"] = f"{h.text} ですか？"
        elif d.action == "execute":
            h = d.top.hypothesis
            ev["applied"] = self._execute(h.intent, h.slots, h.params, h.text)
            ev["speech"] = ev["applied"].get("speech")
        ev["state"] = self.snapshot()
        self.log.append({"free": d.free_kana, "action": d.action, "top": d.top.to_dict() if d.top else None, "state": self.sm.state})
        return ev

    def _execute(self, intent: str, slots: dict[str, str], params: dict[str, Any], text: str) -> dict[str, Any]:
        pending = self.sm.context.get("pending")
        if self.sm.state == "CONFIRM":
            if intent == "yes" and pending:
                self.sm.back()
                return self._execute(pending["intent"], pending["slots"], pending["params"], pending["text"])
            self.sm.back()
            return {"intent": intent, "speech": "取り消しました"}
        cam_id = self.sm.context.get("camera")
        if intent == "select_camera":
            cid = slots["camera"]
            e = self.lexicon.get(cid)
            self.ptz.setdefault(cid, Ptz())
            self.sm.goto("FOCUS", camera=cid, pending=None)
            return {"intent": intent, "camera": cid, "speech": f"{e.label} を表示します"}
        if intent == "back":
            self.sm.goto("WALL", camera=None, pending=None)
            return {"intent": intent, "speech": "一覧に戻ります"}
        if intent == "help":
            return {"intent": intent, "speech": "、".join(c["example"] for c in self.allowed_commands()[:5])}
        if cam_id is None:
            return {"intent": intent, "speech": "カメラを選んでください"}
        p = self.ptz.setdefault(cam_id, Ptz())
        if intent == "pan_right": p.pan = min(p.pan + PAN_STEP, 170)
        elif intent == "pan_left": p.pan = max(p.pan - PAN_STEP, -170)
        elif intent == "tilt_up": p.tilt = min(p.tilt + TILT_STEP, 45)
        elif intent == "tilt_down": p.tilt = max(p.tilt - TILT_STEP, -45)
        elif intent == "zoom_in": p.zoom = min(p.zoom * ZOOM_STEP, 12.0)
        elif intent == "zoom_out": p.zoom = max(p.zoom / ZOOM_STEP, 1.0)
        elif intent == "home": p.pan, p.tilt, p.zoom = 0.0, 0.0, 1.0
        elif intent == "save_preset": p.presets.append({"pan": p.pan, "tilt": p.tilt, "zoom": p.zoom})
        return {"intent": intent, "camera": cam_id, "ptz": p.to_dict(), "speech": text}

    def snapshot(self) -> dict[str, Any]:
        s = self.sm.snapshot()
        cid = s["context"].get("camera")
        e = self.lexicon.get(cid) if cid else None
        s["camera"] = {"id": cid, "label": e.label, "attrs": e.attrs} if e else None
        s["ptz"] = self.ptz[cid].to_dict() if cid and cid in self.ptz else None
        s["allowed_commands"] = self.allowed_commands()
        s["n_hypotheses"] = len(self.command_set())
        s["n_cameras"] = len(self.entities())
        s["scope"] = {"id": self.scope.id, "name": self.scope.name}
        return s


AppClass = CameraApp
