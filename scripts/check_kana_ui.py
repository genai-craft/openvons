"""音声デモの画面で、kana 入力・振り分けの切り替えを実際のブラウザ (headless Chromium) で通す.

    .venv/bin/python scripts/check_kana_ui.py --url http://127.0.0.1:8600 --phrases "日吉倉1上り,はい、お世話になっております"

各組み合わせ (振り分け: サーバー / ブラウザ WebAssembly × エンジン) について、画面のテキスト試験 (TTS → 認識) を通し、
判断・聞こえたカナ・各段の時間を表にする。マイクは使わない (サーバーの TTS 音声をブラウザで振り分ける)。
"""
from __future__ import annotations

import argparse
import json
import time

from playwright.sync_api import sync_playwright


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8600")
    ap.add_argument("--phrases", default="日吉倉1上り,はい、お世話になっております")
    ap.add_argument("--combos", default="server:kana-whisper,server:komimi-v12a,wasm:komimi-v12m,wasm:komimi-v12a,wasm:komimi-v12s")
    ap.add_argument("--timeout", type=float, default=120)
    args = ap.parse_args()
    phrases = [p for p in args.phrases.split(",") if p]
    rows, errors = [], []
    with sync_playwright() as pw:
        br = pw.chromium.launch()
        pg = br.new_page()
        pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto(args.url, wait_until="networkidle")
        pg.wait_for_function("() => window.OVKana && document.querySelector('#kanaCtl select')")
        pg.evaluate("() => { window.__res = []; const ws = WebSocket.prototype; }")
        # 結果イベントを横取りして記録する (デモの onMessage はそのまま動く)
        pg.evaluate("""() => { const orig = OVKana.onMessage; OVKana.onMessage = (m) => { if (m.type === 'result') window.__res.push(m); return orig(m); }; }""")
        for combo in args.combos.split(","):
            route, engine = combo.split(":")
            pg.select_option("#kanaCtl select[data-k=route]", route)
            pg.select_option("#kanaCtl select[data-k=engine]", engine)
            t0 = time.time()
            if route == "wasm":
                ok = pg.evaluate("() => OVKana.ensureWorker()")
                if not ok:
                    rows.append({"combo": combo, "error": pg.inner_text("#kanaCtl .kstat")}); continue
            ready_s = round(time.time() - t0, 1)
            time.sleep(0.5)
            for ph in phrases:
                n0 = pg.evaluate("() => window.__res.length")
                pg.fill("#sayText", ph)
                pg.dispatch_event("#sayForm", "submit")
                pg.wait_for_function(f"() => window.__res.length > {n0}", timeout=args.timeout * 1000)
                m = pg.evaluate("() => window.__res[window.__res.length - 1]")
                d = m["decision"]
                rows.append({"combo": combo, "ready_s": ready_s, "phrase": ph, "free_kana": d["free_kana"], "action": d["action"],
                             "top": d["top"]["text"] if d["top"] else None, "p": d["top"]["prob"] if d["top"] else None,
                             "none": d["none_prob"], "route": m.get("route"), "engine": m.get("engine"), "timings": d["timings_ms"]})
                # 元の状態に戻す (カメラを開いたら戻る)
                if d["action"] in ("execute", "confirm"):
                    pg.evaluate("() => { const ws = document.querySelector('#kanaCtl'); }")
                    pg.keyboard.press("Escape")
                    time.sleep(0.4)
        status = pg.inner_text("#kanaCtl .kstat")
        br.close()
    for r in rows:
        print(json.dumps(r, ensure_ascii=False))
    print("status:", status)
    print("console errors:", errors[:10])


if __name__ == "__main__":
    main()
