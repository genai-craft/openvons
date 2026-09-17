#!/usr/bin/env bash
# テキスト判断デモの起動/停止 (pid ファイル方式)。 scripts/serve_text.sh start|stop|restart [PORT]
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; STATE=${OPENVONS_STATE:-$ROOT/state}/text_decision; mkdir -p "$STATE/logs"; PID=$STATE/server.pid; LOG=$STATE/logs/server.log
PORT="${2:-8604}"
stop() { if [ -f "$PID" ] && kill -0 "$(cat "$PID")" 2>/dev/null; then kill "$(cat "$PID")"; sleep 2; fi; rm -f "$PID"; }
start() { cd "$ROOT"; nohup "$ROOT/.venv/bin/python" -m examples.text_decision.server --port "$PORT" >"$LOG" 2>&1 & echo $! >"$PID"
  for i in $(seq 1 40); do sleep 1; grep -q "Uvicorn running" "$LOG" 2>/dev/null && { echo "ready on :$PORT"; return 0; }; kill -0 "$(cat "$PID")" 2>/dev/null || { tail -15 "$LOG"; return 1; }; done; tail -5 "$LOG"; return 1; }
case "${1:-}" in start) start;; stop) stop;; restart) stop; start;; *) echo "usage: $0 start|stop|restart [PORT]"; exit 1;; esac
