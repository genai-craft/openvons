#!/usr/bin/env bash
# ランディングページの起動/停止。 scripts/serve_landing.sh start|stop|restart [PORT]
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; STATE=${OPENVONS_STATE:-$ROOT/state}/landing; mkdir -p "$STATE/logs"; PID=$STATE/server.pid; LOG=$STATE/logs/server.log
PORT="${2:-8605}"
stop() { if [ -f "$PID" ] && kill -0 "$(cat "$PID")" 2>/dev/null; then kill "$(cat "$PID")"; sleep 1; fi; rm -f "$PID"; }
start() { cd "$ROOT"; nohup "$ROOT/.venv/bin/python" -m examples.landing.server --port "$PORT" >"$LOG" 2>&1 & echo $! >"$PID"
  for i in $(seq 1 30); do sleep 1; curl -sf -o /dev/null "http://127.0.0.1:$PORT/healthz" && { echo "ready on :$PORT"; return 0; }; done; tail -5 "$LOG"; return 1; }
case "${1:-}" in start) start;; stop) stop;; restart) stop; start;; *) echo "usage: $0 start|stop|restart [PORT]"; exit 1;; esac
