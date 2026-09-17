#!/usr/bin/env bash
# jev.lm.api.server の起動/停止 (pid ファイル方式)。  scripts/serve_lm_api.sh start|stop|restart [PORT]
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; STATE=/data/openjev/state/lm_api; mkdir -p "$STATE"; PID=$STATE/server.pid; LOG=$STATE/server.log
PORT="${2:-8410}"
stop() { if [ -f "$PID" ] && kill -0 "$(cat "$PID")" 2>/dev/null; then kill "$(cat "$PID")"; sleep 2; fi; rm -f "$PID"; }
start() { cd "$ROOT"; DM_LLM_URL="${DM_LLM_URL:-http://127.0.0.1:8300/v1}" DM_PORT="$PORT" HF_HOME="${HF_HOME:-/data/decision_model/hf_home}" nohup "$ROOT/.venv/bin/python" -m jev.lm.api.server >"$LOG" 2>&1 & echo $! >"$PID"
  for i in $(seq 1 30); do sleep 1; grep -q "Uvicorn running" "$LOG" 2>/dev/null && { echo "ready on :$PORT"; return 0; }; done; tail -5 "$LOG"; return 1; }
case "${1:-}" in start) start;; stop) stop;; restart) stop; start;; *) echo "usage: $0 start|stop|restart [PORT]"; exit 1;; esac
