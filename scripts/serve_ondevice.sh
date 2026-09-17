#!/usr/bin/env bash
# 端末内推論デモの起動/停止。 scripts/serve_ondevice.sh start|stop|restart [GPU] [PORT]
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; STATE=${OPENVONS_STATE:-$ROOT/state}/ondevice; mkdir -p "$STATE/logs"; PID=$STATE/server.pid; LOG=$STATE/logs/server.log
GPU="${2:-2}"; PORT="${3:-8606}"
stop() { if [ -f "$PID" ] && kill -0 "$(cat "$PID")" 2>/dev/null; then kill "$(cat "$PID")"; sleep 2; fi; rm -f "$PID"; }
start() { cd "$ROOT"; HF_HOME="${HF_HOME:-/data/lychee_ja/hf_home}" CUDA_VISIBLE_DEVICES="$GPU" nohup "$ROOT/.venv/bin/python" -m examples.ondevice.server --port "$PORT" --server-asr >"$LOG" 2>&1 & echo $! >"$PID"
  for i in $(seq 1 60); do sleep 2; grep -q "Uvicorn running" "$LOG" 2>/dev/null && { echo "ready on :$PORT"; return 0; }; kill -0 "$(cat "$PID")" 2>/dev/null || { tail -15 "$LOG"; return 1; }; done; tail -5 "$LOG"; return 1; }
case "${1:-}" in start) start;; stop) stop;; restart) stop; start;; *) echo "usage: $0 start|stop|restart [GPU] [PORT]"; exit 1;; esac
