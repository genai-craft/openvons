#!/usr/bin/env bash
# 顔・全身の属性デモの起動/停止 (pid ファイル方式)。 scripts/serve_vision.sh start|stop|restart [GPU] [PORT]
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; STATE=${OPENVONS_STATE:-$ROOT/state}/vision_attrs; mkdir -p "$STATE/logs"; PID=$STATE/server.pid; LOG=$STATE/logs/server.log
GPU="${2:-0}"; PORT="${3:-8602}"
FACE="${OPENVONS_FACE_CKPT:-/data/decision_model/checkpoints/vis_fairface_2b_meanmax}"; BODY="${OPENVONS_BODY_CKPT:-/data/decision_model/checkpoints/vis_pa100k_2b_balanced}"
stop() { if [ -f "$PID" ] && kill -0 "$(cat "$PID")" 2>/dev/null; then kill "$(cat "$PID")"; sleep 2; fi; rm -f "$PID"; }
start() { cd "$ROOT"; HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" CUDA_VISIBLE_DEVICES="$GPU" nohup "$ROOT/.venv/bin/python" -m openvons.vision.demo_server --port "$PORT" --face "$FACE" --body "$BODY" >"$LOG" 2>&1 & echo $! >"$PID"
  for i in $(seq 1 90); do sleep 2; grep -q "Uvicorn running" "$LOG" 2>/dev/null && { echo "ready on :$PORT"; return 0; }; kill -0 "$(cat "$PID")" 2>/dev/null || { tail -20 "$LOG"; return 1; }; done; tail -5 "$LOG"; return 1; }
case "${1:-}" in start) start;; stop) stop;; restart) stop; start;; *) echo "usage: $0 start|stop|restart [GPU] [PORT]"; exit 1;; esac
