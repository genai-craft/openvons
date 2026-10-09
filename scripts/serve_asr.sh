#!/usr/bin/env bash
# 共有 ASR サーバー (kana-whisper + komimi) の起動/停止。音声デモは全部これに問い合わせる (OPENVONS_ASR_URL)。
#   scripts/serve_asr.sh start [GPU] [PORT]      既定 GPU 0、:8630 (127.0.0.1 のみで待ち受け)
#   scripts/serve_asr.sh stop | restart [GPU] [PORT] | status
# pid ファイル方式 (pkill -f は自分の shell を殺す事故があるため使わない)。
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; STATE=${OPENVONS_STATE:-$ROOT/state}/asr; mkdir -p "$STATE/logs"
PID=$STATE/server.pid; LOG=$STATE/logs/server.log
GPU="${2:-0}"; PORT="${3:-8630}"
stop() { if [ -f "$PID" ] && kill -0 "$(cat "$PID")" 2>/dev/null; then kill "$(cat "$PID")"; sleep 2; fi; rm -f "$PID"; }
start() {
  cd "$ROOT"
  HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" CUDA_VISIBLE_DEVICES="$GPU" nohup "$ROOT/.venv/bin/python" -m openvons.voice.asr_server --port "$PORT" >"$LOG" 2>&1 &
  echo $! > "$PID"
  for i in $(seq 1 90); do sleep 2; if curl -sf -m 2 "http://127.0.0.1:$PORT/healthz" >/dev/null; then echo "ready on :$PORT (pid $(cat "$PID"))"; return 0; fi
    if ! kill -0 "$(cat "$PID")" 2>/dev/null; then echo "failed:"; tail -20 "$LOG"; return 1; fi; done
  echo "timeout"; tail -5 "$LOG"; return 1
}
case "${1:-}" in start) start;; stop) stop;; restart) stop; start;; status) curl -s -m 3 "http://127.0.0.1:$PORT/v1/engines" || echo "down";; *) echo "usage: $0 start|stop|restart|status [GPU] [PORT]"; exit 1;; esac
