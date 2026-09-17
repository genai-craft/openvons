#!/usr/bin/env bash
# デモサーバーの起動/停止。pid ファイルで管理する (pkill -f / ps|grep は自分の shell を殺す事故があるため使わない)
#   scripts/serve_demo.sh start [GPU] [PORT]
#   scripts/serve_demo.sh stop
#   scripts/serve_demo.sh restart [GPU] [PORT]
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APPNAME=$(basename "$(echo "${JEV_APP:-examples.road_cameras.app}" | tr . /)" ); APPNAME=$(basename "$(dirname "$(echo "${JEV_APP:-examples.road_cameras.app}" | tr . /)")"); STATE=${JEV_STATE_DIR:-${OPENVONS_STATE:-$ROOT/state}/$APPNAME}
PID=$STATE/server.pid
LOG=$STATE/logs/server.log
GPU="${2:-2}"; PORT="${3:-8600}"
TTS_URL="${JEV_TTS_URL:-voicevox://127.0.0.1:50021}"
mkdir -p "$STATE/logs"
stop() { if [ -f "$PID" ] && kill -0 "$(cat "$PID")" 2>/dev/null; then kill "$(cat "$PID")"; sleep 2; fi; rm -f "$PID"; }
start() {
  cd "$ROOT"
  HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" JEV_STATE_DIR="$STATE" CUDA_VISIBLE_DEVICES="$GPU" nohup "$ROOT/.venv/bin/python" -m openvons.voice.demo_server --app "${JEV_APP:-examples.road_cameras.app}" --port "$PORT" --tts "$TTS_URL" ${JEV_SSL_DIR:+--ssl-dir "$JEV_SSL_DIR"} >"$LOG" 2>&1 &
  echo $! > "$PID"
  for i in $(seq 1 60); do sleep 2; if grep -q "Uvicorn running" "$LOG" 2>/dev/null; then echo "ready on :$PORT (pid $(cat "$PID"))"; return 0; fi; if ! kill -0 "$(cat "$PID")" 2>/dev/null; then echo "failed:"; tail -20 "$LOG"; return 1; fi; done
  echo "timeout"; tail -5 "$LOG"; return 1
}
case "${1:-}" in start) start;; stop) stop;; restart) stop; start;; *) echo "usage: $0 start|stop|restart [GPU] [PORT]"; exit 1;; esac
