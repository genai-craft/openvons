#!/usr/bin/env bash
# 公開デモ 8 本 (+ 共有 ASR サーバー) の一括 状態確認 / 起動 / 再起動。共有 ASR サーバーは音声デモより先に起動する。
#   scripts/serve_all.sh status     現在の生死 (ローカル + 公開URL)
#   scripts/serve_all.sh start      落ちているものだけ起動
#   scripts/serve_all.sh restart    全部入れ直す
# pkill -f は自分の shell を殺すので使わない。各デモの pid ファイル方式に任せる。
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"

# name:port:hostname:起動コマンド
SERVICES=(
  "asr:8630::scripts/serve_asr.sh start 0 8630"
  "road_cameras:8600:shirei:JEV_APP=examples.road_cameras.app scripts/serve_demo.sh start 0 8600"
  "stations:8601:eki:JEV_APP=examples.stations.app scripts/serve_demo.sh start 0 8601"
  "vision_attrs:8602:kao:scripts/serve_vision.sh start 1 8602"
  "kasen:8603:kasen:JEV_APP=examples.kasen.app scripts/serve_demo.sh start 0 8603"
  "text_decision:8604::scripts/serve_text.sh start 8604"
  "landing:8605::scripts/serve_landing.sh start 8605"
  "ondevice:8606:ondevice:scripts/serve_ondevice.sh start 2 8606"
  "judge:8607:judge:JUDGE_GPU=4 scripts/serve_judge.sh start 8607"
)

alive() { if [ "$1" = 8630 ]; then curl -sf -o /dev/null -m 5 "http://127.0.0.1:$1/healthz" 2>/dev/null; else curl -sf -o /dev/null -m 5 "http://127.0.0.1:$1/" 2>/dev/null; fi; }

status() {
  printf "%-15s %-6s %-9s %s\n" "デモ" "ポート" "ローカル" "公開URL"
  for s in "${SERVICES[@]}"; do
    IFS=: read -r name port host cmd <<< "$s"
    if alive "$port"; then l="OK"; else l="停止"; fi
    if [ -n "$host" ]; then
      code=$(curl -sS -o /dev/null -m 15 -w "%{http_code}" "https://$host.openvons.com/" 2>/dev/null)
      r="$host.openvons.com $code"
    else r="(非公開)"; fi
    printf "%-15s %-6s %-9s %s\n" "$name" "$port" "$l" "$r"
  done
}

start_missing() {
  for s in "${SERVICES[@]}"; do
    IFS=: read -r name port host cmd <<< "$s"
    if alive "$port"; then printf "%-15s 稼働中\n" "$name"; continue; fi
    printf "%-15s 起動中... " "$name"; eval "$cmd" >/dev/null 2>&1
    if alive "$port"; then echo "OK"; else echo "失敗 (state/$name/logs/server.log を確認)"; fi
  done
}

restart_all() {
  for s in "${SERVICES[@]}"; do
    IFS=: read -r name port host cmd <<< "$s"
    printf "%-15s 再起動中... " "$name"
    eval "${cmd/ start / stop }" >/dev/null 2>&1 || true
    eval "$cmd" >/dev/null 2>&1
    if alive "$port"; then echo "OK"; else echo "失敗"; fi
  done
}

case "${1:-status}" in
  status) status;;
  start) start_missing;;
  restart) restart_all;;
  *) echo "usage: $0 status|start|restart"; exit 1;;
esac
