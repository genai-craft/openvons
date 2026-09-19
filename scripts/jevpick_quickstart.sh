#!/usr/bin/env bash
# JevPick を 1 GPU で一通り動かす (Tool Call、Qwen3-4B)。所要 20〜40 分、ディスク ~10GB (N=300 のとき)。
#   scripts/jevpick_quickstart.sh [GPU] [N_TRAIN] [N_TEST]
# 出力: $JEVPICK_DATA (既定 state/jevpick) に trace / hidden / 候補 / 学習済み JevPick、
#       experiments/jevpick/*/ に oracle 表・学習結果・runtime 比較 (json/md)。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
GPU="${1:-0}"; N_TRAIN="${2:-300}"; N_TEST="${3:-100}"
PY="$ROOT/.venv/bin/python"; export CUDA_VISIBLE_DEVICES="$GPU"
D="$($PY -c 'from openvons.jevpick.paths import DATA; print(DATA)')"
MODEL="${JEVPICK_MODEL:-Qwen/Qwen3-4B}"; DRAFT="${JEVPICK_DRAFT:-z-lab/Qwen3-4B-DFlash-b16}"; TAG="${MODEL##*/}"
echo "data dir: $D   model: $MODEL   draft: $DRAFT"

echo "[1/6] prompt (glaive tool call: train $N_TRAIN / test $N_TEST)"
$PY openvons/jevpick/data/build_prompts_v2.py --toolcall-train "$N_TRAIN" --toolcall-test "$N_TEST" --no-python
echo "[2/6] greedy trace"
$PY openvons/jevpick/data/collect.py --model "$MODEL" --no-think --domain toolcall --prompts "$D/prompts_v2.jsonl" --out "$D/traces_v2_$TAG.jsonl" --batch 24 --max-new 128
echo "[3/6] hidden state + DFlash 候補"
$PY openvons/jevpick/data/extract.py --model "$MODEL" --draft "$DRAFT" --traces "$D/traces_v2_$TAG.jsonl" --out "$D/extract_v2_${TAG}_0"
echo "[4/6] 候補メニューの上限値 (oracle)"
$PY experiments/jevpick/phase1_oracle/run_v2.py --model "$MODEL" --traces "$D/traces_v2_$TAG.jsonl" --extract "$D/extract_v2_${TAG}_0" --domains toolcall --workers 8 --out "$D/oracle_v2_$TAG.pkl"
$PY experiments/jevpick/phase1_oracle/report_v2.py --pkl "$D/oracle_v2_$TAG.pkl" --bench experiments/jevpick/phase1_oracle/bench_verify_reference.json --out "experiments/jevpick/phase1_oracle/results_quickstart_$TAG.md"
echo "[5/6] JevPick の学習と評価 (有限候補 / DFlash 併用)"
for SRC in finite union; do
  $PY experiments/jevpick/phase2_scorer/train.py --pkl "$D/oracle_v2_$TAG.pkl" --extract "$D/extract_v2_${TAG}_0" --traces "$D/traces_v2_$TAG.jsonl" --model "$MODEL" \
     --bench experiments/jevpick/phase1_oracle/bench_verify_reference.json --domain toolcall --source $SRC --L 8 --layer 3 --encoder pool --epochs 3 \
     --save "$D/jevpick_toolcall_${SRC}_L8.pt" --out "experiments/jevpick/phase2_scorer/result_quickstart_${TAG}_${SRC}.json"
done
echo "[6/6] 実測 (同一 verifier: 何もしない / JevPick / DFlash / 併用 / hybrid)"
$PY experiments/jevpick/phase4_runtime/runtime_v2.py --model "$MODEL" --draft "$DRAFT" --prompts "$D/prompts_v2.jsonl" --traces "$D/traces_v2_$TAG.jsonl" \
   --domain toolcall --n 20 --L 8 --modes baseline,prior,scorer,dflash,union,hybrid --scorer "$D/jevpick_toolcall_union_L8.pt" --out "experiments/jevpick/phase4_runtime/runtime_quickstart_$TAG.json"
echo "done. 結果: experiments/jevpick/phase1_oracle/results_quickstart_$TAG.md, phase2_scorer/result_quickstart_*.json, phase4_runtime/runtime_quickstart_$TAG.json"
