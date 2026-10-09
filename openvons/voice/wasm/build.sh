#!/bin/bash
# 振り分け (ov_route.c) と komimi の C エンジンを 1 つの WebAssembly にする (Emscripten は docker)。
#   KOMIMI_HOME=~/dev/komimi openvons/voice/wasm/build.sh
# 出力: openvons/voice/demo_static/wasm/ovkana.{js,wasm} (デモは /shared/wasm/ で配る)。
# ネイティブ版 (Python との一致試験 scripts/check_route_equivalence.py 用) も同じソースから libovroute.so に作る。
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(cd "$HERE/../../.." && pwd)"
KOMIMI_HOME="${KOMIMI_HOME:-$HOME/dev/komimi}"; K="$KOMIMI_HOME/csrc"
OUT="$ROOT/openvons/voice/demo_static/wasm"; mkdir -p "$OUT"
SRC="km_model.c km_feat.c km_conformer.c km_stream.c"
# ネイティブ (AVX2)
gcc -O2 -std=c11 -D_GNU_SOURCE -fPIC -shared -mavx2 -DKM_KERNEL_AVX2 -I"$K" $(for f in $SRC; do echo "$K/$f"; done) "$HERE/ov_route.c" -o "$HERE/libovroute.so" -lm
# WebAssembly (SIMD128)
docker run --rm -u "$(id -u):$(id -g)" -v "$K":/komimi:ro -v "$HERE":/src:ro -v "$OUT":/out -w /src emscripten/emsdk:latest \
  emcc -O3 -std=c11 -D_GNU_SOURCE -msimd128 -DKM_KERNEL_WASM_SIMD -I /komimi $(for f in $SRC; do echo "/komimi/$f"; done) /src/ov_route.c \
  -s WASM=1 -s ALLOW_MEMORY_GROWTH=1 -s INITIAL_MEMORY=64MB -s MODULARIZE=1 -s EXPORT_NAME=createOVKana -s ENVIRONMENT=web,worker \
  -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","HEAPF32","HEAPF64","HEAPU8","UTF8ToString"]' \
  -s EXPORTED_FUNCTIONS='["_malloc","_free","_ovk_new","_ovk_vocab","_ovk_set_vocab_scores","_ovk_set_params","_ovk_set_commands","_ovk_route","_ovk_route_logprobs","_ovk_transcribe","_ovk_free"]' \
  -o /out/ovkana.js
# SIMD の無いブラウザ用 (内積は generic C)。ワーカーが WebAssembly.validate で SIMD の有無を見て選ぶ
docker run --rm -u "$(id -u):$(id -g)" -v "$K":/komimi:ro -v "$HERE":/src:ro -v "$OUT":/out -w /src emscripten/emsdk:latest \
  emcc -O3 -std=c11 -D_GNU_SOURCE -I /komimi $(for f in $SRC; do echo "/komimi/$f"; done) /src/ov_route.c \
  -s WASM=1 -s ALLOW_MEMORY_GROWTH=1 -s INITIAL_MEMORY=64MB -s MODULARIZE=1 -s EXPORT_NAME=createOVKana -s ENVIRONMENT=web,worker \
  -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","HEAPF32","HEAPF64","HEAPU8","UTF8ToString"]' \
  -s EXPORTED_FUNCTIONS='["_malloc","_free","_ovk_new","_ovk_vocab","_ovk_set_vocab_scores","_ovk_set_params","_ovk_set_commands","_ovk_route","_ovk_route_logprobs","_ovk_transcribe","_ovk_free"]' \
  -o /out/ovkana_nosimd.js
ls -la "$OUT"
