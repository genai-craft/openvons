/* ブラウザ内の kana 入力 + 振り分け (WebAssembly: komimi の C エンジン + openvons/voice/wasm/ov_route.c)。
 * UI スレッドを止めないようにワーカーで動かす。1 ワーカー = 1 モデル。
 *
 *   {type:'init', wasm, model, vocab}        → {type:'ready', ms, mb}  (取得の進みは {type:'progress', loaded, total})
 *   {type:'commands', key, blob, params}     → {type:'commands', key, n}
 *   {type:'route', req, pcm (Float32 16k)}   → {type:'result', req, key, out (判断の JSON)}
 */
let M = null, h = 0, curKey = null;

async function fetchBytes(url, onProgress) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  const total = +(r.headers.get('content-length') || 0);
  if (!r.body || !total) return new Uint8Array(await r.arrayBuffer());
  const reader = r.body.getReader(); const buf = new Uint8Array(total); let off = 0;
  for (;;) {
    const { done, value } = await reader.read(); if (done) break;
    buf.set(value, off); off += value.length; onProgress && onProgress(off, total);
  }
  return buf.subarray(0, off);
}

self.onmessage = async (e) => {
  const d = e.data;
  try {
    if (d.type === 'init') {
      const t0 = performance.now();
      // WebAssembly SIMD が使えるか (v128 を返すだけの小さなモジュールが検証を通るか)。無ければ generic 版を読む
      const simd = WebAssembly.validate(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11]));
      const js = simd ? d.wasm : d.wasm.replace('ovkana.js', 'ovkana_nosimd.js');
      importScripts(js);
      M = await createOVKana({ locateFile: (f) => js.replace(/ovkana(_nosimd)?\.js(\?.*)?$/, f + '$2') });
      const model = await fetchBytes(d.model, (loaded, total) => postMessage({ type: 'progress', loaded, total }));
      const p = M._malloc(model.length); M.HEAPU8.set(model, p);
      h = M._ovk_new(p, model.length); M._free(p);
      if (!h) throw new Error('model load failed');
      const vocab = await (await fetch(d.vocab)).json();
      const sc = new Float32Array(vocab.scores); const q = M._malloc(sc.length * 4); M.HEAPF32.set(sc, q >> 2);
      M._ovk_set_vocab_scores(h, q, sc.length); M._free(q);
      postMessage({ type: 'ready', ms: Math.round(performance.now() - t0), mb: +(model.length / 1e6).toFixed(1), simd });
    } else if (d.type === 'commands') {
      const blob = new Uint8Array(d.blob); const p = M._malloc(blob.length); M.HEAPU8.set(blob, p);
      const n = M._ovk_set_commands(h, p, blob.length); M._free(p);
      const pr = new Float64Array(d.params); const q = M._malloc(pr.length * 8); M.HEAPF64.set(pr, q >> 3);
      M._ovk_set_params(h, q, pr.length); M._free(q);
      curKey = d.key;
      postMessage({ type: 'commands', key: d.key, n });
    } else if (d.type === 'route') {
      const pcm = d.pcm; const p = M._malloc(pcm.length * 4); M.HEAPF32.set(pcm, p >> 2);
      const t0 = performance.now();
      const out = JSON.parse(M.UTF8ToString(M._ovk_route(h, p, pcm.length)));
      M._free(p);
      out.wall_ms = +(performance.now() - t0).toFixed(1);
      postMessage({ type: 'result', req: d.req, key: curKey, out });
    }
  } catch (err) {
    postMessage({ type: 'error', req: d.req, msg: String(err && err.message || err) });
  }
};
