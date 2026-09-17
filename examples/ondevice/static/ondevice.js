/* 端末内推論: onnxruntime-web (WebGPU / WASM) で ONNX の kana ASR を動かす。
   transformers.js は log-mel (前処理) とトークナイザだけに使い、生成と採点は自前で行う
   (merged decoder の cache 分岐や fp16 の型で詰まるため、KV cache なしの decoder を自分で回す)。 */
import { AutoTokenizer, AutoProcessor, env as tjsEnv } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0';
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.0/dist/ort.all.bundle.min.mjs';

const $ = (s) => document.querySelector(s);
const log = (...a) => { const el = $('#log'); el.textContent = a.join(' ') + '\n' + el.textContent; console.log(...a); };
const S = { enc: null, dec: null, tok: null, proc: null, busy: false, state: 'MAP' };
const PREFIX = ['<|startoftranscript|>', '<|ja|>', '<|transcribe|>', '<|notimestamps|>'];

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.0/dist/';
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;

const cfg = await (await fetch('/api/config')).json();
S.cfg = cfg; S.cal = cfg.calibration; S.name = cfg.models[0] || 'kana-small-2l';
S.sets = await (await fetch('/api/commands?app_name=kasen')).json();
$('#stateSel').innerHTML = Object.keys(S.sets).map(k => `<option>${k}</option>`).join('');
$('#stateSel').onchange = (e) => { S.state = e.target.value; showState(); };
function showState() { const s = S.sets[S.state]; $('#stateName').textContent = S.state; $('#stateInfo').textContent = `${s.n} 通りの言い方 / ${s.description}`; }
showState();
$('#envInfo').textContent = `WebGPU: ${('gpu' in navigator) ? '使えます' : '使えません (WASM で動きます)'} / WASM スレッド: ${ort.env.wasm.numThreads} / 校正 T=${S.cal.temperature.toFixed(2)} β0=${S.cal.none_bias.toFixed(1)} β1=${S.cal.len_bonus.toFixed(1)} γ=${S.cal.residual_penalty.toFixed(1)}`;

/* ---------------- モデル読み込み ---------------- */
$('#loadBtn').onclick = async () => {
  const want = $('#device').value, dtype = $('#dtype').value;
  const ep = want === 'auto' ? (('gpu' in navigator) ? 'webgpu' : 'wasm') : want;
  $('#loadBtn').disabled = true; $('#loadMsg').textContent = `読み込み中 (${ep} / ${dtype})…`;
  const t0 = performance.now();
  try {
    tjsEnv.allowLocalModels = true; tjsEnv.allowRemoteModels = false; tjsEnv.localModelPath = '/model/';
    S.tok = await AutoTokenizer.from_pretrained(S.name);
    S.proc = await AutoProcessor.from_pretrained(S.name);
    const suffix = dtype === 'q8' ? '_quantized' : '';
    const opts = { executionProviders: [ep], graphOptimizationLevel: 'all' };
    const load = async (file) => {
      const url = `/model/${S.name}/onnx/${file}${suffix}.onnx`;
      const buf = await fetchWithProgress(url);
      return ort.InferenceSession.create(buf, opts);
    };
    S.enc = await load('encoder_model');
    S.dec = await load('decoder_model');
    S.prefix = cfg.prefix; S.eot = cfg.eot; S.suppress = cfg.suppress || [];   // 特殊トークンの id はサーバーが解決して渡す
    log('decoder inputs:', JSON.stringify(S.dec.inputNames), 'suppress:', S.suppress.length);
    $('#loadMsg').textContent = `読み込み完了 ${((performance.now() - t0) / 1000).toFixed(1)} 秒 (${ep} / ${dtype})`;
    $('#micBtn').disabled = false; $('#selfTest').disabled = false;
  } catch (e) { $('#loadMsg').textContent = 'エラー: ' + e.message; log('load error', e.message); $('#loadBtn').disabled = false; }
};
async function fetchWithProgress(url) {
  const r = await fetch(url); const total = +(r.headers.get('content-length') || 0);
  if (!r.body || !total) return new Uint8Array(await r.arrayBuffer());
  const reader = r.body.getReader(); const chunks = []; let got = 0;
  for (;;) { const { done, value } = await reader.read(); if (done) break; chunks.push(value); got += value.length;
    $('#dlBar').style.width = (got / total * 100).toFixed(0) + '%'; }
  const out = new Uint8Array(got); let o = 0; for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

/* ---------------- カナと絞り込み ---------------- */
const norm = (s) => s.replace(/[\s、。，．,.!?！？「」『』・~〜()（）\[\]【】"'’‘\-–—:;/…]/g, '')
  .replace(/ヲ/g, 'オ').replace(/ヂ/g, 'ジ').replace(/ヅ/g, 'ズ').replace(/ヰ/g, 'イ').replace(/ヱ/g, 'エ');
function lev(a, b) {
  const m = a.length, n = b.length; if (!m || !n) return Math.max(m, n);
  let prev = Array.from({ length: n + 1 }, (_, j) => j), cur = new Array(n + 1);
  for (let i = 1; i <= m; i++) { cur[0] = i;
    for (let j = 1; j <= n; j++) cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
    [prev, cur] = [cur, prev]; }
  return prev[n];
}
function shortlist(free, hyps, k = 12) {
  const scored = hyps.map((h, i) => ({ i, s: 1 - lev(free, h.k) / Math.max(free.length, h.k.length, 1) }));
  scored.sort((a, b) => b.s - a.s);
  const seen = new Map(), out = [];
  for (const { i } of scored) {
    const key = hyps[i].i + '|' + JSON.stringify(hyps[i].s);
    if ((seen.get(key) || 0) >= 2) continue;
    seen.set(key, (seen.get(key) || 0) + 1); out.push(i);
    if (out.length >= k) break;
  }
  return out;
}

/* ---------------- 推論 (自前) ---------------- */
async function encode(audio) {
  const t0 = performance.now();
  const inputs = await S.proc(audio);
  const f = inputs.input_features;
  const feats = new ort.Tensor('float32', f.data instanceof Float32Array ? f.data : Float32Array.from(f.data), f.dims);
  const out = await S.enc.run({ input_features: feats });
  return { h: out[S.enc.outputNames[0]], ms: performance.now() - t0 };
}
async function greedy(h, maxNew = 24) {
  const t0 = performance.now(); const seq = [...S.prefix];
  for (let step = 0; step < maxNew; step++) {
    const ids = new ort.Tensor('int64', BigInt64Array.from(seq.map(BigInt)), [1, seq.length]);
    const out = await S.dec.run({ input_ids: ids, encoder_hidden_states: h });
    const logits = out[S.dec.outputNames[0]]; const V = logits.dims[2];
    const off = (seq.length - 1) * V; const d = logits.data;
    let best = -1, bv = -Infinity;
    for (let v = 0; v < V; v++) { const x = d[off + v]; if (x > bv && !S.supSet.has(v)) { bv = x; best = v; } }
    if (best === S.eot || best < 0) break;
    seq.push(best);
  }
  const content = seq.slice(S.prefix.length);
  return { tokens: content, kana: norm(S.tok.decode(content, { skip_special_tokens: true }).trim()), ms: performance.now() - t0 };
}
async function scoreAll(h, tokenLists) {
  const B = tokenLists.length; if (!B) return { scores: [], lens: [], ms: 0 };
  const seqs = tokenLists.map(t => [...S.prefix, ...t, S.eot]);
  const L = Math.max(...seqs.map(s => s.length));
  const inLen = L - 1;
  const ids = new BigInt64Array(B * inLen); const masks = [];
  for (let i = 0; i < B; i++) {
    const s = seqs[i], m = [];
    for (let j = 0; j < inLen; j++) { ids[i * inLen + j] = BigInt(j < s.length ? s[j] : S.eot); m.push(j + 1 < s.length && j + 1 >= S.prefix.length); }
    masks.push(m);
  }
  const hb = repeatBatch(h, B);
  const t0 = performance.now();
  const out = await S.dec.run({ input_ids: new ort.Tensor('int64', ids, [B, inLen]), encoder_hidden_states: hb });
  const logits = out[S.dec.outputNames[0]]; const V = logits.dims[2]; const d = logits.data;
  const scores = [], lens = [];
  for (let i = 0; i < B; i++) {
    let tot = 0, n = 0;
    for (let j = 0; j < inLen; j++) {
      if (!masks[i][j]) continue;
      const off = (i * inLen + j) * V;
      let mx = -Infinity; for (let v = 0; v < V; v++) if (d[off + v] > mx) mx = d[off + v];
      let se = 0; for (let v = 0; v < V; v++) se += Math.exp(d[off + v] - mx);
      const tgt = Number(ids[i * inLen + j + 1] ?? BigInt(S.eot));
      tot += (d[off + tgt] - mx) - Math.log(se); n++;
    }
    scores.push(tot); lens.push(n);
  }
  return { scores, lens, ms: performance.now() - t0 };
}
function repeatBatch(h, B) {
  if (B === 1) return h;
  const [, T, D] = h.dims; const src = h.data; const dst = new src.constructor(B * T * D);
  for (let i = 0; i < B; i++) dst.set(src, i * T * D);
  return new ort.Tensor(h.type, dst, [B, T, D]);
}
function calibrate(scores, lens, freeScore, nFree) {
  const { temperature: T, none_bias: b0, len_bonus: b1, residual_penalty: g } = S.cal;
  const z = scores.map((s, i) => (s + b1 * Math.min(lens[i], nFree) - g * Math.max(0, nFree - lens[i])) / T);
  z.push((freeScore - b0) / T);
  const mx = Math.max(...z); const ex = z.map(x => Math.exp(x - mx)); const sum = ex.reduce((a, b) => a + b, 0);
  return ex.map(e => e / sum);
}

/* ---------------- 1 発話 ---------------- */
async function handle(audio) {
  if (S.busy || !S.enc) return; S.busy = true;
  try {
    S.supSet = new Set(S.suppress);
    const hyps = S.sets[S.state].hyps;
    const e = await encode(audio);
    const free = await greedy(e.h);
    const idx = shortlist(free.kana, hyps);
    const cands = idx.map(i => hyps[i]);
    const lists = cands.map(c => S.tok.encode(c.k, { add_special_tokens: false }));
    const freeTok = free.tokens;
    const all = await scoreAll(e.h, [...lists, freeTok]);
    const nFree = freeTok.length + 1;
    const freeScore = all.scores[all.scores.length - 1];
    const probs = calibrate(all.scores.slice(0, -1), all.lens.slice(0, -1), freeScore, nFree);
    const none = probs[probs.length - 1];
    const agg = new Map();
    cands.forEach((c, i) => { const k = c.i + '|' + JSON.stringify(c.s); const cur = agg.get(k) || { c, p: 0, best: -1e9 };
      cur.p += probs[i]; if (all.scores[i] > cur.best) { cur.best = all.scores[i]; cur.c = c; } agg.set(k, cur); });
    const ranked = [...agg.values()].sort((a, b) => b.p - a.p);
    const top = ranked[0];
    const action = !top || none > top.p ? 'none' : top.c.r === 'high' ? (top.p >= .4 ? 'confirm' : 'reject')
      : top.p >= .85 ? 'execute' : top.p >= .4 ? 'confirm' : 'reject';
    render({ free, encMs: e.ms, scoreMs: all.ms, ranked, none, action, audio });
  } catch (err) { log('error', err.message); $('#speech').textContent = 'エラー: ' + err.message; }
  finally { S.busy = false; }
}
function render(r) {
  $('#freeKana').textContent = r.free.kana;
  const dec = $('#decision'); dec.className = 'decision ' + r.action;
  dec.textContent = `${{ execute: '実行', confirm: '確認', reject: '棄却', none: '該当なし' }[r.action]}　${r.ranked[0] ? r.ranked[0].c.t : ''}`;
  $('#speech').textContent = r.ranked[0] && r.action !== 'none' ? r.ranked[0].c.t : '(システム宛ではないと判断)';
  $('#nbest').innerHTML = r.ranked.slice(0, 5).map(x => bar(x.c.t, x.p, '')).concat([bar('該当なし', r.none, 'none')]).join('');
  const tot = r.encMs + r.free.ms + r.scoreMs;
  $('#times').innerHTML = [['encoder + 前処理', r.encMs], [`自由認識 (${r.free.tokens.length} トークン)`, r.free.ms], ['候補採点', r.scoreMs], ['合計 (端末内)', tot]]
    .map(([k, v]) => `<tr><td>${k}</td><td>${v.toFixed(0)} ms</td></tr>`).join('');
  log(`[${S.state}] ${r.free.kana} -> ${r.action} ${r.ranked[0] ? r.ranked[0].c.t : '-'} p=${r.ranked[0] ? r.ranked[0].p.toFixed(3) : 0} (${tot.toFixed(0)}ms)`);
  if ($('#cmpServer').checked) compareServer(r.audio, tot);
}
function bar(text, p, cls) { return `<div class="row2"><div class="bar ${cls}"><i style="width:${(p * 100).toFixed(1)}%"></i><span>${text}</span></div><div>${(p * 100).toFixed(1)}%</div></div>`; }
async function compareServer(audio, localMs) {
  $('#serverCmp').textContent = 'サーバーに問い合わせ中…';
  const t0 = performance.now();
  const r = await (await fetch('/api/server_decide', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ audio_base64: encodeWav(audio, 16000), state: S.state }) })).json();
  const ms = performance.now() - t0;
  if (r.error) { $('#serverCmp').textContent = 'サーバー比較: ' + r.error; return; }
  $('#serverCmp').innerHTML = `サーバー (kana-whisper 809M): <b>${r.free_kana}</b> → ${r.action} ${r.top ? r.top.text : '-'} p=${r.top ? r.top.prob : 0}　往復 ${ms.toFixed(0)}ms (端末内 ${localMs.toFixed(0)}ms)`;
}
function encodeWav(f32, sr) {
  const n = f32.length; const buf = new ArrayBuffer(44 + n * 2); const v = new DataView(buf);
  const w = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  w(0, 'RIFF'); v.setUint32(4, 36 + n * 2, true); w(8, 'WAVEfmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true);
  v.setUint16(22, 1, true); v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  w(36, 'data'); v.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) v.setInt16(44 + i * 2, Math.max(-32768, Math.min(32767, f32[i] * 32768)), true);
  let b = ''; const u8 = new Uint8Array(buf); const CH = 8192;
  for (let i = 0; i < u8.length; i += CH) b += String.fromCharCode(...u8.subarray(i, i + CH));
  return btoa(b);
}

/* ---------------- マイク ---------------- */
let media = null, chunks = [];
async function startRec() {
  if (!media) {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
    const ac = new AudioContext({ sampleRate: 16000 });
    const src = ac.createMediaStreamSource(stream);
    const proc = ac.createScriptProcessor(4096, 1, 1);
    proc.onaudioprocess = (e) => {
      const x = e.inputBuffer.getChannelData(0);
      let peak = 0; for (let i = 0; i < x.length; i++) peak = Math.max(peak, Math.abs(x[i]));
      $('#vuBar').style.width = Math.min(100, peak * 300) + '%';
      if (media && media.on) chunks.push(new Float32Array(x));
    };
    src.connect(proc); proc.connect(ac.destination);
    media = { stream, proc, ac, on: false };
    log('マイク開始 (sampleRate', ac.sampleRate, ')');
  }
  chunks = []; media.on = true; $('#vadChip').textContent = '録音中'; $('#vadChip').classList.add('on');
}
async function stopRec() {
  if (!media || !media.on) return;
  media.on = false; $('#vadChip').textContent = '処理中'; $('#vadChip').classList.remove('on');
  const n = chunks.reduce((a, c) => a + c.length, 0);
  const audio = new Float32Array(Math.max(n, 16000)); let o = 0;
  for (const c of chunks) { audio.set(c, o); o += c.length; }
  await handle(audio);
  $('#vadChip').textContent = '待機';
}
const mb = $('#micBtn');
mb.addEventListener('pointerdown', (e) => { e.preventDefault(); startRec(); });
mb.addEventListener('pointerup', stopRec);
mb.addEventListener('pointercancel', stopRec);
mb.addEventListener('pointerleave', () => { if (media && media.on) stopRec(); });
window.__handle = handle;     // 自動検証から 1 発話を流すため
$('#selfTest').onclick = async () => {
  $('#speech').textContent = 'セルフテスト中…';
  const r = await fetch('/api/sample_wav'); const buf = await r.arrayBuffer();
  const ac = new AudioContext({ sampleRate: 16000 }); const ab = await ac.decodeAudioData(buf);
  await handle(ab.getChannelData(0));
};
log('準備完了。モデルを読み込んでください。');
