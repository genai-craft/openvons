/* 端末内推論: transformers.js (WebGPU/WASM) で ONNX の kana ASR を動かし、
   自由認識 → 候補の絞り込み → 強制トークン採点 → 校正 → 判断 までをブラウザで行う */
import { AutoTokenizer, AutoProcessor, WhisperForConditionalGeneration, Tensor, env }
  from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0';

const $ = (s) => document.querySelector(s);
const log = (...a) => { const el = $('#log'); el.textContent = a.join(' ') + '\n' + el.textContent; console.log(...a); };
const S = { model: null, tok: null, proc: null, cfg: null, sets: null, state: 'MAP', rec: null, busy: false, cal: null };
const PREFIX_TOKENS = ['<|startoftranscript|>', '<|ja|>', '<|transcribe|>', '<|notimestamps|>'];

/* ---------------- 設定とコマンド集合 ---------------- */
const cfg = await (await fetch('/api/config')).json();
S.cfg = cfg; S.cal = cfg.calibration;
S.sets = await (await fetch('/api/commands?app_name=kasen')).json();
$('#stateSel').innerHTML = Object.keys(S.sets).map(k => `<option>${k}</option>`).join('');
$('#stateSel').onchange = (e) => { S.state = e.target.value; showState(); };
function showState() { const s = S.sets[S.state]; $('#stateName').textContent = S.state; $('#stateInfo').textContent = `${s.n} 通りの言い方 / ${s.description}`; }
showState();
$('#envInfo').textContent = `WebGPU: ${('gpu' in navigator) ? '使えます' : '使えません (WASM で動きます)'} / モデル: ${cfg.models.join(', ')} / 校正 T=${S.cal.temperature.toFixed(2)} β0=${S.cal.none_bias.toFixed(1)} β1=${S.cal.len_bonus.toFixed(1)} γ=${S.cal.residual_penalty.toFixed(1)}`;

/* ---------------- モデル読み込み ---------------- */
$('#loadBtn').onclick = async () => {
  const id = '/model/' + (S.cfg.models[0] || 'kana-small-2l');
  const want = $('#device').value, dtype = $('#dtype').value;
  const device = want === 'auto' ? (('gpu' in navigator) ? 'webgpu' : 'wasm') : want;
  $('#loadBtn').disabled = true; $('#loadMsg').textContent = `読み込み中 (${device} / ${dtype})…`;
  const t0 = performance.now();
  try {
    env.allowLocalModels = true; env.allowRemoteModels = false; env.localModelPath = '/model/';
    const name = (S.cfg.models[0] || 'kana-small-2l');
    S.tok = await AutoTokenizer.from_pretrained(name);
    S.proc = await AutoProcessor.from_pretrained(name);
    S.model = await WhisperForConditionalGeneration.from_pretrained(name, {
      device, dtype: { encoder_model: dtype, decoder_model_merged: dtype },
      progress_callback: (p) => { if (p.status === 'progress' && p.total) $('#dlBar').style.width = (p.loaded / p.total * 100).toFixed(0) + '%'; },
    });
    S.sessionNames = Object.keys(S.model.sessions || {});
    log('sessions:', JSON.stringify(S.sessionNames));
    const dec = S.model.sessions[S.sessionNames.find(n => n.includes('decoder'))];
    S.decNames = dec.inputNames || (dec.handler && dec.handler.inputNames) || [];
    log('decoder inputs:', JSON.stringify(S.decNames.slice(0, 6)), '…', S.decNames.length);
    S.prefix = PREFIX_TOKENS.map(t => S.tok.model.tokens_to_ids.get(t));
    S.eot = S.tok.model.tokens_to_ids.get('<|endoftext|>');
    S.suppress = await buildSuppress();
    $('#loadMsg').textContent = `読み込み完了 ${(performance.now() - t0) / 1000 | 0} 秒 (${device} / ${dtype})`;
    $('#micBtn').disabled = false;
    $('#dlBar').style.width = '100%';
  } catch (e) { $('#loadMsg').textContent = 'エラー: ' + e.message; log('load error', e.message); $('#loadBtn').disabled = false; }
};
async function buildSuppress() {
  // 非カナのトークンを抑制 (学習時と同じ)。distill_info.json に入っていればそれを使う
  try {
    const j = await (await fetch(`/model/${S.cfg.models[0]}/distill_info.json`)).json();
    if (j.suppress_tokens_kana_only) { log('suppress from distill_info:', j.suppress_tokens_kana_only.length); return j.suppress_tokens_kana_only; }
  } catch (e) { }
  return [];
}

/* ---------------- カナと候補の絞り込み (kana.py の JS 版) ---------------- */
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
function shortlist(free, hyps, k = 16) {
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

/* ---------------- 推論 ---------------- */
async function encode(audio) {
  const inputs = await S.proc(audio);
  const encName = S.sessionNames.find(n => !n.includes('decoder'));
  const t0 = performance.now();
  const out = await S.model.sessions[encName].run({ input_features: inputs.input_features });
  return { h: out.last_hidden_state ?? Object.values(out)[0], ms: performance.now() - t0, inputs };
}
async function freeTranscribe(inputs) {
  const t0 = performance.now();
  const ids = await S.model.generate({ ...inputs, max_new_tokens: 64, num_beams: 1, do_sample: false,
    language: 'ja', task: 'transcribe', ...(S.suppress.length ? { suppress_tokens: S.suppress } : {}) });
  const seq = Array.from(ids.data ?? ids[0].data ?? []).map(Number);
  const content = seq.filter(t => t < S.eot);
  return { kana: norm(S.tok.decode(content, { skip_special_tokens: true }).trim()), tokens: content, ms: performance.now() - t0 };
}
async function scoreCandidates(h, tokenLists) {
  /* 候補を 1 回の forward でまとめて採点 (KV cache なし)。merged decoder の cache 分岐は false 固定 */
  const B = tokenLists.length; if (!B) return [];
  const seqs = tokenLists.map(t => [...S.prefix, ...t, S.eot]);
  const L = Math.max(...seqs.map(s => s.length));
  const ids = new BigInt64Array(B * (L - 1)); const maskArr = [];
  for (let i = 0; i < B; i++) {
    const s = seqs[i]; const m = [];
    for (let j = 0; j < L - 1; j++) { const v = j < s.length ? s[j] : S.eot; ids[i * (L - 1) + j] = BigInt(v); m.push(j + 1 < s.length && j + 1 >= S.prefix.length); }
    maskArr.push(m);
  }
  const dec = S.model.sessions[S.sessionNames.find(n => n.includes('decoder'))];
  const feeds = {
    input_ids: new Tensor('int64', ids, [B, L - 1]),
    encoder_hidden_states: repeatBatch(h, B),
  };
  for (const n of S.decNames) {
    if (n === 'use_cache_branch') feeds[n] = new Tensor('bool', new Uint8Array([0]), [1]);
    else if (n.startsWith('past_key_values')) {
      const isDec = n.includes('.decoder.');
      const dims = isDec ? [B, 12, 0, 64] : [B, 12, 1500, 64];
      feeds[n] = new Tensor('float32', new Float32Array(dims.reduce((a, b) => a * b, 1)), dims);
    }
  }
  const t0 = performance.now();
  const out = await dec.run(feeds);
  const logits = out.logits; const V = logits.dims[2];
  const data = logits.data;
  const scores = [], lens = [];
  for (let i = 0; i < B; i++) {
    let tot = 0, n = 0;
    for (let j = 0; j < L - 1; j++) {
      if (!maskArr[i][j]) continue;
      const off = (i * (L - 1) + j) * V;
      let mx = -Infinity; for (let v = 0; v < V; v++) { const x = data[off + v]; if (x > mx) mx = x; }
      let se = 0; for (let v = 0; v < V; v++) se += Math.exp(data[off + v] - mx);
      const tgt = Number(ids[i * (L - 1) + j + 1] ?? BigInt(S.eot));
      tot += (data[off + tgt] - mx) - Math.log(se); n++;
    }
    scores.push(tot); lens.push(n);
  }
  return { scores, lens, ms: performance.now() - t0 };
}
function repeatBatch(h, B) {
  if (B === 1) return h;
  const [_, T, D] = h.dims; const src = h.data; const dst = new src.constructor(B * T * D);
  for (let i = 0; i < B; i++) dst.set(src, i * T * D);
  return new Tensor(h.type, dst, [B, T, D]);
}
function calibrate(scores, lens, freeScore, nFree) {
  const { temperature: T, none_bias: b0, len_bonus: b1, residual_penalty: g } = S.cal;
  const z = scores.map((s, i) => (s + b1 * Math.min(lens[i], nFree) - g * Math.max(0, nFree - lens[i])) / T);
  z.push((freeScore - b0) / T);
  const mx = Math.max(...z); const ex = z.map(x => Math.exp(x - mx)); const sum = ex.reduce((a, b) => a + b, 0);
  return ex.map(e => e / sum);
}

/* ---------------- 1 発話の処理 ---------------- */
async function handle(audio) {
  if (S.busy || !S.model) return; S.busy = true;
  try {
    const hyps = S.sets[S.state].hyps;
    const { h, ms: encMs, inputs } = await encode(audio);
    const free = await freeTranscribe(inputs);
    const idx = shortlist(free.kana, hyps);
    const cands = idx.map(i => hyps[i]);
    const tokenLists = cands.map(c => S.tok.encode(c.k, { add_special_tokens: false }));
    const freeTok = S.tok.encode(free.kana, { add_special_tokens: false });
    const all = await scoreCandidates(h, [...tokenLists, freeTok]);
    const nFree = freeTok.length + 1;
    const freeScore = all.scores[all.scores.length - 1];
    const probs = calibrate(all.scores.slice(0, -1), all.lens.slice(0, -1), freeScore, nFree);
    const none = probs[probs.length - 1];
    const agg = new Map();
    cands.forEach((c, i) => { const key = c.i + '|' + JSON.stringify(c.s); const cur = agg.get(key) || { c, p: 0 }; cur.p += probs[i]; if (!cur.best || all.scores[i] > cur.best) { cur.best = all.scores[i]; cur.c = c; } agg.set(key, cur); });
    const ranked = [...agg.values()].sort((a, b) => b.p - a.p);
    const top = ranked[0];
    const action = !top || none > top.p ? 'none' : top.c.r === 'high' ? (top.p >= .4 ? 'confirm' : 'reject')
      : top.p >= .85 ? 'execute' : top.p >= .4 ? 'confirm' : 'reject';
    render({ free, encMs, scoreMs: all.ms, ranked, none, action, audio });
  } catch (e) { log('error', e.message); $('#speech').textContent = 'エラー: ' + e.message; }
  finally { S.busy = false; }
}
function render(r) {
  $('#freeKana').textContent = r.free.kana;
  const dec = $('#decision'); dec.className = 'decision ' + r.action;
  const label = { execute: '実行', confirm: '確認', reject: '棄却', none: '該当なし' }[r.action];
  dec.textContent = `${label}　${r.ranked[0] ? r.ranked[0].c.t : ''}`;
  $('#speech').textContent = r.ranked[0] && r.action !== 'none' ? r.ranked[0].c.t : '(システム宛ではないと判断)';
  $('#nbest').innerHTML = r.ranked.slice(0, 5).map(x => bar(x.c.t + (x.c.i !== 'select_camera' ? ` (${x.c.i})` : ''), x.p, ''))
    .concat([bar('該当なし', r.none, 'none')]).join('');
  const tot = r.encMs + r.free.ms + r.scoreMs;
  $('#times').innerHTML = [['encoder', r.encMs], ['自由認識', r.free.ms], ['候補採点', r.scoreMs], ['合計 (端末内)', tot]]
    .map(([k, v]) => `<tr><td>${k}</td><td>${v.toFixed(0)} ms</td></tr>`).join('');
  log(`[${S.state}] ${r.free.kana} -> ${r.action} ${r.ranked[0] ? r.ranked[0].c.t : '-'} p=${r.ranked[0] ? r.ranked[0].p.toFixed(3) : 0} (${tot.toFixed(0)}ms)`);
  if ($('#cmpServer').checked) compareServer(r.audio, tot);
}
function bar(text, p, cls) { return `<div class="row2"><div class="bar ${cls}"><i style="width:${(p * 100).toFixed(1)}%"></i><span>${text}</span></div><div>${(p * 100).toFixed(1)}%</div></div>`; }
async function compareServer(audio, localMs) {
  $('#serverCmp').textContent = 'サーバーに問い合わせ中…';
  const wav = encodeWav(audio, 16000);
  const t0 = performance.now();
  const r = await (await fetch('/api/server_decide', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ audio_base64: wav, state: S.state }) })).json();
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
  let b = ''; const u8 = new Uint8Array(buf); for (let i = 0; i < u8.length; i++) b += String.fromCharCode(u8[i]);
  return btoa(b);
}

/* ---------------- マイク (押している間だけ録音) ---------------- */
let media = null, chunks = [], ac = null;
async function startRec() {
  if (!media) {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
    ac = new AudioContext({ sampleRate: 16000 });
    const src = ac.createMediaStreamSource(stream);
    const proc = ac.createScriptProcessor(4096, 1, 1);
    proc.onaudioprocess = (e) => {
      const x = e.inputBuffer.getChannelData(0);
      let peak = 0; for (let i = 0; i < x.length; i++) peak = Math.max(peak, Math.abs(x[i]));
      $('#vuBar').style.width = Math.min(100, peak * 300) + '%';
      if (media && media.on) chunks.push(new Float32Array(x));
    };
    src.connect(proc); proc.connect(ac.destination);
    media = { stream, proc, on: false };
  }
  chunks = []; media.on = true; $('#vadChip').textContent = '録音中'; $('#vadChip').classList.add('on');
}
async function stopRec() {
  if (!media || !media.on) return;
  media.on = false; $('#vadChip').textContent = '処理中'; $('#vadChip').classList.remove('on');
  const n = chunks.reduce((a, c) => a + c.length, 0); const audio = new Float32Array(Math.max(n, 16000));
  let o = 0; for (const c of chunks) { audio.set(c, o); o += c.length; }
  await handle(audio);
  $('#vadChip').textContent = '待機';
}
const mb = $('#micBtn');
mb.addEventListener('pointerdown', (e) => { e.preventDefault(); startRec(); });
mb.addEventListener('pointerup', stopRec); mb.addEventListener('pointerleave', () => { if (media && media.on) stopRec(); });
log('ready. モデルを読み込んでください。');
