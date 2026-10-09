/* kana 入力と振り分けの切り替え (全音声デモ共通)。
 *
 *   カナ入力: kana-whisper (サーバー GPU) / komimi v12・v12a・v12m・v12s
 *   振り分け: サーバー (Python、従来どおり音声をサーバーへ送る)
 *             ブラウザ (WebAssembly: komimi の C エンジンで音声 → CTC 行列、ov_route.c で絞り込み・採点・校正・判断)
 *             → 結果 (仮説の index と確率) だけをサーバーへ送り、状態機械を進める。音声はブラウザから出ない
 *
 * デモ側の使い方 (指令・駅・河川):
 *   OVKana.mount('#kanaCtl', { onMessage, session: () => state.session });   // 画面に切り替えを出す
 *   OVKana.attach(ws)                     // WebSocket を作るたびに
 *   if (OVKana.onMessage(m)) return;      // onMessage の先頭で (hello / state / result を覗き、engine / stale は引き受ける)
 *   OVKana.feed(int16, ws)                // マイクの 16 kHz PCM16 (従来の ws.send(out.buffer) の代わり)
 *   if (OVKana.isLocal()) return OVKana.say(text, snr)   // テキスト試験もブラウザで振り分ける
 */
window.OVKana = (() => {
  const LS = 'ovkana_cfg_v1';
  const V = (document.currentScript && (document.currentScript.src.split('?v=')[1] || '')) || String(Date.now());
  let cfg = { server: 'kana-whisper', route: 'server', browser: 'komimi-v12m' };
  try { Object.assign(cfg, JSON.parse(localStorage.getItem(LS) || '{}')); } catch (_) { /* 既定のまま */ }
  const save = () => { try { localStorage.setItem(LS, JSON.stringify(cfg)); } catch (_) { /* 保存できなくても動く */ } };

  let opts = { onMessage: () => {}, session: () => '' };
  let ws = null, root = null, kconf = null, engines = [], serverEngine = null, calibrated = null;
  let worker = null, workerModel = null, workerReady = null, workerResolvers = {}, reqSeq = 0;
  let cmdKey = null, cmdLoading = null, stateSig = '';
  let vad = null, vadLoading = null, busy = false, pending = [];
  let lastNote = '';

  const $ = (sel) => root && root.querySelector(sel);
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  /* ------------------------------------------------------------ 画面 */
  function mount(el, o) {
    opts = { ...opts, ...o };
    root = typeof el === 'string' ? document.querySelector(el) : el;
    if (!root) return;
    root.classList.add('kanactl');
    if (!document.getElementById('kanactl-css')) {
      const st = document.createElement('style'); st.id = 'kanactl-css';
      st.textContent = '.kanactl{display:flex;flex-wrap:wrap;gap:6px 12px;align-items:center;font-size:12px;margin:6px 0 2px}'
        + '.kanactl label{display:flex;gap:4px;align-items:center;color:var(--muted,#8b98a5)}'
        + '.kanactl select{padding:3px 6px;font-size:12px;max-width:230px}'
        + '.kanactl .kstat{flex-basis:100%;font-size:11px;color:var(--muted,#8b98a5)}';
      document.head.appendChild(st);
    }
    root.innerHTML = `
      <label>カナ入力 <select data-k="engine"></select></label>
      <label>振り分け <select data-k="route">
        <option value="server">サーバー (Python)</option>
        <option value="wasm">ブラウザ (WebAssembly)</option></select></label>
      <span class="kstat muted" data-k="stat"></span>`;
    $('[data-k=route]').value = cfg.route;
    $('[data-k=route]').onchange = (e) => { cfg.route = e.target.value; save(); render(); if (cfg.route === 'wasm') ensureWorker(); syncServerEngine(); };
    $('[data-k=engine]').onchange = (e) => {
      if (cfg.route === 'wasm') { cfg.browser = e.target.value; workerReady = null; ensureWorker(); } else cfg.server = e.target.value;
      save(); syncServerEngine(); render();
    };
    fetch('/api/kana/config').then((r) => r.json()).then((c) => { kconf = c; engines = c.engines; render(); if (cfg.route === 'wasm') ensureWorker(); }).catch(() => {});
    render();
  }

  function render() {
    if (!root) return;
    const sel = $('[data-k=engine]'); const list = engines.length ? engines : [{ id: 'kana-whisper', label: 'kana-whisper', server: true }];
    const wasm = cfg.route === 'wasm';
    sel.innerHTML = list.map((e) => {
      const ok = wasm ? !!e.browser_model : e.server !== false;
      const lab = wasm ? (e.browser_model ? `${e.label}・${e.mb} MB` : `${e.label} (ブラウザ不可)`) : e.label;
      return `<option value="${e.id}" ${ok ? '' : 'disabled'}>${esc(lab)}</option>`;
    }).join('');
    sel.value = wasm ? cfg.browser : cfg.server;
    if (wasm && sel.value !== cfg.browser) { cfg.browser = 'komimi-v12m'; sel.value = cfg.browser; }
    $('[data-k=route]').value = cfg.route;
    stat();
  }
  function stat(extra) {
    if (!root) return;
    if (extra !== undefined) lastNote = extra;
    const el = $('[data-k=stat]');
    const cal = calibrated === null ? '' : (calibrated ? '校正: 事前学習済み' : '校正: 既定値 (この範囲は未学習)');
    if (cfg.route === 'wasm') {
      const st = workerReady === true ? 'ブラウザで準備完了' : (workerReady && workerReady.msg) || '準備中…';
      el.textContent = [st, lastNote].filter(Boolean).join(' / ');
    } else {
      el.textContent = [kconf && kconf.asr === 'shared' ? '共有 ASR サーバー' : 'このサーバー', cal, lastNote].filter(Boolean).join(' / ');
    }
  }

  /* ------------------------------------------------------------ WebSocket との橋渡し */
  function attach(w) { ws = w; cmdKey = null; }
  function sendJSON(o) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(o)); }
  function syncServerEngine() { sendJSON({ type: 'engine', engine: cfg.route === 'wasm' ? cfg.browser : cfg.server }); }

  function onMessage(m) {
    if (m.type === 'hello') {
      if (m.engines) engines = m.engines;
      serverEngine = m.engine; cmdKey = null; stateSig = '';
      setTimeout(syncServerEngine, 0); render(); return false;
    }
    if (m.type === 'engine') { serverEngine = m.engine; calibrated = m.calibrated; stat(); return true; }
    if (m.type === 'stale') { cmdKey = null; return true; }
    if (m.type === 'state' || m.type === 'result') {
      const s = m.state; if (s) { const sig = `${s.scope && s.scope.id}:${s.state}:${s.n_hypotheses}`; if (sig !== stateSig) { stateSig = sig; cmdKey = null; } }
      if (m.type === 'result' && m.decision) {
        const t = m.decision.timings_ms || {};
        stat(`直近: ${m.engine || ''} ${m.route === 'wasm' ? 'ブラウザ' : 'サーバー'} 合計 ${t.total ?? '?'} ms`);
      }
      return false;
    }
    if (m.type === 'pretrain' && m.status === 'done') { cmdKey = null; syncServerEngine(); }
    return false;
  }

  /* ------------------------------------------------------------ ブラウザの振り分け */
  function ensureWorker() {
    if (cfg.route !== 'wasm') return Promise.resolve(false);
    if (workerReady === true && workerModel === cfg.browser) return Promise.resolve(true);
    if (workerReady && workerReady.promise && workerModel === cfg.browser) return workerReady.promise;
    const spec = engines.find((e) => e.id === cfg.browser);
    if (!spec || !spec.browser_model || !kconf || !kconf.vocab) { workerReady = { msg: 'このエンジンはブラウザで動かせません' }; stat(); return Promise.resolve(false); }
    if (worker) worker.terminate();
    worker = new Worker(`${kconf.worker}?v=${V}`);
    workerModel = cfg.browser; cmdKey = null;
    const st = { msg: `${spec.label} を取得中…` };
    st.promise = new Promise((resolve) => {
      worker.onmessage = (e) => {
        const d = e.data;
        if (d.type === 'progress') { st.msg = `${spec.label} を取得中 ${Math.round(100 * d.loaded / d.total)}%`; stat(); }
        else if (d.type === 'ready') { workerReady = true; stat(`取得・初期化 ${d.ms} ms${d.simd ? '' : ' (SIMD なし)'}`); resolve(true); }
        else if (d.type === 'error' && !d.req) { workerReady = { msg: 'ブラウザで動かせません: ' + d.msg }; stat(); resolve(false); }
        else if (d.req && workerResolvers[d.req]) { workerResolvers[d.req](d); delete workerResolvers[d.req]; }
        else if (d.type === 'commands' && workerResolvers['cmd']) { workerResolvers['cmd'](d); delete workerResolvers['cmd']; }
      };
    });
    workerReady = st; stat();
    worker.postMessage({ type: 'init', wasm: `${kconf.wasm}?v=${V}`, model: spec.browser_model, vocab: kconf.vocab });
    return st.promise;
  }

  function packCommands(hyps) {
    const enc = new TextEncoder(); const parts = []; let n = 4;
    for (const [k, m, r, f] of hyps) { const b = enc.encode(k); parts.push([m, r, f, b]); n += 8 + b.length; }
    const buf = new Uint8Array(n); const dv = new DataView(buf.buffer); dv.setUint32(0, hyps.length, true); let o = 4;
    for (const [m, r, f, b] of parts) { dv.setUint32(o, m, true); buf[o + 4] = r; buf[o + 5] = f; dv.setUint16(o + 6, b.length, true); buf.set(b, o + 8); o += 8 + b.length; }
    return buf.buffer;
  }
  async function ensureCommands() {
    if (cmdKey) return cmdKey;
    if (cmdLoading) return cmdLoading;
    cmdLoading = (async () => {
      const j = await (await fetch(`/api/kana/commands?session=${encodeURIComponent(opts.session())}&engine=${encodeURIComponent(cfg.browser)}`)).json();
      calibrated = j.calibrated;
      const blob = packCommands(j.hyps);
      await new Promise((resolve) => { workerResolvers['cmd'] = resolve; worker.postMessage({ type: 'commands', key: j.cs_key, blob, params: j.params }, [blob]); });
      cmdKey = j.cs_key; return cmdKey;
    })().finally(() => { cmdLoading = null; });
    return cmdLoading;
  }
  async function routeLocal(pcm, meta = {}) {
    if (!(await ensureWorker())) return;
    for (let attempt = 0; attempt < 2; attempt++) {
      const key = await ensureCommands();
      const req = `r${++reqSeq}`;
      const d = await new Promise((resolve) => { workerResolvers[req] = resolve; worker.postMessage({ type: 'route', req, pcm: pcm.slice() }); });
      if (d.type === 'error') { stat('振り分けに失敗: ' + d.msg); return; }
      if (d.key !== key) { cmdKey = null; continue; }
      const o = d.out;
      const reply = await sendDecision({ type: 'decision', req, cs_key: key, engine: cfg.browser, free_kana: o.free_kana, free_score: o.free_score,
        n_free: o.n_free, cands: o.cands, none_prob: o.none_prob, action: o.action, reason: o.reason, timings_ms: { ...o.timings_ms, wall: o.wall_ms },
        audio_sec: +(pcm.length / 16000).toFixed(2), ...meta });
      if (reply === 'stale') { cmdKey = null; continue; }
      return;
    }
  }
  function sendDecision(msg) {
    // 結果は通常の result としてデモの onMessage に届く。stale のときだけここで振り分け直す
    return new Promise((resolve) => {
      if (!ws || ws.readyState !== 1) { resolve('closed'); return; }
      const onm = (e) => { let m; try { m = JSON.parse(e.data); } catch (_) { return; }
        if (m.type === 'stale') { ws.removeEventListener('message', onm); resolve('stale'); }
        else if (m.type === 'result' && m.req === msg.req) { ws.removeEventListener('message', onm); resolve('ok'); } };
      ws.addEventListener('message', onm);
      ws.send(JSON.stringify(msg));
      setTimeout(() => { ws.removeEventListener('message', onm); resolve('timeout'); }, 15000);
    });
  }

  async function ensureVad() {
    if (vad) return vad;
    if (!vadLoading) vadLoading = import(`/shared/vad.js?v=${V}`).then((mod) => {
      vad = mod.makeVad({
        onUtterance: (f32) => { pending.push(f32); drain(); },
        onState: (s) => opts.onMessage({ type: 'vad', speaking: s === 'speech' }),
      });
      return vad;
    });
    return vadLoading;
  }
  async function drain() {
    if (busy) return; busy = true;
    try { while (pending.length) await routeLocal(pending.shift()); } finally { busy = false; }
  }

  /** マイクの 16 kHz PCM16。サーバー振り分けなら従来どおり送り、ブラウザ振り分けなら手元の VAD → WebAssembly へ */
  function feed(int16, w) {
    if (w) ws = w;
    if (cfg.route !== 'wasm') { if (ws && ws.readyState === 1) ws.send(int16.buffer); return; }
    const f = new Float32Array(int16.length); for (let i = 0; i < int16.length; i++) f[i] = int16[i] / 32768;
    if (vad) vad.feed(f); else ensureVad().then((v) => v.feed(f));
  }

  /** テキスト → サーバーで TTS → ブラウザで振り分け (マイク無しで WebAssembly の経路を試す) */
  async function say(text, snr) {
    const r = await fetch('/api/say', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session: opts.session(), text, snr_db: snr ? +snr : null, recognize: false }) });
    const j = await r.json(); if (j.error) { stat('TTS エラー: ' + j.error); return j; }
    const bin = atob(j.pcm16_b64); const n = bin.length >> 1; const f = new Float32Array(n);
    for (let i = 0; i < n; i++) { let v = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8); if (v >= 32768) v -= 65536; f[i] = v / 32768; }
    await routeLocal(f, { tts_text: j.tts_text });
    return j;
  }

  return { mount, attach, onMessage, feed, say, isLocal: () => cfg.route === 'wasm', config: () => ({ ...cfg }), ensureWorker };
})();
