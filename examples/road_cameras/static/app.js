/* 指図 デモ UI: WebSocket で音声を送り、状態とカメラ描画を更新する */
(() => {
const $ = (s) => document.querySelector(s);
const state = { session: localStorage.getItem('sashizu_session') || '', ws: null, snap: null, cams: [], scopes: [], hier: null,
  reg: { bureau: null, office: null, routes: new Set() }, mic: null };

/* ---------------- WebSocket ---------------- */
function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/ws?session=${state.session}`);
  ws.binaryType = 'arraybuffer';
  ws.onmessage = (e) => onMessage(JSON.parse(e.data));
  ws.onclose = () => setTimeout(connect, 1500);
  state.ws = ws;
}
function send(obj) { if (state.ws && state.ws.readyState === 1) state.ws.send(JSON.stringify(obj)); }

function onMessage(m) {
  if (m.type === 'hello') { state.session = m.session; localStorage.setItem('sashizu_session', m.session); state.scopes = m.scopes; renderScopes(); applyState(m.state); loadCams(); }
  else if (m.type === 'state') applyState(m.state);
  else if (m.type === 'vad') { $('#vadChip').textContent = m.speaking ? '発話中' : '待機'; $('#vadChip').classList.toggle('on', m.speaking); }
  else if (m.type === 'result') onResult(m);
  else if (m.type === 'pretrain') onPretrain(m);
}

/* ---------------- 状態・描画 ---------------- */
function applyState(s) {
  const prevScope = state.snap && state.snap.scope.id;
  state.snap = s;
  $('#stateChip').textContent = s.state; $('#stateChip').className = 'statechip ' + s.state;
  $('#stateDesc').textContent = s.description;
  $('#hypCount').textContent = `カメラ ${s.n_cameras} 台 / 仮説 ${s.n_hypotheses.toLocaleString()} 通り`;
  $('#scopeInfo').textContent = `${s.n_cameras} 台`;
  $('#allowedN').textContent = `(${s.allowed_commands.length} 種)`;
  $('#allowed').innerHTML = s.allowed_commands.map(c => `<li><span class="ex">${c.example}</span><span>${c.description}</span>${c.risk !== 'low' ? `<span class="risk">要確認</span>` : ''}</li>`).join('');
  $('#quick').innerHTML = s.allowed_commands.filter(c => !c.example.includes('<')).slice(0, 8).map(c => `<button data-say="${c.example}">${c.example}</button>`).join('');
  if ($('#scopeSel').value !== s.scope.id) $('#scopeSel').value = s.scope.id;
  if (prevScope !== s.scope.id) loadCams();
  if (s.state === 'WALL' || !s.camera) { $('#focus').hidden = true; $('#wall').hidden = false; }
  else { $('#focus').hidden = false; $('#wall').hidden = true; renderFocus(s); }
}

function onResult(m) {
  const d = m.decision;
  $('#speech').textContent = m.speech || (d.action === 'none' ? '(システム宛ではないと判断)' : '—');
  $('#freeKana').textContent = d.free_kana || '';
  const dec = $('#decision'); dec.className = 'decision ' + d.action;
  const lbl = { execute: '実行', confirm: '確認', reject: '棄却', none: '該当なし' }[d.action];
  dec.textContent = `${lbl}  ${d.top ? d.top.text : ''}  ${d.reason || ''}`;
  const rows = d.candidates.slice(0, 5).map(c => bar(c.text + (c.intent !== 'select_camera' ? ` (${c.intent})` : ''), c.prob, ''));
  rows.push(bar('該当なし (自由認識そのまま)', d.none_prob, 'none'));
  $('#nbest').innerHTML = rows.join('');
  const t = d.timings_ms; $('#timings').textContent = `encoder ${t.encode}ms / 自由認識 ${t.transcribe}ms / 絞り込み ${t.shortlist ?? 0}ms / 採点 ${t.score ?? 0}ms / 合計 ${t.total}ms  (音声 ${m.audio_sec ?? '?'}s)`;
  if (m.speech && d.action !== 'none') speak(m.speech);
  if (m.applied && m.applied.camera) flashTile(m.applied.camera);
  applyState(m.state);
  const log = $('#log'); log.textContent = `${new Date().toLocaleTimeString()} [${d.state}] ${d.free_kana} -> ${d.action} ${d.top ? d.top.text + ' p=' + d.top.prob : ''}\n` + log.textContent;
}
function bar(text, p, cls) { return `<div class="row"><div class="bar ${cls}"><i style="width:${(p * 100).toFixed(1)}%"></i><span>${text}</span></div><div>${(p * 100).toFixed(1)}%</div></div>`; }

function speak(text) {
  if (!('speechSynthesis' in window)) return;
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text); u.lang = 'ja-JP'; u.rate = 1.1;
  // 自分の声を拾わないようマイク送信を一時停止
  state.muteUntil = Date.now() + Math.min(4000, 400 + text.length * 120);
  speechSynthesis.speak(u);
}

/* ---------------- カメラ壁 ---------------- */
async function loadCams() {
  const sc = state.scopes.find(s => s.id === (state.snap && state.snap.scope.id));
  if (!sc) return;
  const params = new URLSearchParams();
  // 範囲の条件を API に渡す (単一値の filter のみ簡易対応、複数は全件取ってクライアントで絞る)
  const all = await (await fetch('/api/cameras?limit=5000')).json();
  state.cams = all.filter(c => inScope(sc, c));
  renderWall();
}
function inScope(sc, c) {
  if (sc.exclude_ids.includes(c.id)) return false;
  if (sc.ids.includes(c.id)) return true;
  const f = sc.filters; if (!f || !Object.keys(f).length) return false;
  return Object.entries(f).every(([k, v]) => v.includes(c.attrs[k]));
}
function renderWall() {
  const wall = $('#wall'); wall.innerHTML = '';
  for (const c of state.cams.slice(0, 400)) {
    const div = document.createElement('div'); div.className = 'tile'; div.dataset.id = c.id;
    const cv = document.createElement('canvas'); cv.width = 320; cv.height = 180;
    drawScene(cv.getContext('2d'), c, { pan: 0, tilt: 0, zoom: 1 }, 320, 180, true);
    const lbl = document.createElement('div'); lbl.className = 'lbl'; lbl.innerHTML = `${c.label}<small>${c.attrs.route_label}</small>`;
    div.append(cv, lbl); div.onclick = () => send({ type: 'click_camera', id: c.id, label: c.label });
    wall.append(div);
  }
}
function flashTile(id) { const t = document.querySelector(`.tile[data-id="${id}"]`); if (t) { t.classList.add('hl'); setTimeout(() => t.classList.remove('hl'), 1200); } }

function renderFocus(s) {
  const c = state.cams.find(x => x.id === s.camera.id) || { id: s.camera.id, label: s.camera.label, attrs: s.camera.attrs };
  $('#focusLabel').textContent = c.label;
  $('#focusMeta').textContent = `${c.attrs.office} / ${c.attrs.route_label} / ${c.attrs.direction}`;
  const p = s.ptz || { pan: 0, tilt: 0, zoom: 1 };
  $('#ptzInfo').textContent = `pan ${p.pan.toFixed(0)}°  tilt ${p.tilt.toFixed(0)}°  zoom ×${p.zoom.toFixed(1)}  preset ${p.presets ? p.presets.length : 0}`;
  const cv = $('#focusCanvas'); drawScene(cv.getContext('2d'), c, p, cv.width, cv.height, false);
}

/* 手続き的な道路シーン: カメラ ID を種にした決定的な景色。pan/tilt で視野が動き、zoom で拡大する */
function hash(s) { let h = 2166136261; for (const ch of s) { h ^= ch.charCodeAt(0); h = Math.imul(h, 16777619); } return h >>> 0; }
function rng(seed) { let x = seed || 1; return () => { x ^= x << 13; x >>>= 0; x ^= x >> 17; x ^= x << 5; x >>>= 0; return (x >>> 0) / 4294967296; }; }
function drawScene(ctx, cam, ptz, W, H, thumb) {
  const r = rng(hash(cam.id));
  const hour = new Date().getHours(); const night = hour < 6 || hour >= 18;
  const zoom = ptz.zoom, panPx = -ptz.pan / 170 * W * 0.8, tiltPx = ptz.tilt / 45 * H * 0.5;
  ctx.save(); ctx.clearRect(0, 0, W, H);
  ctx.translate(W / 2, H / 2); ctx.scale(zoom, zoom); ctx.translate(-W / 2 + panPx, -H / 2 + tiltPx);
  // 空と遠景 (視野外まで描くため広めに)
  const g = ctx.createLinearGradient(0, -H, 0, H * 0.6); g.addColorStop(0, night ? '#0a1020' : '#7fb6e8'); g.addColorStop(1, night ? '#1a2030' : '#dfeaf5');
  ctx.fillStyle = g; ctx.fillRect(-W, -H, 3 * W, 1.6 * H);
  ctx.fillStyle = night ? '#1c2430' : '#9fb3a5';
  for (let i = -6; i < 12; i++) { const bh = 20 + r() * 60; ctx.fillRect(i * W / 6, H * 0.6 - bh, W / 6 - 4, bh); }
  // 路面
  ctx.fillStyle = night ? '#2a2c30' : '#5a5f66'; ctx.fillRect(-W, H * 0.6, 3 * W, H);
  ctx.beginPath(); ctx.moveTo(-W, H * 1.6); ctx.lineTo(W * 0.35, H * 0.6); ctx.lineTo(W * 0.65, H * 0.6); ctx.lineTo(2 * W, H * 1.6); ctx.closePath();
  ctx.fillStyle = night ? '#35383d' : '#6f747b'; ctx.fill();
  ctx.strokeStyle = '#e8e8e8'; ctx.setLineDash([12, 14]); ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(W / 2, H * 0.6); ctx.lineTo(W / 2, H * 1.6); ctx.stroke(); ctx.setLineDash([]);
  // 車 (時刻で位置が変わる = 生きている感)
  const t = Date.now() / 1000; const n = 3 + Math.floor(r() * 5);
  for (let i = 0; i < n; i++) { const ph = (r() * 10 + t * (0.15 + r() * 0.2)) % 1; const y = H * 0.6 + ph * H * 0.9; const s = 0.15 + ph; const lane = r() < 0.5 ? -1 : 1;
    const x = W / 2 + lane * (W * 0.06 + ph * W * 0.28) - 20 * s; ctx.fillStyle = ['#c33', '#ddd', '#3563c9', '#333', '#eb2'][i % 5]; ctx.fillRect(x, y - 14 * s, 40 * s, 22 * s);
    if (night) { ctx.fillStyle = '#ffd'; ctx.fillRect(x + 2, y - 8 * s, 6 * s, 4 * s); ctx.fillRect(x + 32 * s, y - 8 * s, 6 * s, 4 * s); } }
  // 標識
  ctx.fillStyle = '#1f6f3f'; ctx.fillRect(W * 0.66, H * 0.42, W * 0.2, H * 0.1); ctx.fillStyle = '#fff'; ctx.font = `${Math.floor(H * 0.05)}px sans-serif`; ctx.fillText(cam.attrs.route_label || '', W * 0.68, H * 0.49);
  ctx.restore();
  // オーバーレイ (カメラ映像風)
  ctx.fillStyle = 'rgba(0,0,0,.45)'; ctx.fillRect(0, 0, W, thumb ? 16 : 26);
  ctx.fillStyle = '#fff'; ctx.font = `${thumb ? 11 : 15}px ui-monospace, monospace`;
  ctx.fillText(`${cam.label}  ${new Date().toLocaleString('ja-JP')}  P${ptz.pan.toFixed(0)} T${ptz.tilt.toFixed(0)} Z${ptz.zoom.toFixed(1)}`, 6, thumb ? 12 : 18);
}
setInterval(() => { if (state.snap && state.snap.state !== 'WALL' && state.snap.camera) renderFocus(state.snap); }, 250);

/* ---------------- マイク ---------------- */
async function toggleMic() {
  if (state.mic) { stopMic(); return; }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) { alert('マイクは https または localhost でのみ使えます (ssh -L で転送するか --ssl-dir で起動)'); return; }
  const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
  const ac = new AudioContext(); const src = ac.createMediaStreamSource(stream);
  const proc = ac.createScriptProcessor(4096, 1, 1); let carry = 0;
  proc.onaudioprocess = (e) => {
    const inSr = e.inputBuffer.sampleRate;             // 端末側でルートが変わると変動するので毎回読む
    const x = e.inputBuffer.getChannelData(0);
    let peak = 0; for (let i = 0; i < x.length; i++) peak = Math.max(peak, Math.abs(x[i]));
    $('#vuBar').style.width = Math.min(100, peak * 300) + '%';
    if (state.muteUntil && Date.now() < state.muteUntil) return;
    const ratio = inSr / 16000; const outLen = Math.floor((x.length - carry) / ratio);
    const out = new Int16Array(outLen); let pos = carry;
    for (let i = 0; i < outLen; i++) { const j = Math.floor(pos); out[i] = Math.max(-32768, Math.min(32767, x[Math.min(j, x.length - 1)] * 32768)); pos += ratio; }
    carry = pos - x.length;
    if (state.ws && state.ws.readyState === 1) state.ws.send(out.buffer);
  };
  src.connect(proc); proc.connect(ac.destination);
  state.mic = { stream, ac, proc };
  $('#micBtn').textContent = '⏹ マイク停止'; $('#micBtn').classList.add('on');
}
function stopMic() { const m = state.mic; if (!m) return; m.proc.disconnect(); m.stream.getTracks().forEach(t => t.stop()); m.ac.close(); state.mic = null; $('#micBtn').textContent = '🎙 マイク開始'; $('#micBtn').classList.remove('on'); $('#vuBar').style.width = '0'; }

/* ---------------- 範囲 (scope) ---------------- */
function renderScopes() {
  const sel = $('#scopeSel'); sel.innerHTML = state.scopes.map(s => `<option value="${s.id}">${s.name} (${s.n_cameras})</option>`).join('');
  if (state.snap) sel.value = state.snap.scope.id;
  const tb = $('#scopeTable tbody');
  tb.innerHTML = state.scopes.map(s => `<tr><td>${s.name}<br><small class="muted">${s.id}</small></td><td><small>${Object.entries(s.filters).map(([k, v]) => `${k}: ${v.join(', ')}`).join('<br>') || '(個別指定)'}${s.ids.length ? `<br>+${s.ids.length} 台` : ''}</small></td><td>${s.n_cameras}</td>
    <td>${s.profile && s.profile.trained_at ? `<small>${s.profile.trained_at}<br>精度 ${((s.profile.accuracy_after ?? s.profile.accuracy) * 100).toFixed(1)}% / 該当なし再現 ${(s.profile.none_recall * 100).toFixed(0)}%<br>T=${s.profile.calibration.temperature.toFixed(2)} β=${s.profile.calibration.none_bias.toFixed(2)}</small>` : '<small class="muted">未学習</small>'}</td>
    <td><button data-use="${s.id}">使う</button> <button data-del="${s.id}">削除</button></td></tr>`).join('');
}
async function refreshScopes() { state.scopes = await (await fetch('/api/scopes')).json(); renderScopes(); }
async function useScope(id) { const r = await (await fetch('/api/session/scope', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session: state.session, scope_id: id }) })).json(); if (r.state) { applyState(r.state); loadCams(); renderCamTable(); } }

async function loadHierarchy() { state.hier = await (await fetch('/api/hierarchy')).json(); renderReg(); }
function renderReg() {
  const h = state.hier, rg = state.reg;
  $('#bureauList').innerHTML = Object.entries(h).map(([b, offs]) => `<li data-b="${b}" class="${rg.bureau === b ? 'sel' : ''}"><span>${b}</span><small class="muted">${Object.values(offs).reduce((a, o) => a + o.n_cameras, 0)} 台</small></li>`).join('');
  const offs = rg.bureau ? h[rg.bureau] : {};
  $('#officeList').innerHTML = Object.entries(offs).map(([o, v]) => `<li data-o="${o}" class="${rg.office === o ? 'sel' : ''}"><span>${o}</span><small class="muted">${v.prefs.join('・')} ${v.n_cameras} 台</small></li>`).join('');
  $('#officeHint').textContent = rg.bureau ? '' : '(整備局を選択)';
  const routes = rg.office ? offs[rg.office].routes : [];
  $('#routeList').innerHTML = routes.map(r => `<li data-r="${r}" class="${rg.routes.has(r) ? 'sel' : ''}"><span>国道 ${r} 号</span></li>`).join('');
  const f = currentFilters(); const n = f ? state.allCams.filter(c => Object.entries(f).every(([k, v]) => v.includes(c.attrs[k]))).length : 0;
  $('#regPreview').textContent = f ? `→ ${n} 台` : '';
}
function currentFilters() { const rg = state.reg; if (!rg.bureau) return null; const f = { bureau: [rg.bureau] }; if (rg.office) f.office = [rg.office]; if (rg.routes.size) f.route = [...rg.routes]; return f; }
async function saveScope() {
  const f = currentFilters(); if (!f) { alert('整備局を選んでください'); return; }
  const name = $('#scopeName').value || `${state.reg.office || state.reg.bureau}${state.reg.routes.size ? ' R' + [...state.reg.routes].join('/') : ''}`;
  const body = { id: $('#scopeId').value || undefined, name, filters: f };
  const r = await (await fetch('/api/scopes', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })).json();
  await refreshScopes(); await useScope(r.id);
}

/* ---------------- 事前学習 ---------------- */
async function startPretrain() {
  const seeds = Array.from({ length: +$('#ptSeeds').value }, (_, i) => i + 1);
  const carriers = $('#ptCarriers').value === '1' ? ['{camera}'] : ['{camera}', '{camera}を表示'];
  const r = await fetch('/api/pretrain', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session: state.session, scope_id: state.snap.scope.id, seeds, carriers }) });
  const j = await r.json(); if (j.error) { $('#ptStatus').textContent = 'エラー: ' + j.error; return; }
  $('#ptStatus').textContent = '開始しました…'; $('#ptResult').innerHTML = '';
}
function onPretrain(m) {
  if (m.status === 'running') { $('#ptBar').style.width = (m.step / m.total * 100).toFixed(1) + '%'; $('#ptStatus').textContent = `${m.step}/${m.total}  ${m.text}  → ${m.free}  ${m.ok ? '○' : '×'}   累積正解率 ${(m.acc * 100).toFixed(1)}%`; }
  else if (m.status === 'done') {
    const r = m.result; $('#ptBar').style.width = '100%'; $('#ptStatus').textContent = `完了 (${r.seconds.toFixed(0)} 秒, ${r.n_utts} 発話)`;
    $('#ptResult').innerHTML = `<div class="kv">
      <span>正解率 (校正前 → 校正後)</span><b>${(r.accuracy * 100).toFixed(1)}% → ${(r.accuracy_after * 100).toFixed(1)}%</b>
      <span>雑音別</span><b>${Object.entries(r.accuracy_by_snr).map(([k, v]) => `${k} ${(v * 100).toFixed(0)}%`).join(' / ')}</b>
      <span>文法外 → 該当なし/棄却</span><b>${(r.none_recall * 100).toFixed(0)}%  (誤実行 ${(r.false_accept * 100).toFixed(0)}%)</b>
      <span>校正</span><b>T=${r.calibration.temperature.toFixed(2)}  β=${r.calibration.none_bias.toFixed(2)}  ECE ${r.ece_before.toFixed(3)} → ${r.ece_after.toFixed(3)}</b>
      <span>遅延 (中央値)</span><b>${Object.entries(r.latency_ms).map(([k, v]) => `${k} ${v.toFixed(0)}ms`).join(' / ')}</b>
      <span>追加した読み</span><b>${r.added_readings.length ? r.added_readings.map(a => `${a.label}: ${a.from} → <u>${a.reading}</u>`).join('<br>') : 'なし'}</b>
      <span>混同しやすい対</span><b>${r.confusions.length ? r.confusions.map(c => `${c.from} → ${c.to} (${c.count})`).join('<br>') : 'なし'}</b>
      <span>警告</span><b>${r.warnings.length ? r.warnings.slice(0, 10).join('<br>') : 'なし'}</b></div>`;
    refreshScopes(); applyState(m.state); renderCamTable();
  } else if (m.status === 'error') $('#ptStatus').textContent = 'エラー: ' + m.error;
}
async function renderCamTable() {
  const cams = state.cams;
  $('#camTable').innerHTML = `<table><tr><th>カメラ</th><th>事務所 / 路線</th><th>読み (ASR 形カナ)</th></tr>` + cams.slice(0, 300).map(c => `<tr><td>${c.label}</td><td>${c.attrs.office} / ${c.attrs.route_label}</td><td>${c.readings.map(r => `<span class="rd">${r}</span>`).join(' , ')} <span class="rd" data-add="${c.id}">＋追加</span></td></tr>`).join('') + '</table>';
}

/* ---------------- イベント ---------------- */
$('#micBtn').onclick = toggleMic;
// マウス/キーボード操作も音声と同じ状態機械を通す (WS の制御メッセージ)
$('#backBtn').onclick = () => send({ type: 'intent', intent: 'back' });
$('#focus').addEventListener('click', (e) => { if (e.target === e.currentTarget) send({ type: 'intent', intent: 'back' }); });   // 背景クリックで閉じる
document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  if (!state.snap || state.snap.state === 'WALL') return;
  const map = { Escape: 'back', ArrowRight: 'pan_right', ArrowLeft: 'pan_left', ArrowUp: 'tilt_up', ArrowDown: 'tilt_down', '+': 'zoom_in', '=': 'zoom_in', '-': 'zoom_out', Home: 'home' };
  const it = map[e.key]; if (!it) return;
  e.preventDefault(); send({ type: 'intent', intent: it });
});
$('#sayForm').onsubmit = async (e) => { e.preventDefault(); const text = $('#sayText').value.trim(); if (!text) return; await say(text); };
$('#quick').onclick = (e) => { const b = e.target.closest('button'); if (b) say(b.dataset.say); };
async function say(text) { $('#speech').textContent = '… TTS 合成中'; const snr = $('#saySnr').value; const r = await fetch('/api/say', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session: state.session, text, snr_db: snr ? +snr : null }) }); const j = await r.json(); if (j.error) $('#speech').textContent = 'エラー: ' + j.error; }
$('#scopeSel').onchange = (e) => useScope(e.target.value);
$('#tabMonitor').onclick = () => { $('#monitor').hidden = false; $('#register').hidden = true; $('#tabMonitor').classList.add('active'); $('#tabRegister').classList.remove('active'); };
$('#tabRegister').onclick = async () => { $('#monitor').hidden = true; $('#register').hidden = false; $('#tabRegister').classList.add('active'); $('#tabMonitor').classList.remove('active'); if (!state.hier) { state.allCams = await (await fetch('/api/cameras?limit=5000')).json(); await loadHierarchy(); } renderCamTable(); };
$('#bureauList').onclick = (e) => { const li = e.target.closest('li'); if (!li) return; state.reg = { bureau: li.dataset.b, office: null, routes: new Set() }; renderReg(); };
$('#officeList').onclick = (e) => { const li = e.target.closest('li'); if (!li) return; state.reg.office = li.dataset.o; state.reg.routes = new Set(); renderReg(); };
$('#routeList').onclick = (e) => { const li = e.target.closest('li'); if (!li) return; const r = +li.dataset.r; state.reg.routes.has(r) ? state.reg.routes.delete(r) : state.reg.routes.add(r); renderReg(); };
$('#saveScope').onclick = saveScope;
$('#scopeTable').onclick = async (e) => { const b = e.target.closest('button'); if (!b) return; if (b.dataset.use) useScope(b.dataset.use); if (b.dataset.del && confirm('削除しますか?')) { await fetch('/api/scopes/' + b.dataset.del, { method: 'DELETE' }); refreshScopes(); } };
$('#pretrainBtn').onclick = startPretrain;
$('#camTable').onclick = async (e) => { const s = e.target.closest('[data-add]'); if (!s) return; const r = prompt('追加する読み (カタカナ)'); if (!r) return; await fetch(`/api/cameras/${s.dataset.add}/readings`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ reading: r }) }); await loadCams(); renderCamTable(); };

connect();
})();
