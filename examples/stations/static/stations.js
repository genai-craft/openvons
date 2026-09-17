/* 駅名で動く路線図: WebSocket で音声を送り、状態に合わせて路線図を描く */
(() => {
const $ = (s) => document.querySelector(s);
const state = { session: localStorage.getItem('jev_st_session') || '', ws: null, snap: null, cams: [], scopes: [], hier: null,
  reg: { bureau: null, lines: new Map() }, mic: null, view: { cx: 0, cy: 0, zoom: 1, drag: null } };

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
  if (m.type === 'hello') { state.session = m.session; localStorage.setItem('jev_st_session', m.session); state.scopes = m.scopes; renderScopes(); applyState(m.state); loadCams(); }
  else if (m.type === 'state') applyState(m.state);
  else if (m.type === 'vad') { $('#vadChip').textContent = m.speaking ? '発話中' : '待機'; $('#vadChip').classList.toggle('on', m.speaking); }
  else if (m.type === 'result') onResult(m);
  else if (m.type === 'pretrain') onPretrain(m);
}

/* ---------------- 状態 ---------------- */
function applyState(s) {
  const prev = state.snap; const prevScope = prev && prev.scope.id;
  state.snap = s;
  $('#stateChip').textContent = s.state; $('#stateChip').className = 'statechip ' + (s.state === 'STATION' ? 'FOCUS' : s.state);
  $('#stateDesc').textContent = s.description;
  $('#hypCount').textContent = `駅 ${s.n_cameras} / 仮説 ${s.n_hypotheses.toLocaleString()} 通り`;
  $('#scopeInfo').textContent = `${s.n_cameras} 駅`;
  $('#allowedN').textContent = `(${s.allowed_commands.length} 種)`;
  $('#allowed').innerHTML = s.allowed_commands.map(c => `<li><span class="ex">${c.example}</span><span>${c.description}</span>${c.risk !== 'low' ? `<span class="risk">要確認</span>` : ''}</li>`).join('');
  $('#quick').innerHTML = s.allowed_commands.filter(c => !c.example.includes('<')).slice(0, 8).map(c => `<button data-say="${c.example}">${c.example}</button>`).join('');
  if ($('#scopeSel').value !== s.scope.id) $('#scopeSel').value = s.scope.id;
  if (prevScope !== s.scope.id || !state.cams) { loadCams(); return; }
  renderCard(s); drawFocus();
  if (s.state === 'MAP' && (!prev || prev.state !== 'MAP')) fitAll();      // 「全体に戻る」で必ず全体表示へ
}
function onResult(m) {
  const d = m.decision;
  $('#speech').textContent = m.speech || (d.action === 'none' ? '(システム宛ではないと判断)' : '—');
  $('#freeKana').textContent = d.free_kana || '';
  const dec = $('#decision'); dec.className = 'decision ' + d.action;
  dec.textContent = `${{ execute: '実行', confirm: '確認', reject: '棄却', none: '該当なし' }[d.action]}  ${d.top ? d.top.text : ''}  ${d.reason || ''}`;
  const rows = d.candidates.slice(0, 5).map(c => bar(c.text + (!['select_station', 'route'].includes(c.intent) ? ` (${c.intent})` : ''), c.prob, ''));
  rows.push(bar('該当なし (自由認識そのまま)', d.none_prob, 'none'));
  $('#nbest').innerHTML = rows.join('');
  const t = d.timings_ms; $('#timings').textContent = `encoder ${t.encode}ms / 自由認識 ${t.transcribe}ms / 絞り込み ${t.shortlist ?? 0}ms / 採点 ${t.score ?? 0}ms / 合計 ${t.total}ms`;
  if (m.speech && d.action !== 'none') speak(m.speech);
  applyState(m.state);
  const log = $('#log'); log.textContent = `${new Date().toLocaleTimeString()} [${d.state}] ${d.free_kana} -> ${d.action} ${d.top ? d.top.text + ' p=' + d.top.prob : ''}\n` + log.textContent;
}
function bar(text, p, cls) { return `<div class="row"><div class="bar ${cls}"><i style="width:${(p * 100).toFixed(1)}%"></i><span>${text}</span></div><div>${(p * 100).toFixed(1)}%</div></div>`; }
function speak(text) { if (!('speechSynthesis' in window)) return; speechSynthesis.cancel(); const u = new SpeechSynthesisUtterance(text); u.lang = 'ja-JP'; u.rate = 1.1; state.muteUntil = Date.now() + Math.min(4000, 400 + text.length * 120); speechSynthesis.speak(u); }

/* ---------------- 駅データ・地図 (Leaflet + OpenStreetMap) ---------------- */
const map = L.map('map', { zoomControl: true, attributionControl: true }).setView([35.68, 139.76], 11);
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19, attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors' }).addTo(map);
const layers = { lines: L.layerGroup().addTo(map), stations: L.layerGroup().addTo(map), route: L.layerGroup().addTo(map) };
const PALETTE = ['#4cc2ff', '#3fb950', '#d29922', '#f85149', '#a371f7', '#79c0ff', '#ff7b72', '#56d364', '#e3b341', '#f778ba'];
let lineColor = {};
async function loadCams() {
  const sc = state.scopes.find(s => s.id === (state.snap && state.snap.scope.id)); if (!sc) return;
  if (!state.all) state.all = await (await fetch('/api/cameras?limit=20000')).json();
  if (!state.hier) { state.hier = await (await fetch('/api/hierarchy')).json(); for (const ls of Object.values(state.hier)) for (const v of Object.values(ls)) lineColor[v.code] = (v.color && v.color !== 'NULL') ? v.color : null; }
  state.cams = state.all.filter(c => inScope(sc, c));
  drawBase(); fitAll(); renderCard(state.snap); drawFocus();
}
function inScope(sc, c) {
  if (sc.exclude_ids.includes(c.id)) return false; if (sc.ids.includes(c.id)) return true;
  const f = sc.filters; if (!f || !Object.keys(f).length) return false;
  return Object.entries(f).every(([k, v]) => Array.isArray(c.attrs[k]) ? c.attrs[k].some(x => v.includes(x)) : v.includes(c.attrs[k]));
}
function colorOf(lc, i) { return lineColor[lc] || PALETTE[i % PALETTE.length]; }
function drawBase() {
  layers.lines.clearLayers(); layers.stations.clearLayers(); state.markers = {};
  const byLine = new Map();
  for (const c of state.cams) for (const p of c.attrs.positions) { if (!byLine.has(p.line_code)) byLine.set(p.line_code, []); byLine.get(p.line_code).push([p.index, c]); }
  let i = 0;
  for (const [lc, arr] of byLine) { arr.sort((a, b) => a[0] - b[0]); const col = colorOf(lc, i++);
    let seg = []; const flush = () => { if (seg.length > 1) L.polyline(seg, { color: col, weight: 5, opacity: .85 }).addTo(layers.lines); seg = []; };
    arr.forEach(([idx, c], k) => { if (k && idx !== arr[k - 1][0] + 1) flush(); seg.push([c.attrs.lat, c.attrs.lng]); }); flush(); }
  const many = state.cams.length > 60;
  for (const c of state.cams) {
    const m = L.circleMarker([c.attrs.lat, c.attrs.lng], { radius: 6, color: '#fff', weight: 2, fillColor: '#0b0f14', fillOpacity: 1 }).addTo(layers.stations);
    m.bindTooltip(c.label, { permanent: !many, direction: 'top', offset: [0, -6], className: 'st-label' });
    m.on('click', () => send({ type: 'click_camera', id: c.id, label: c.label }));
    state.markers[c.id] = m;
  }
}
function fitAll() { if (!state.cams || !state.cams.length) return; map.fitBounds(L.latLngBounds(state.cams.map(c => [c.attrs.lat, c.attrs.lng])).pad(0.08)); }
function drawFocus() {
  const s = state.snap; layers.route.clearLayers();
  for (const m of Object.values(state.markers || {})) { m.setStyle({ radius: 6, fillColor: '#0b0f14', color: '#fff' }); const t = m.getTooltip(); if (t && t.getElement()) t.getElement().classList.remove('focus'); }
  if (!s || !s.station) return;
  const st = s.station; const all = state.all || state.cams;
  const r = s.view && s.view.route;
  if (r) {
    const pts = r.stations.map(id => all.find(c => c.id === id)).filter(Boolean);
    let k = 0;
    for (const leg of r.legs) { const ids = r.stations.slice(r.stations.indexOf(leg.from), r.stations.indexOf(leg.to) + 1); const seg = ids.map(id => all.find(c => c.id === id)).filter(Boolean).map(c => [c.attrs.lat, c.attrs.lng]);
      L.polyline(seg, { color: '#000', weight: 12, opacity: .5 }).addTo(layers.route); L.polyline(seg, { color: colorOf(leg.line_code, k++), weight: 7, opacity: 1 }).addTo(layers.route); }
    for (const c of pts) L.circleMarker([c.attrs.lat, c.attrs.lng], { radius: 7, color: '#fff', weight: 2, fillColor: '#4cc2ff', fillOpacity: 1 }).addTo(layers.route).bindTooltip(c.label, { permanent: true, direction: 'top', offset: [0, -7], className: 'st-label' });
    map.flyToBounds(L.latLngBounds(pts.map(c => [c.attrs.lat, c.attrs.lng])).pad(0.2), { duration: 0.8 });
  } else {
    const m = state.markers[st.id]; if (m) { m.setStyle({ radius: 10, fillColor: '#4cc2ff', color: '#fff' }); const t = m.getTooltip(); if (t && t.getElement()) t.getElement().classList.add('focus'); }
    const z = Math.min(16, 12 + Math.log2(Math.max(1, s.view.zoom)));
    map.flyTo([st.attrs.lat, st.attrs.lng], z, { duration: 0.7 });
  }
}
function renderCard(s) {
  const card = $('#stationCard'); if (!s || !s.station) { card.hidden = true; return; }
  card.hidden = false; const st = s.station; const lc = s.view.line_code; const all = state.all || state.cams;
  const pos = st.attrs.positions.find(p => p.line_code === lc) || st.attrs.positions[0];
  $('#stLabel').textContent = st.label; $('#stMeta').textContent = `${st.attrs.pref} / ${pos.line} (${pos.index + 1}/${pos.n})  ${st.attrs.lines.length > 1 ? '乗換 ' + (st.attrs.lines.length - 1) + ' 路線: ' + st.attrs.lines.filter(l => l !== pos.line).slice(0, 4).join('・') : ''}`;
  const at = (i) => all.find(c => c.attrs.positions.some(p => p.line_code === pos.line_code && p.index === i));
  const prev = at(pos.index - 1), next = at(pos.index + 1);
  $('#stNeighbors').innerHTML = `${prev ? '◀ ' + prev.label : '（始点）'} <span class="cur">${st.label}</span> ${next ? next.label + ' ▶' : '（終点）'}`;
  const r = s.view.route; const rd = $('#stRoute');
  if (r) { rd.hidden = false; let k = 0; const o = all.find(c => c.id === r.stations[0]); rd.innerHTML = `<b>${o ? o.label : ''} → ${st.label}</b>　${r.stations.length - 1} 駅・乗換 ${r.transfers} 回・約 ${r.minutes} 分<br>` + r.legs.map(l => `<span class="leg" style="background:${colorOf(l.line_code, k++)}">${l.line} ${l.hops} 駅</span>`).join('→ '); }
  else rd.hidden = true;
}
window.addEventListener('resize', () => map.invalidateSize());
/* ---------------- マイク ---------------- */
async function toggleMic() {
  if (state.mic) { stopMic(); return; }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) { alert('マイクは https または localhost でのみ使えます'); return; }
  const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
  const ac = new AudioContext(); const src = ac.createMediaStreamSource(stream);
  const proc = ac.createScriptProcessor(4096, 1, 1); let carry = 0;
  proc.onaudioprocess = (e) => {
    const inSr = e.inputBuffer.sampleRate; const x = e.inputBuffer.getChannelData(0);
    let peak = 0; for (let i = 0; i < x.length; i++) peak = Math.max(peak, Math.abs(x[i])); $('#vuBar').style.width = Math.min(100, peak * 300) + '%';
    if (state.muteUntil && Date.now() < state.muteUntil) return;
    const ratio = inSr / 16000; const outLen = Math.floor((x.length - carry) / ratio); const out = new Int16Array(outLen); let pos = carry;
    for (let i = 0; i < outLen; i++) { const j = Math.floor(pos); out[i] = Math.max(-32768, Math.min(32767, x[Math.min(j, x.length - 1)] * 32768)); pos += ratio; }
    carry = pos - x.length; if (state.ws && state.ws.readyState === 1) state.ws.send(out.buffer);
  };
  src.connect(proc); proc.connect(ac.destination); state.mic = { stream, ac, proc };
  $('#micBtn').textContent = '⏹ マイク停止'; $('#micBtn').classList.add('on');
}
function stopMic() { const m = state.mic; if (!m) return; m.proc.disconnect(); m.stream.getTracks().forEach(t => t.stop()); m.ac.close(); state.mic = null; $('#micBtn').textContent = '🎙 マイク開始'; $('#micBtn').classList.remove('on'); $('#vuBar').style.width = '0'; }

/* ---------------- 範囲 (路線の登録) ---------------- */
function renderScopes() {
  const sel = $('#scopeSel'); sel.innerHTML = state.scopes.map(s => `<option value="${s.id}">${s.name} (${s.n_cameras})</option>`).join('');
  if (state.snap) sel.value = state.snap.scope.id;
  $('#scopeTable tbody').innerHTML = state.scopes.map(s => `<tr><td>${s.name}<br><small class="muted">${s.id}</small></td><td><small>${Object.entries(s.filters).map(([k, v]) => `${k}: ${v.length} 件`).join('<br>') || '(個別指定)'}</small></td><td>${s.n_cameras}</td>
    <td>${s.profile && s.profile.trained_at ? `<small>${s.profile.trained_at}<br>精度 ${((s.profile.accuracy_after ?? s.profile.accuracy) * 100).toFixed(1)}% / 該当なし再現 ${(s.profile.none_recall * 100).toFixed(0)}%</small>` : '<small class="muted">未学習</small>'}</td>
    <td><button data-use="${s.id}">使う</button> <button data-del="${s.id}">削除</button></td></tr>`).join('');
}
async function refreshScopes() { state.scopes = await (await fetch('/api/scopes')).json(); renderScopes(); }
async function useScope(id) { const r = await (await fetch('/api/session/scope', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session: state.session, scope_id: id }) })).json(); if (r.state) { applyState(r.state); loadCams(); renderCamTable(); } }
async function loadHierarchy() { state.hier = await (await fetch('/api/hierarchy')).json(); renderReg(); }
function renderReg() {
  const h = state.hier, rg = state.reg;
  $('#bureauList').innerHTML = Object.entries(h).map(([b, ls]) => `<li data-b="${b}" class="${rg.bureau === b ? 'sel' : ''}"><span>${b}</span><small class="muted">${Object.keys(ls).length} 路線</small></li>`).join('');
  const ls = rg.bureau ? h[rg.bureau] : {};
  $('#officeList').innerHTML = Object.entries(ls).map(([n, v]) => `<li data-l="${v.code}" data-n="${n}" class="${rg.lines.has(v.code) ? 'sel' : ''}"><span>${n}</span><small class="muted">${v.n_stations} 駅</small></li>`).join('');
  $('#officeHint').textContent = rg.bureau ? '' : '(事業者を選択)';
  $('#routeList').innerHTML = [...rg.lines.entries()].map(([code, n]) => `<li data-r="${code}"><span>${n}</span><small>×</small></li>`).join('');
  const f = currentFilters(); $('#regPreview').textContent = f && state.all ? `→ ${state.all.filter(c => c.attrs.line_codes.some(x => f.line_codes.includes(x))).length} 駅` : '';
}
function currentFilters() { return state.reg.lines.size ? { line_codes: [...state.reg.lines.keys()] } : null; }
async function saveScope() {
  const f = currentFilters(); if (!f) { alert('路線を選んでください'); return; }
  const name = $('#scopeName').value || [...state.reg.lines.values()].join(' + ');
  const r = await (await fetch('/api/scopes', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ id: $('#scopeId').value || undefined, name, filters: f }) })).json();
  await refreshScopes(); await useScope(r.id);
}

/* ---------------- 事前学習 ---------------- */
async function startPretrain() {
  const seeds = Array.from({ length: +$('#ptSeeds').value }, (_, i) => i + 1);
  const carriers = $('#ptCarriers').value === '1' ? ['{station}'] : ['{station}', '{station}まで'];
  const r = await fetch('/api/pretrain', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session: state.session, scope_id: state.snap.scope.id, seeds, carriers }) });
  const j = await r.json(); if (j.error) { $('#ptStatus').textContent = 'エラー: ' + j.error; return; } $('#ptStatus').textContent = '開始しました…'; $('#ptResult').innerHTML = '';
}
function onPretrain(m) {
  if (m.status === 'running') { $('#ptBar').style.width = (m.step / m.total * 100).toFixed(1) + '%'; $('#ptStatus').textContent = `${m.step}/${m.total}  ${m.text} → ${m.free}  ${m.ok ? '○' : '×'}   累積正解率 ${(m.acc * 100).toFixed(1)}%`; }
  else if (m.status === 'done') { const r = m.result; $('#ptBar').style.width = '100%'; $('#ptStatus').textContent = `完了 (${r.seconds.toFixed(0)} 秒, ${r.n_utts} 発話)`;
    $('#ptResult').innerHTML = `<div class="kv"><span>正解率 (校正前 → 後)</span><b>${(r.accuracy * 100).toFixed(1)}% → ${(r.accuracy_after * 100).toFixed(1)}%</b>
      <span>雑音別</span><b>${Object.entries(r.accuracy_by_snr).map(([k, v]) => `${k} ${(v * 100).toFixed(0)}%`).join(' / ')}</b>
      <span>文法外 → 該当なし</span><b>${(r.none_recall * 100).toFixed(0)}% (誤実行 ${(r.false_accept * 100).toFixed(0)}%)</b>
      <span>校正</span><b>T=${r.calibration.temperature.toFixed(2)} β0=${r.calibration.none_bias.toFixed(2)} β1=${(r.calibration.len_bonus ?? 0).toFixed(2)} γ=${(r.calibration.residual_penalty ?? 0).toFixed(2)}  ECE ${r.ece_before.toFixed(3)} → ${r.ece_after.toFixed(3)}</b>
      <span>追加した読み</span><b>${r.added_readings.length ? r.added_readings.map(a => `${a.label}: ${a.from} → <u>${a.reading}</u>`).join('<br>') : 'なし'}</b>
      <span>混同しやすい駅</span><b>${r.confusions.length ? r.confusions.map(c => `${c.from} → ${c.to} (${c.count})`).join('<br>') : 'なし'}</b>
      <span>警告</span><b>${r.warnings.length ? r.warnings.slice(0, 10).join('<br>') : 'なし'}</b></div>`;
    refreshScopes(); applyState(m.state); renderCamTable(); }
  else if (m.status === 'error') $('#ptStatus').textContent = 'エラー: ' + m.error;
}
function renderCamTable() { $('#camTable').innerHTML = `<table><tr><th>駅</th><th>路線</th><th>読み</th></tr>` + state.cams.slice(0, 300).map(c => `<tr><td>${c.label}</td><td>${c.attrs.route_label}</td><td>${c.readings.map(r => `<span class="rd">${r}</span>`).join(' , ')} <span class="rd" data-add="${c.id}">＋追加</span></td></tr>`).join('') + '</table>'; }

/* ---------------- イベント ---------------- */
$('#micBtn').onclick = toggleMic;
$('#sayForm').onsubmit = async (e) => { e.preventDefault(); const t = $('#sayText').value.trim(); if (t) say(t); };
$('#quick').onclick = (e) => { const b = e.target.closest('button'); if (b) say(b.dataset.say); };
async function say(text) { $('#speech').textContent = '… TTS 合成中'; const snr = $('#saySnr').value; const r = await fetch('/api/say', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session: state.session, text, snr_db: snr ? +snr : null }) }); const j = await r.json(); if (j.error) $('#speech').textContent = 'エラー: ' + j.error; }
$('#stationCard').onclick = (e) => { const b = e.target.closest('button'); if (b) send({ type: 'intent', intent: b.dataset.intent }); };
document.addEventListener('keydown', (e) => { if (e.target.tagName === 'INPUT') return; const map = { Escape: 'back', ArrowRight: 'next_station', ArrowLeft: 'prev_station', '+': 'zoom_in', '=': 'zoom_in', '-': 'zoom_out' }; const it = map[e.key]; if (it && state.snap && state.snap.state !== 'MAP') { e.preventDefault(); send({ type: 'intent', intent: it }); } });
$('#scopeSel').onchange = (e) => useScope(e.target.value);
$('#tabMonitor').onclick = () => { $('#monitor').hidden = false; $('#register').hidden = true; $('#tabMonitor').classList.add('active'); $('#tabRegister').classList.remove('active'); setTimeout(() => map.invalidateSize(), 50); };
$('#tabRegister').onclick = async () => { $('#monitor').hidden = true; $('#register').hidden = false; $('#tabRegister').classList.add('active'); $('#tabMonitor').classList.remove('active'); if (!state.hier) await loadHierarchy(); renderCamTable(); };
$('#bureauList').onclick = (e) => { const li = e.target.closest('li'); if (!li) return; state.reg.bureau = li.dataset.b; renderReg(); };
$('#officeList').onclick = (e) => { const li = e.target.closest('li'); if (!li) return; const c = li.dataset.l; state.reg.lines.has(c) ? state.reg.lines.delete(c) : state.reg.lines.set(c, li.dataset.n); renderReg(); };
$('#routeList').onclick = (e) => { const li = e.target.closest('li'); if (!li) return; state.reg.lines.delete(li.dataset.r); renderReg(); };
$('#saveScope').onclick = saveScope;
$('#scopeTable').onclick = async (e) => { const b = e.target.closest('button'); if (!b) return; if (b.dataset.use) useScope(b.dataset.use); if (b.dataset.del && confirm('削除しますか?')) { await fetch('/api/scopes/' + b.dataset.del, { method: 'DELETE' }); refreshScopes(); } };
$('#pretrainBtn').onclick = startPretrain;
$('#camTable').onclick = async (e) => { const s = e.target.closest('[data-add]'); if (!s) return; const r = prompt('追加する読み (カタカナ)'); if (!r) return; await fetch(`/api/cameras/${s.dataset.add}/readings`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ reading: r }) }); await loadCams(); renderCamTable(); };
connect();
})();
