/* 動画の審判デモ: 動画のアップロード/試験動画 → 窓ごとの確率の時系列、ライブ (カメラ) → フレームごとの判断 */
(() => {
const $ = (s) => document.querySelector(s);
const COLORS = { yes: '#f85149', no: '#3fb950', none: '#8b98a5' };
let checks = [], samplesList = [], current = { file: null, sample: null }, lastRes = null;

// ---- tabs ----
document.querySelectorAll('.tab').forEach(b => b.addEventListener('click', () => {
  document.querySelectorAll('.tab').forEach(x => x.classList.toggle('on', x === b));
  document.querySelectorAll('.tabpane').forEach(p => p.hidden = p.id !== 'tab-' + b.dataset.tab);
  if (b.dataset.tab !== 'live') stopCam();
}));

// ---- checks / scenes ----
fetch('/api/checks').then(r => r.json()).then(d => {
  checks = d.checks;
  const sel = $('#scene');
  if (d.auto_scene) { const o = document.createElement('option'); o.value = 'auto'; o.textContent = '自動 (シーンごとに場面を判定して項目を選ぶ)'; sel.appendChild(o); sel.value = 'auto'; }
  d.scenes.forEach(s => { const o = document.createElement('option'); o.value = s; o.textContent = s + ' (' + checks.filter(c => c.scene === s).length + ')'; sel.appendChild(o); });
  $('#scene option[value=""]').textContent = `全部の項目 (${checks.length})`;
  const tb = $('#checkTable tbody');
  checks.forEach(c => { const tr = document.createElement('tr'); tr.innerHTML = `<td>${c.title}</td><td>${c.scene}${c.sport ? ' / ' + c.sport : ''}</td><td>${c.fps} fps × ${c.window_s} 秒</td><td>${c.risk}</td><td class="muted small">${c.question}</td>`; tb.appendChild(tr); });
});
fetch('/api/samples').then(r => r.json()).then(d => {
  samplesList = d.samples || [];
  if (!samplesList.length) return;
  const box = $('#samples'); box.innerHTML = '';
  const byScene = {};
  samplesList.forEach(s => (byScene[s.title] ||= []).push(s));
  Object.entries(byScene).forEach(([title, arr]) => {
    arr.forEach(s => {
      const el = document.createElement('div'); el.className = 'sample'; el.dataset.file = s.file;
      el.innerHTML = `<img src="/clips/${s.id}.jpg" alt="" loading="lazy"><div class="t">${title}</div><div class="t muted">${s.id}</div>`;
      el.addEventListener('click', () => pickSample(s, el));
      box.appendChild(el);
    });
  });
});

let jafKey = '';
function pickSample(s, el) {
  document.querySelectorAll('.sample').forEach(x => x.classList.toggle('on', x === el));
  if (s.jaf) {
    current = { file: null, sample: 'jaf:' + s.file };
    $('#player').src = '/jaf/' + s.file + '?key=' + encodeURIComponent(jafKey); $('#player').poster = '/jaf/' + s.id + '.jpg?key=' + encodeURIComponent(jafKey);
    $('#status').textContent = s.title + ' (JAF、社内確認用)'; $('#scene').value = 'auto';
  } else {
    current = { file: null, sample: s.file };
    $('#player').src = '/clips/' + s.file; $('#player').poster = '/clips/' + s.id + '.jpg';
    $('#status').textContent = s.title + ' の試験動画';
  }
  $('#judgeBtn').disabled = false; $('#answer').hidden = true;
  const sc = checks.find(c => c.key === s.check)?.scene; if (sc && $('#scene').value !== 'auto') $('#scene').value = sc;
  $('#picked').hidden = false; $('#picked').textContent = `選択中: ${s.title} / ${s.id}`;
}
$('#toTop')?.addEventListener('click', () => $('#player').scrollIntoView({ behavior: 'smooth', block: 'start' }));
// 社内確認用の実写 (JAF): パスワードを入れると一覧の先頭に出る
$('#jafBtn')?.addEventListener('click', async () => {
  const k = prompt('JAF 動画 (社内確認用) のパスワード'); if (!k) return;
  const r = await fetch('/api/jaf/samples?key=' + encodeURIComponent(k)); const d = await r.json();
  if (d.error) { alert(d.error); return; }
  jafKey = k;
  const box = $('#samples');
  document.querySelectorAll('.sample.jaf').forEach(x => x.remove());
  d.samples.slice().reverse().forEach(s => {
    const el = document.createElement('div'); el.className = 'sample jaf'; el.dataset.file = s.file;
    el.innerHTML = `<img src="/jaf/${s.id}.jpg?key=${encodeURIComponent(k)}" alt="" loading="lazy"><div class="t">JAF ${s.title}</div><div class="t muted">${s.id} (実写・社内確認)</div>`;
    el.addEventListener('click', () => pickSample({ ...s, jaf: true }, el));
    box.prepend(el);
  });
  $('#jafBtn').textContent = `JAF ${d.samples.length} 本を表示中`;
  box.scrollTop = 0;
});
$('#file').addEventListener('change', (e) => {
  const f = e.target.files[0]; if (!f) return;
  current = { file: f, sample: null }; document.querySelectorAll('.sample').forEach(x => x.classList.remove('on'));
  $('#player').src = URL.createObjectURL(f); $('#judgeBtn').disabled = false; $('#answer').hidden = true;
  $('#status').textContent = f.name + ' (' + (f.size / 1e6).toFixed(1) + ' MB)';
});

// ---- judge ----
$('#judgeBtn').addEventListener('click', async () => {
  const fd = new FormData();
  if (current.sample) fd.append('sample', current.sample); else if (current.file) fd.append('file', current.file); else return;
  fd.append('scene', $('#scene').value); fd.append('mode', $('#mode').value); if (jafKey) fd.append('key', jafKey);
  $('#judgeBtn').disabled = true; $('#status').textContent = '判定中… (動画の長さの 1/5 程度の時間)'; $('#results').innerHTML = '<div class="muted">判定中…</div>';
  const t0 = performance.now();
  try {
    const r = await fetch('/api/judge', { method: 'POST', body: fd }); const d = await r.json();
    if (d.error) throw new Error(d.error);
    lastRes = d; render(d); $('#status').textContent = `完了 ${(performance.now() - t0) / 1000 | 0} 秒`;
    if (current.sample) showAnswer(current.sample, d);
  } catch (e) { $('#status').textContent = 'エラー: ' + e.message; }
  $('#judgeBtn').disabled = false;
});

function render(d) {
  const tm = d.timing;
  $('#timings').textContent = `動画 ${d.duration.toFixed(1)} 秒 → 窓 ${tm.windows} 個 / 質問 ${tm.asks} 回 | 読み込み ${tm.decode_s.toFixed(2)}s + 符号化 ${tm.encode_s.toFixed(2)}s + 質問 ${tm.ask_s.toFixed(2)}s = ${tm.total_s.toFixed(2)}s (${(d.duration / tm.total_s).toFixed(1)}x リアルタイム) | ${d.model}`;
  $('#resInfo').textContent = `${d.summary.length} 項目 (${d.mode === 'head' ? '学習 head' : '質問方式'})`;
  const res = $('#results'); res.innerHTML = '';
  d.summary.forEach(it => {
    const segs = it.segments.map(s => `${s[0].toFixed(0)}〜${s[1].toFixed(0)} 秒`).join(', ');
    const el = document.createElement('div'); el.className = 'card';
    el.innerHTML = `<div class="head"><b>${it.title}</b><span class="lv ${it.label}">${it.label}</span></div>
      <div class="sub">「はい」の最大 ${(it.p_max * 100).toFixed(0)}% (${it.t_max.toFixed(0)} 秒付近)${segs ? ' / 該当: ' + segs : ''} / 危険度 ${it.risk}</div>`;
    el.addEventListener('click', () => { $('#player').currentTime = it.t_max; $('#player').play(); });
    res.appendChild(el);
  });
  // timeline strips (時刻に比例した絶対配置。窓は半分重なるので後の窓を上に描く。判定対象外の窓はグレー)
  const tl = $('#timeline'); tl.innerHTML = '';
  const dur = d.duration || 1;
  const scenes = d.scenes || [];
  if (scenes.length > 1 || (scenes[0] && scenes[0].type_ja)) {
    const row = document.createElement('div'); row.className = 'trow scenes';
    const strip = document.createElement('div'); strip.className = 'strip sc';
    scenes.forEach((sc, k) => { const b = document.createElement('b'); b.style.left = (sc.t0 / dur * 100) + '%'; b.style.width = (Math.max(0.2, sc.t1 - sc.t0) / dur * 100) + '%';
      b.textContent = sc.type_ja || `シーン ${k + 1}`; b.title = `${sc.type_ja ? 'この区間の場面: ' + sc.type_ja + ' / ' : ''}${sc.t0.toFixed(1)}〜${sc.t1.toFixed(1)} 秒 (${{ start: '開始', cut: 'カット', camera: 'カメラの動き (PTZ)' }[sc.kind] || sc.kind})`;
      b.addEventListener('click', () => { $('#player').currentTime = sc.t0; $('#player').play(); }); strip.appendChild(b); });
    const cur = document.createElement('div'); cur.className = 'cursor'; strip.appendChild(cur);
    row.innerHTML = `<div class="name">シーン (${scenes.length})</div>`; row.appendChild(strip); row.appendChild(document.createElement('div'));
    tl.appendChild(row);
  }
  d.summary.forEach(it => {
    const s = d.series[it.key]; if (!s) return;
    const row = document.createElement('div'); row.className = 'trow' + (it.action !== 'reject' ? ' hot' : '');
    const strip = document.createElement('div'); strip.className = 'strip';
    s.windows.forEach(w => { const i = document.createElement('i'); const p = w.p[0];
      i.style.left = (w.t0 / dur * 100) + '%'; i.style.width = (Math.max(0.2, w.t1 - w.t0) / dur * 100) + '%';
      i.style.background = w.skipped ? 'rgba(139,152,165,0.25)' : `rgba(248,81,73,${(0.08 + p * 0.92).toFixed(2)})`;
      if (w.long) i.classList.add('long');
      i.title = `${w.long ? '[文脈窓] ' : ''}${w.t0.toFixed(1)}〜${w.t1.toFixed(1)} 秒: ` + (w.skipped ? '場面が違うので判定対象外' : `はい ${(p * 100).toFixed(0)}% / いいえ ${(w.p[1] * 100).toFixed(0)}% / 判別できない ${(w.p[2] * 100).toFixed(0)}%`);
      i.addEventListener('click', () => { $('#player').currentTime = w.t0; $('#player').play(); }); strip.appendChild(i); });
    scenes.slice(1).forEach(sc => { const m = document.createElement('s'); m.style.left = (sc.t0 / dur * 100) + '%'; strip.appendChild(m); });
    const cur = document.createElement('div'); cur.className = 'cursor'; strip.appendChild(cur);
    row.innerHTML = `<div class="name" title="${it.title}">${it.title}</div>`; row.appendChild(strip);
    const pm = document.createElement('div'); pm.className = 'pmax'; pm.textContent = it.label === '該当シーンなし' ? '—' : (it.p_max * 100).toFixed(0) + '%'; row.appendChild(pm);
    tl.appendChild(row);
  });
  $('#player').ontimeupdate = () => { const x = ($('#player').currentTime / dur * 100) + '%'; document.querySelectorAll('.strip .cursor').forEach(c => c.style.left = x); };
}

async function showAnswer(file, d) {
  const a = await (await fetch('/api/samples/answer?file=' + encodeURIComponent(file))).json();
  if (!a || a.label === undefined) return;
  const it = d.summary.find(x => x.key === a.check);
  const flagged = it && (it.action === 'execute' || it.action === 'confirm');
  const pos = checks.find(c => c.key === a.check)?.positive;
  const ok = (a.label === 1) === flagged;
  const el = $('#answer'); el.hidden = false; el.className = 'answer ' + (ok ? 'ok' : 'ng');
  el.innerHTML = `<b>正解:</b> この動画は「${it ? it.title : a.check}」が <b>${a.label === 1 ? `ある (${a.event_start}〜${a.event_end} 秒)` : 'ない'}</b>。判定は「${it ? it.label : '-'}」→ ${ok ? '一致' : '不一致'}。`
    + (a.suspect ? `<br><span class="muted small">※ お題どおりに描けていない疑いのある生成動画 (別モデルの監査で事象を確認できず)。正解ラベルの方が間違っている可能性があります。</span>` : '');
}

// ---- live ----
let stream = null, liveTimer = null, busy = false;
async function startCam() {
  stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: $('#facing').value, width: { ideal: 640 } }, audio: false });
  const v = $('#cam'); v.srcObject = stream; await v.play();
  $('#camBtn').textContent = '■ 停止'; $('#camBtn').classList.add('on'); $('#navhud').hidden = $('#liveMode').value !== 'nav';
  schedule();
}
function stopCam() { if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; } clearTimeout(liveTimer); $('#camBtn').textContent = '📷 カメラ開始'; $('#camBtn').classList.remove('on'); }
$('#camBtn').addEventListener('click', () => stream ? stopCam() : startCam().catch(e => alert('カメラを開けません: ' + e.message)));
$('#liveMode').addEventListener('change', () => { $('#navhud').hidden = $('#liveMode').value !== 'nav'; });
function schedule() { if (!stream) return; liveTimer = setTimeout(tick, 1000 / Number($('#liveFps').value)); }
async function tick() {
  if (!stream) return;
  if (busy) return schedule();
  busy = true;
  const v = $('#cam'), c = $('#grab'); c.width = 512; c.height = Math.round(512 * v.videoHeight / Math.max(1, v.videoWidth)) || 288;
  c.getContext('2d').drawImage(v, 0, 0, c.width, c.height);
  const t0 = performance.now();
  try {
    const r = await fetch('/api/live', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ frames: [c.toDataURL('image/jpeg', 0.8)], mode: $('#liveMode').value }) });
    const d = await r.json(); renderLive(d, performance.now() - t0);
  } catch (e) { $('#liveTimings').textContent = 'エラー: ' + e.message; }
  busy = false; schedule();
}
const ARROWS = { 'turn left': ['◀', '左へ'], 'turn right': ['▶', '右へ'], 'go straight': ['▲', '直進'], 'stop': ['■', '停止'] };
function renderLive(d, rtt) {
  $('#liveTimings').textContent = `往復 ${rtt.toFixed(0)} ms (符号化 ${d.timing.encode_ms.toFixed(0)} + 質問 ${d.timing.ask_ms.toFixed(0)} ms)`;
  const box = $('#liveResults'); box.innerHTML = '';
  d.items.forEach(it => {
    const el = document.createElement('div'); el.className = 'card';
    const bars = it.p.map((p, i) => `<i style="flex:${Math.max(p, 0.001)};background:${i === it.top ? '#4cc2ff' : '#2f3d4d'}"></i>`).join('');
    const opts = it.options.map((o, i) => `<span${i === it.top ? ' style="color:#e6edf3;font-weight:700"' : ''}>${o} ${(it.p[i] * 100).toFixed(0)}%</span>`).join('');
    const lv = { execute: '確定', confirm: '要確認', reject: '不明', none: '該当なし' }[it.action];
    el.innerHTML = `<div class="head"><b>${it.title}</b><span class="lv ${it.action}">${lv}</span></div><div class="bars">${bars}</div><div class="opts">${opts}</div>`;
    box.appendChild(el);
  });
  const nav = d.items.find(i => i.key === 'nav');
  if (nav && !$('#navhud').hidden) {
    const key = ['turn left', 'turn right', 'go straight', 'stop'][nav.top];
    // 確信が低ければ停止に倒す
    const conf = nav.p[nav.top]; const use = (nav.action === 'execute' || nav.action === 'confirm') ? key : 'stop';
    const [sym, lab] = ARROWS[use]; $('#arrow').textContent = sym; $('#navlabel').textContent = `${lab} (${(conf * 100).toFixed(0)}%)`;
    $('#arrow').style.color = use === 'stop' ? '#f85149' : '#4cc2ff';
  }
}
})();
