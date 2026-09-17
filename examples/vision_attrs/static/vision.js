/* 顔と全身の属性デモ: カメラのフレームをサーバーに送り、確率つきの答えを描く */
(() => {
const $ = (s) => document.querySelector(s);
const video = $('#video'), frame = $('#frame'), overlay = $('#overlay');
const st = { stream: null, timer: null, busy: false, bodyBox: null, drag: null, lastImg: null };
const COLORS = ['#4cc2ff', '#3fb950', '#d29922', '#f85149', '#a371f7', '#79c0ff', '#ff7b72', '#56d364', '#e3b341'];

fetch('/api/info').then(r => r.json()).then(i => {
  $('#faceInfo').textContent = i.face ? `${i.face.ckpt}: 凍結 ${i.face.frozen_M}M + head ${i.face.trainable_K}K` : '(未ロード)';
  $('#bodyInfo').textContent = i.body ? `${i.body.ckpt}: 凍結 ${i.body.frozen_M}M + head ${i.body.trainable_K}K` : '(未ロード)';
});

async function startCam() {
  if (st.stream) { stopCam(); return; }
  try { st.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: $('#facing').value, width: { ideal: 1280 }, height: { ideal: 960 } }, audio: false }); }
  catch (e) { alert('カメラを開けません (https か localhost が必要): ' + e.message); return; }
  video.srcObject = st.stream; video.hidden = false; await video.play();
  $('#camBtn').textContent = '⏹ カメラ停止'; $('#camBtn').classList.add('on');
  if ($('#live').checked) st.timer = setInterval(() => { if (!st.busy) analyzeFrame(); }, 1500);
}
function stopCam() { if (st.timer) clearInterval(st.timer); st.timer = null; if (st.stream) st.stream.getTracks().forEach(t => t.stop()); st.stream = null; video.hidden = true; $('#camBtn').textContent = '📷 カメラ開始'; $('#camBtn').classList.remove('on'); }
$('#live').onchange = () => { if (st.timer) { clearInterval(st.timer); st.timer = null; } if (st.stream && $('#live').checked) st.timer = setInterval(() => { if (!st.busy) analyzeFrame(); }, 1500); };

function grab() {
  const w = video.videoWidth || 640, h = video.videoHeight || 480; frame.width = w; frame.height = h; overlay.width = w; overlay.height = h;
  frame.getContext('2d').drawImage(video, 0, 0, w, h); video.hidden = true;
  return frame.toDataURL('image/jpeg', 0.85);
}
async function analyzeFrame() { if (!st.stream) return; const data = grab(); video.hidden = st.stream ? true : true; await analyze(data); }
async function analyze(dataUrl) {
  st.busy = true; st.lastImg = dataUrl;
  try {
    const r = await fetch('/api/analyze', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ image_base64: dataUrl, body_box: st.bodyBox }) });
    const j = await r.json(); if (j.error) { $('#hint').textContent = j.error; return; }
    render(j);
  } finally { st.busy = false; }
}
const hist = { faces: [], body: [] };   // 平滑化用 (連続判定で揺れを抑える)
function iou(a, b) { const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]), x2 = Math.min(a[0] + a[2], b[0] + b[2]), y2 = Math.min(a[1] + a[3], b[1] + b[3]); const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1); return inter / (a[2] * a[3] + b[2] * b[3] - inter); }
function smoothAttr(prevList, attr) {
  const lists = [attr, ...prevList]; const keys = Object.keys(attr.probabilities); const avg = {};
  for (const k of keys) avg[k] = lists.reduce((s, a) => s + (a.probabilities[k]?.p ?? 0), 0) / lists.length;
  const out = JSON.parse(JSON.stringify(attr)); for (const k of keys) out.probabilities[k].p = avg[k];
  const best = keys.reduce((a, b) => avg[a] >= avg[b] ? a : b); out.choice = best; out.choice_ja = attr.range ? attr.choice_ja : attr.probabilities[best].label;
  out.confidence = attr.range ? attr.confidence : avg[best]; return out;
}
function smooth(j) {
  if (!st.stream) { hist.faces = []; hist.body = []; return j; }   // 静止画は平滑化しない
  const N = 4;
  j.faces = j.faces.map(f => { const h = hist.faces.find(x => iou(x.box, f.box) > 0.4); const prev = h ? h.items : []; const sm = { ...f };
    for (const k of ['age', 'gender']) if (f[k]) sm[k] = smoothAttr(prev.map(p => p[k]).filter(Boolean), f[k]);
    const items = [f, ...prev].slice(0, N); if (h) { h.box = f.box; h.items = items; } else hist.faces.push({ box: f.box, items }); return sm; });
  hist.faces = hist.faces.filter(h => j.faces.some(f => iou(h.box, f.box) > 0.4));
  if (j.body) { const prev = hist.body; const sm = { ...j.body }; for (const k of ['gender', 'age', 'orientation', 'carrying']) if (j.body[k]) sm[k] = smoothAttr(prev.map(p => p[k]).filter(Boolean), j.body[k]); hist.body = [j.body, ...prev].slice(0, N); j.body = sm; }
  return j;
}
function render(j) {
  j = smooth(j);
  const ctx = overlay.getContext('2d'); ctx.clearRect(0, 0, overlay.width, overlay.height); ctx.lineWidth = Math.max(2, overlay.width / 300); ctx.font = `${Math.max(12, overlay.width / 45)}px system-ui, sans-serif`;
  if (j.body) { const b = j.body.box; ctx.strokeStyle = 'rgba(210,153,34,.9)'; ctx.setLineDash([8, 6]); ctx.strokeRect(b[0], b[1], b[2], b[3]); ctx.setLineDash([]); }
  j.faces.forEach((f, i) => { const [x, y, w, h] = f.box; ctx.strokeStyle = COLORS[i % COLORS.length]; ctx.strokeRect(x, y, w, h);
    const label = `${f.age.choice_ja} / ${f.gender.choice_ja}`; const tw = ctx.measureText(label).width + 10; ctx.fillStyle = 'rgba(11,15,20,.8)'; ctx.fillRect(x, y - 22, tw, 22); ctx.fillStyle = COLORS[i % COLORS.length]; ctx.fillText(label, x + 5, y - 6); });
  $('#faces').innerHTML = j.faces.length ? j.faces.map((f, i) => card(`顔 ${i + 1}`, COLORS[i % COLORS.length], ['age', 'gender'].map(k => f[k]))).join('') : '<div class="muted">顔が見つかりません (正面・明るい場所で)</div>';
  $('#body').innerHTML = j.body ? card((st.bodyBox ? '指定した範囲' : 'フレーム全体') + (j.body.reference_only ? ' <span class="muted small">— 全身が写っていないので参考値</span>' : ''), '#d29922', ['gender', 'age', 'orientation', 'carrying'].map(k => j.body[k]).filter(Boolean)) : '';
  $('#timings').textContent = `顔検出 ${j.detect_ms ?? 0}ms / 顔 ${j.face_ms ?? 0}ms / 全身 ${j.body_ms ?? 0}ms / 合計 ${j.total_ms}ms`;
  $('#hint').textContent = j.faces.length ? `${j.faces.length} 人の顔。全身は${st.bodyBox ? '指定範囲' : 'フレーム全体'}を判定 (人物が枚 1 人で全身が入ると精度が出ます。枠をドラッグで指定できます)` : '顔なし。全身の判定のみ表示しています';
}
function card(title, color, attrs) {
  return `<div class="card"><div class="head"><b style="color:${color}">${title}</b></div>` + attrs.map(a => {
    const entries = Object.entries(a.probabilities); const bars = entries.map(([id, v], k) => `<i style="width:${(v.p * 100).toFixed(1)}%;background:${COLORS[k % COLORS.length]}" title="${v.label} ${(v.p * 100).toFixed(0)}%"></i>`).join('');
    const head = a.range ? `${a.choice_ja} <span class="muted small">(${(a.confidence * 100).toFixed(0)}%、推定 ${a.range.expected} 歳前後)</span>` : `${a.choice_ja} <span class="muted small">${(a.confidence * 100).toFixed(0)}%</span>`;
    return `<div class="attr"><span>${a.title}</span><div class="bars">${bars}</div><span class="ans"><span class="lv ${a.level}">${a.level}</span></span></div>
            <div class="attr" style="margin-top:-2px"><span></span><span class="muted small">${entries.filter(([, v]) => v.p >= 0.08).map(([, v]) => `${v.label} ${(v.p * 100).toFixed(0)}%`).join(' · ')}</span><span class="ans"><b>${head}</b></span></div>`; }).join('') + '</div>';
}
/* 全身の範囲をドラッグで指定 */
overlay.addEventListener('pointerdown', (e) => { const r = overlay.getBoundingClientRect(); const sx = overlay.width / r.width, sy = overlay.height / r.height; st.drag = [(e.clientX - r.left) * sx, (e.clientY - r.top) * sy]; });
overlay.addEventListener('pointerup', (e) => { if (!st.drag) return; const r = overlay.getBoundingClientRect(); const sx = overlay.width / r.width, sy = overlay.height / r.height; const x2 = (e.clientX - r.left) * sx, y2 = (e.clientY - r.top) * sy;
  const [x1, y1] = st.drag; st.drag = null; const w = Math.abs(x2 - x1), h = Math.abs(y2 - y1);
  st.bodyBox = (w > 30 && h > 30) ? [Math.round(Math.min(x1, x2)), Math.round(Math.min(y1, y2)), Math.round(w), Math.round(h)] : null;
  if (st.lastImg) analyze(st.lastImg); });
$('#camBtn').onclick = startCam;
$('#shotBtn').onclick = () => { if (st.stream) analyzeFrame(); else if (st.lastImg) analyze(st.lastImg); };
$('#file').onchange = (e) => { const f = e.target.files[0]; if (!f) return; stopCam(); const rd = new FileReader(); rd.onload = () => { const img = new Image(); img.onload = () => { frame.width = img.width; frame.height = img.height; overlay.width = img.width; overlay.height = img.height; frame.getContext('2d').drawImage(img, 0, 0); st.bodyBox = null; analyze(frame.toDataURL('image/jpeg', 0.9)); }; img.src = rd.result; }; rd.readAsDataURL(f); };
$('#facing').onchange = () => { if (st.stream) { stopCam(); startCam(); } };
})();
