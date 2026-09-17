/* テキストの判断デモ: 状態 + 質問 → 質問ごとの確率 */
(() => {
const $ = (s) => document.querySelector(s);
const st = { presets: [], qs: [], busy: false };
const COLORS = ['#4cc2ff', '#3fb950', '#d29922', '#f85149', '#a371f7', '#79c0ff', '#ff7b72', '#56d364', '#e3b341', '#f778ba'];

/* ---------------- 質問の編集 ---------------- */
function critToText(q) {
  if (q.type === 'noul') return q.criteria ? `true: ${q.criteria.true || ''}\nfalse: ${q.criteria.false || ''}` : '';
  if (q.type === 'score') return (q.criteria || []).join('\n');
  return Object.entries(q.criteria || {}).map(([k, v]) => `${k}: ${v ?? ''}`).join('\n');
}
function textToCrit(type, text) {
  const lines = text.split('\n').map(l => l.trim()).filter(Boolean);
  if (type === 'score') return lines.map(l => l.replace(/^[^:：]*[:：]\s*/, '') || l);
  if (type === 'noul') { const o = {}; for (const l of lines) { const m = l.match(/^(true|false)\s*[:：]\s*(.*)$/i); if (m) o[m[1].toLowerCase()] = m[2]; } return Object.keys(o).length ? o : null; }
  const o = {}; for (const l of lines) { const m = l.match(/^([^:：]+)[:：]?\s*(.*)$/); if (m) o[m[1].trim()] = m[2] || null; } return o;
}
function renderQuestions() {
  $('#qCount').textContent = `(${st.qs.length} 問)`;
  $('#questions').innerHTML = st.qs.map((q, i) => `<div class="qrow" data-i="${i}">
      <select class="qtype">${['noul', 'choice', 'score'].map(t => `<option ${q.type === t ? 'selected' : ''}>${t}</option>`).join('')}</select>
      <div><input class="qinst" value="${(q.instructions || '').replace(/"/g, '&quot;')}" placeholder="質問文 (例: どの部署が対応すべきか)">
        <textarea class="qcrit" placeholder="${q.type === 'score' ? '段階を 1 行ずつ (低い順)' : q.type === 'noul' ? 'true: …\nfalse: …（任意）' : '選択肢を「id: 説明」で 1 行ずつ'}">${critToText(q)}</textarea></div>
      <button class="del" title="削除">✕</button></div>`).join('');
}
function collect() {
  return [...document.querySelectorAll('.qrow')].map((row, i) => {
    const type = row.querySelector('.qtype').value;
    const instructions = row.querySelector('.qinst').value;
    const criteria = textToCrit(type, row.querySelector('.qcrit').value);
    return { key: st.qs[i]?.key || `q${i + 1}`, type, instructions, criteria };
  }).filter(q => q.instructions.trim());
}
$('#questions').addEventListener('click', (e) => { const b = e.target.closest('.del'); if (!b) return; st.qs = collect(); st.qs.splice(+b.closest('.qrow').dataset.i, 1); renderQuestions(); });
$('#questions').addEventListener('change', (e) => { if (e.target.classList.contains('qtype')) { st.qs = collect(); renderQuestions(); } });
$('#addQ').onclick = () => { st.qs = collect(); st.qs.push({ key: `q${st.qs.length + 1}`, type: 'noul', instructions: '', criteria: null }); renderQuestions(); };

/* ---------------- 実行 ---------------- */
async function decide() {
  if (st.busy) return; st.busy = true;
  const mode = document.querySelector('input[name=mode]:checked').value;
  $('#answers').innerHTML = '<div class="muted">判断中…</div>'; $('#wall').textContent = '';
  try {
    const body = { state: $('#state').value, questions: collect(), mode, model: $('#model').value };
    const r = await (await fetch('/api/decide', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })).json();
    if (r.error) { $('#answers').innerHTML = `<div class="muted">エラー: ${r.error}</div>`; return; }
    render(r);
  } finally { st.busy = false; }
}
function render(r) {
  const per = r.results.map(x => x.latency_ms);
  $('#wall').textContent = `${r.results.length} 問を ${r.wall_ms.toFixed(0)} ms (1 問あたり中央 ${median(per).toFixed(0)} ms, ${r.mode === 'json' ? '1 回の JSON 生成' : '並列'})`;
  $('#answers').innerHTML = r.results.map(x => {
    const a = x.answer; const probs = x.probs; const ids = x.options.map(o => o.id);
    const ans = a.type === 'noul' ? `${(a.noul * 100).toFixed(0)}% はい`
      : a.type === 'score' ? `${a.score.toFixed(2)} / ${ids.length - 1}　(${x.options[Math.round(a.score)]?.description || ''})`
      : `${a.choice}${x.options.find(o => o.id === a.choice)?.description ? ' — ' + x.options.find(o => o.id === a.choice).description : ''}`;
    const bars = probs.map((p, i) => `<i style="width:${(p * 100).toFixed(1)}%;background:${COLORS[i % COLORS.length]}" title="${ids[i]} ${(p * 100).toFixed(0)}%"></i>`).join('');
    const opts = probs.map((p, i) => `<span><b style="color:${COLORS[i % COLORS.length]}">■</b> ${ids[i]} ${(p * 100).toFixed(0)}%</span>`).filter((_, i) => probs[i] >= 0.02).join('');
    return `<div class="acard"><div class="ahead"><span class="q">${x.question}</span><span class="ans">${ans}<span class="lv ${x.level}">${x.level}</span></span></div>
      <div class="bars">${bars}</div><div class="opts">${opts}</div></div>`;
  }).join('');
}
const median = (a) => { const b = [...a].sort((x, y) => x - y); return b.length ? b[Math.floor(b.length / 2)] : 0; };

/* ---------------- 質問数と所要時間 ---------------- */
$('#fanBtn').onclick = async () => {
  $('#fanout').innerHTML = '<div class="muted">計測中… (8 問まで両方式で実行します)</div>';
  const body = { state: $('#state').value, questions: collect(), model: $('#model').value };
  const r = await (await fetch('/api/fanout', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })).json();
  if (r.error) { $('#fanout').textContent = 'エラー: ' + r.error; return; }
  const max = Math.max(...r.points.flatMap(p => [p.prob_ms, p.json_ms]));
  $('#fanout').innerHTML = r.points.map(p => `<div class="fanrow"><span class="muted">${p.n} 問</span><div class="fanbars">
      <div class="fanbar p"><i style="width:${(p.prob_ms / max * 100).toFixed(1)}%"></i><span>確率 ${p.prob_ms.toFixed(0)} ms</span></div>
      <div class="fanbar j"><i style="width:${(p.json_ms / max * 100).toFixed(1)}%"></i><span>JSON 生成 ${p.json_ms.toFixed(0)} ms</span></div></div></div>`).join('');
};

/* ---------------- 初期化 ---------------- */
(async () => {
  const m = await (await fetch('/api/models')).json();
  $('#model').innerHTML = m.models.map(x => `<option ${x === m.default ? 'selected' : ''}>${x}</option>`).join('');
  st.presets = await (await fetch('/api/presets')).json();
  $('#preset').innerHTML = st.presets.map((p, i) => `<option value="${i}">${p.name}</option>`).join('');
  loadPreset(0);
  $('#preset').onchange = (e) => loadPreset(+e.target.value);
  $('#runBtn').onclick = decide;
  document.querySelectorAll('input[name=mode]').forEach(el => el.onchange = decide);
})();
function loadPreset(i) { const p = st.presets[i]; $('#state').value = p.state; st.qs = JSON.parse(JSON.stringify(p.questions)); renderQuestions(); $('#answers').innerHTML = '<div class="muted">「判断する」を押すと、ここに質問ごとの確率が出ます。</div>'; $('#wall').textContent = ''; }
})();
