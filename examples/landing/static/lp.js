/* ランディング: ヒーローの確率バーと生成AIのタイピングを交互に見せる + スクロールで要素を出す */
(() => {
const $ = (s) => document.querySelector(s);
const SCENES = [
  { q: 'どの部署が対応すべきか', ms: 33, opts: [['配送・物流', .92], ['請求・返金', .05], ['返品', .02], ['技術的な不具合', .01]],
    typing: '{"department": "配送・物流", "reason": "お客様は注文した商品が…' },
  { q: '対応の緊急度 (3 段階)', ms: 31, opts: [['翌営業日でよい', .04], ['当日中に返す', .21], ['いますぐ', .75]],
    typing: 'このメールは緊急度が高いと考えられます。理由として…' },
  { q: '声で選ばれた地点は', ms: 48, opts: [['栗橋水位', .97], ['栗橋水衝部', .02], ['該当なし', .01]],
    typing: 'ユーザーは「くりはし」と言ったようです。候補としては…' },
  { q: '写真の人物の年齢層', ms: 18, opts: [['30〜39 歳', .52], ['40〜49 歳', .30], ['20〜29 歳', .13], ['50〜59 歳', .05]],
    typing: 'この写真の人物はおそらく30代から40代くらいに見えます…' },
];
let i = 0, typeTimer = null;
function show(s) {
  $('#pcQ').textContent = s.q; $('#pcTime').textContent = s.ms + ' ms';
  const top = Math.max(...s.opts.map(o => o[1]));
  $('#pcBadge').textContent = top >= .85 ? '確定' : top >= .5 ? '要確認' : '不明';
  $('#pcBadge').style.cssText = top >= .85 ? '' : top >= .5
    ? 'background:rgba(210,153,34,.15);color:#d29922' : 'background:rgba(248,81,73,.15);color:#f85149';
  $('#pcBars').innerHTML = s.opts.map(([label, p], k) => `<div class="pcBar ${k ? 'dim' : ''}">
      <span>${label}</span><div class="track"><div class="fill" data-w="${(p * 100).toFixed(0)}"></div></div><span class="pct">${(p * 100).toFixed(0)}%</span></div>`).join('');
  requestAnimationFrame(() => document.querySelectorAll('.fill').forEach(f => f.style.width = f.dataset.w + '%'));
  clearInterval(typeTimer); const t = s.typing; let n = 0; $('#typeText').textContent = '';
  typeTimer = setInterval(() => { n++; $('#typeText').textContent = t.slice(0, n); if (n >= t.length) clearInterval(typeTimer); }, 42);
}
show(SCENES[0]);
setInterval(() => { i = (i + 1) % SCENES.length; show(SCENES[i]); }, 4200);

const io = new IntersectionObserver((es) => es.forEach(e => { if (e.isIntersecting) { e.target.classList.add('on'); io.unobserve(e.target); } }), { threshold: .15 });
document.querySelectorAll('.reveal').forEach((el, k) => { el.style.transitionDelay = (k % 4) * 70 + 'ms'; io.observe(el); });
})();
