/* 端末内で動かすときの発話区間の切り出し (エネルギー方式).
 *
 * サーバーに送るデモは Silero VAD (openvons/voice/vad.py) をサーバー側で回しているが、
 * 端末内だけで完結する経路 (examples/ondevice、スマホアプリ) では持てないので、
 * 32ms ごとの RMS で「静かなときの底」を追いかけ、その何倍かを超えたら発話とみなす。
 *
 * ハンズフリーのための最小限:
 *   - 一度マイクを入れたら、話し終わって無音が続いたところで 1 発話として切り出す
 *   - 切り出したら呼び出し側に渡して、そのまま次の発話を待ち続ける
 *   - 判断中に話されたぶんも取りこぼさない (呼び出し側が順に処理する)
 *
 * 使い方:
 *   const vad = makeVad({ onUtterance: (f32) => decide(f32) });
 *   vad.feed(float32ChunkAt16k);   // ScriptProcessor / AudioWorklet から
 *   vad.reset();
 */
export function makeVad(opts = {}) {
  const cfg = {
    sampleRate: 16000,
    frame: 512,             // 32ms
    startFactor: 3.0,       // 底の何倍で発話開始とみなすか
    endFactor: 1.8,         // 何倍を下回ったら無音とみなすか (開始より低くしてぶれを防ぐ)
    absFloor: 0.004,        // 完全な無音環境で雑音を拾わないための下限
    endSilenceMs: 500,      // 話し終わりと判断するまでの無音
    minSpeechMs: 250,       // これより短い音は無視 (咳やクリック)
    prerollMs: 300,         // 発話開始の前に足す音 (立ち上がりが切れないように)
    maxUttSec: 8,           // 長すぎるものは強制的に切る
    onUtterance: () => {},
    onState: () => {},      // 'idle' | 'speech' を通知 (画面表示用)
    ...opts,
  };
  const framesPerSec = cfg.sampleRate / cfg.frame;
  const preFrames = Math.ceil((cfg.prerollMs / 1000) * framesPerSec);
  const endFrames = Math.ceil((cfg.endSilenceMs / 1000) * framesPerSec);
  const minFrames = Math.ceil((cfg.minSpeechMs / 1000) * framesPerSec);
  const maxFrames = Math.ceil(cfg.maxUttSec * framesPerSec);

  let carry = new Float32Array(0);
  let floor = 0.01;          // 静かなときの RMS。ゆっくり下げ、上げるのは速く
  let speaking = false;
  let pre = [];              // プリロール用のリングバッファ
  let utt = [];
  let silent = 0;
  let voiced = 0;

  function rms(x) {
    let s = 0;
    for (let i = 0; i < x.length; i++) s += x[i] * x[i];
    return Math.sqrt(s / x.length);
  }

  function endUtterance() {
    const frames = pre.concat(utt);
    speaking = false;
    utt = [];
    silent = 0;
    const n = frames.reduce((a, f) => a + f.length, 0);
    cfg.onState('idle');
    if (voiced < minFrames) { voiced = 0; return; }        // 短すぎる音は捨てる
    voiced = 0;
    const out = new Float32Array(n);
    let o = 0;
    for (const f of frames) { out.set(f, o); o += f.length; }
    cfg.onUtterance(out);
  }

  return {
    /** 16kHz の Float32 を好きな長さで渡す。内部で 32ms ごとに見る。 */
    feed(chunk) {
      const buf = new Float32Array(carry.length + chunk.length);
      buf.set(carry, 0); buf.set(chunk, carry.length);
      let off = 0;
      while (off + cfg.frame <= buf.length) {
        const f = buf.subarray(off, off + cfg.frame);
        off += cfg.frame;
        const e = rms(f);
        const startTh = Math.max(floor * cfg.startFactor, cfg.absFloor);
        const endTh = Math.max(floor * cfg.endFactor, cfg.absFloor * 0.7);
        if (!speaking) {
          // 無音のあいだだけ底を更新する (発話中に上げてしまわない)
          floor = e < floor ? floor * 0.9 + e * 0.1 : floor * 0.995 + e * 0.005;
          pre.push(new Float32Array(f));
          if (pre.length > preFrames) pre.shift();
          if (e > startTh) {
            speaking = true; utt = []; silent = 0; voiced = 1;
            cfg.onState('speech');
          }
        } else {
          utt.push(new Float32Array(f));
          if (e > endTh) { silent = 0; voiced++; } else { silent++; }
          if (silent >= endFrames || utt.length >= maxFrames) endUtterance();
        }
      }
      carry = buf.slice(off);
    },
    reset() {
      carry = new Float32Array(0); pre = []; utt = [];
      speaking = false; silent = 0; voiced = 0;
      cfg.onState('idle');
    },
    get speaking() { return speaking; },
    get noiseFloor() { return floor; },
  };
}
