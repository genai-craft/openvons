/// 発話区間の切り出し (エネルギー方式)。ハンズフリーのために使う。
///
/// サーバーに送るデモは Silero VAD をサーバー側で回しているが、端末内で完結する経路では
/// 持てないので、32ms ごとの RMS で「静かなときの底」を追いかけ、その何倍かを超えたら発話とみなす。
/// ブラウザ版と同じ考え方 (openvons/voice/demo_static/vad.js)。
library;

import 'dart:math';
import 'dart:typed_data';

class VadConfig {
  final int sampleRate, frame;
  final double startFactor, endFactor, absFloor;
  final int endSilenceMs, minSpeechMs, prerollMs;
  final double maxUttSec;
  const VadConfig({
    this.sampleRate = 16000,
    this.frame = 512,          // 32ms
    this.startFactor = 3.0,    // 底の何倍で発話開始とみなすか
    this.endFactor = 1.8,      // 何倍を下回ったら無音とみなすか
    this.absFloor = 0.004,     // 静かすぎる環境で雑音を拾わないための下限
    this.endSilenceMs = 500,   // 話し終わりと判断するまでの無音
    this.minSpeechMs = 250,    // これより短い音は捨てる
    this.prerollMs = 300,      // 立ち上がりが切れないように前に足す
    this.maxUttSec = 8,
  });
}

class Vad {
  final VadConfig cfg;
  final void Function(Float32List) onUtterance;
  final void Function(bool speaking)? onState;

  Vad({required this.onUtterance, this.onState, this.cfg = const VadConfig()});

  final _pre = <Float32List>[];
  final _utt = <Float32List>[];
  Float32List _carry = Float32List(0);
  double _floor = 0.01;
  bool _speaking = false;
  int _silent = 0, _voiced = 0;

  bool get speaking => _speaking;
  double get noiseFloor => _floor;

  int get _framesPerSec => cfg.sampleRate ~/ cfg.frame;
  int get _preFrames => (cfg.prerollMs * _framesPerSec / 1000).ceil();
  int get _endFrames => (cfg.endSilenceMs * _framesPerSec / 1000).ceil();
  int get _minFrames => (cfg.minSpeechMs * _framesPerSec / 1000).ceil();
  int get _maxFrames => (cfg.maxUttSec * _framesPerSec).ceil();

  void reset() {
    _pre.clear();
    _utt.clear();
    _carry = Float32List(0);
    _speaking = false;
    _silent = 0;
    _voiced = 0;
    onState?.call(false);
  }

  /// 16kHz の Float32 を好きな長さで渡す。内部で 32ms ごとに見る。
  void feed(Float32List chunk) {
    final buf = Float32List(_carry.length + chunk.length);
    buf.setAll(0, _carry);
    buf.setAll(_carry.length, chunk);
    var off = 0;
    while (off + cfg.frame <= buf.length) {
      final f = Float32List.sublistView(buf, off, off + cfg.frame);
      off += cfg.frame;
      final e = _rms(f);
      final startTh = max(_floor * cfg.startFactor, cfg.absFloor);
      final endTh = max(_floor * cfg.endFactor, cfg.absFloor * 0.7);
      if (!_speaking) {
        // 底の更新は無音のあいだだけ (発話中に上げてしまわない)
        _floor = e < _floor ? _floor * 0.9 + e * 0.1 : _floor * 0.995 + e * 0.005;
        _pre.add(Float32List.fromList(f));
        if (_pre.length > _preFrames) _pre.removeAt(0);
        if (e > startTh) {
          _speaking = true;
          _utt.clear();
          _silent = 0;
          _voiced = 1;
          onState?.call(true);
        }
      } else {
        _utt.add(Float32List.fromList(f));
        if (e > endTh) {
          _silent = 0;
          _voiced++;
        } else {
          _silent++;
        }
        if (_silent >= _endFrames || _utt.length >= _maxFrames) _end();
      }
    }
    _carry = Float32List.fromList(buf.sublist(off));
  }

  void _end() {
    final frames = [..._pre, ..._utt];
    _speaking = false;
    _utt.clear();
    _silent = 0;
    onState?.call(false);
    if (_voiced < _minFrames) { _voiced = 0; return; }   // 短すぎる音は捨てる
    _voiced = 0;
    var n = 0;
    for (final f in frames) {
      n += f.length;
    }
    final out = Float32List(max(n, cfg.sampleRate));     // モデルの最短長を下回らないように
    var o = 0;
    for (final f in frames) {
      out.setAll(o, f);
      o += f.length;
    }
    onUtterance(out);
  }

  static double _rms(Float32List x) {
    var s = 0.0;
    for (var i = 0; i < x.length; i++) {
      s += x[i] * x[i];
    }
    return sqrt(s / x.length);
  }
}
