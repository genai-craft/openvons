/// VAD (発話区間の切り出し) の確認。合成した音で、切れ目が意図どおりかを見る。
import 'dart:math';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:openvons/engine/vad.dart';

/// 指定秒数の正弦波 (声のつもり) と無音を並べた 16kHz の音を作る。
Float32List build(List<(double seconds, double amplitude)> parts) {
  final total = parts.fold<int>(0, (a, p) => a + (p.$1 * 16000).round());
  final out = Float32List(total);
  var o = 0;
  for (final (sec, amp) in parts) {
    final n = (sec * 16000).round();
    for (var i = 0; i < n; i++) {
      out[o + i] = amp * sin(2 * pi * 220 * i / 16000) + (Random(i).nextDouble() - .5) * 0.0005;
    }
    o += n;
  }
  return out;
}

void main() {
  test('無音を挟んだ 2 発話が 2 つに切れる', () {
    final cuts = <Float32List>[];
    final vad = Vad(onUtterance: cuts.add);
    // 静けさ → 1 秒喋る → 1 秒黙る → 0.8 秒喋る → 1 秒黙る
    final audio = build([(1.0, 0.0), (1.0, 0.2), (1.0, 0.0), (0.8, 0.2), (1.0, 0.0)]);
    for (var i = 0; i < audio.length; i += 1024) {
      vad.feed(Float32List.sublistView(audio, i, min(i + 1024, audio.length)));
    }
    expect(cuts.length, 2);
    // 1 秒の発話 + プリロール 0.3 秒 + 終端無音 0.5 秒 に収まる範囲
    expect(cuts[0].length / 16000, greaterThan(1.0));
    expect(cuts[0].length / 16000, lessThan(2.2));
  });

  test('短すぎる物音は捨てる', () {
    final cuts = <Float32List>[];
    final vad = Vad(onUtterance: cuts.add);
    final audio = build([(1.0, 0.0), (0.1, 0.25), (1.5, 0.0)]);   // 0.1 秒だけの音
    for (var i = 0; i < audio.length; i += 1024) {
      vad.feed(Float32List.sublistView(audio, i, min(i + 1024, audio.length)));
    }
    expect(cuts, isEmpty);
  });

  test('長すぎる発話は強制的に切る', () {
    final cuts = <Float32List>[];
    final vad = Vad(onUtterance: cuts.add, cfg: const VadConfig(maxUttSec: 2));
    final audio = build([(1.0, 0.0), (5.0, 0.2), (1.0, 0.0)]);
    for (var i = 0; i < audio.length; i += 1024) {
      vad.feed(Float32List.sublistView(audio, i, min(i + 1024, audio.length)));
    }
    expect(cuts.length, greaterThanOrEqualTo(2));
    for (final c in cuts) {
      expect(c.length / 16000, lessThan(2.6));
    }
  });
}
