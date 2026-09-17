/// 端末内の音声コマンド: mel → encoder → 自由認識 → 候補の絞り込み → 強制トークン採点 → 校正 → 判断。
library;

import 'dart:math';
import 'dart:typed_data';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';

import 'decision.dart';
import 'kana.dart';
import 'server.dart';

class VoiceResult {
  final String freeKana, action, text, intent;
  final Map<String, dynamic> slots;
  final double prob, noneProb;
  final int melMs, encMs, genMs, scoreMs;
  final List<MapEntry<String, double>> top;

  /// 比較用の「従来のやり方」: 文字起こしに一番近い言い方を選ぶだけ。棄却の手段を持たないので必ず何かを実行する。
  final String naiveText, naiveIntent;
  final Map<String, dynamic> naiveSlots;
  final double naiveSim;

  VoiceResult({required this.freeKana, required this.action, required this.text, required this.intent,
    required this.slots, required this.prob, required this.noneProb, required this.melMs, required this.encMs,
    required this.genMs, required this.scoreMs, required this.top,
    required this.naiveText, required this.naiveIntent, required this.naiveSlots, required this.naiveSim});
  int get totalMs => melMs + encMs + genMs + scoreMs;

  /// 従来のやり方なら誤作動していたか (openvons は実行しない / 別のものを実行する、のどちらか)。
  bool get naiveWouldMisfire => action != 'execute' || naiveIntent != intent ||
      (naiveSlots['camera'] ?? '') != (slots['camera'] ?? '');
}

class VoiceEngine {
  OrtSession? _mel, _enc, _dec;
  late List<int> prefix;
  late int eot;
  late Set<int> suppress;
  late TokenDecoder decoder;
  late Calibration cal;
  Map<String, dynamic> sets = {};
  String state = 'MAP';

  bool get ready => _mel != null && _enc != null && _dec != null;

  Future<void> load({required String melPath, required String encPath, required String decPath,
      required Map<String, dynamic> config, required TokenDecoder tokenDecoder, required Map<String, dynamic> commandSets,
      List<OrtProvider>? providers}) async {
    final ort = OnnxRuntime();
    final opt = OrtSessionOptions(providers: providers, intraOpNumThreads: 4);
    _mel = await ort.createSession(melPath, options: opt);
    _enc = await ort.createSession(encPath, options: opt);
    _dec = await ort.createSession(decPath, options: opt);
    prefix = List<int>.from(config['prefix']);
    eot = config['eot'] as int;
    suppress = Set<int>.from(List<int>.from(config['suppress'] ?? const []));
    decoder = tokenDecoder;
    cal = Calibration.fromJson(Map<String, dynamic>.from(config['calibration']));
    sets = commandSets;
  }

  Future<VoiceResult> run(Float32List audio) async {
    // decoder_head.onnx は (採点, 次トークン) だけを返す。語彙全体のロジットを端末に渡すと
    // 候補 20 件で 1300 万要素になり、Android の Java ヒープ (268MB) を超えて落ちる

    final t0 = DateTime.now();
    final melIn = await OrtValue.fromList(audio, [1, audio.length]);
    final melOut = await _mel!.run({'audio': melIn});
    final mel = melOut.values.first;
    final t1 = DateTime.now();
    final encOut = await _enc!.run({'input_features': mel});
    final h = encOut.values.first;
    final t2 = DateTime.now();

    // 自由認識 (greedy)。KV cache なしなので毎回 prefix から作り直す。
    // カナ以外の抑制はグラフに焼いてあるので、返るのは次の 1 トークンの id だけ
    final seq = List<int>.from(prefix);
    for (var step = 0; step < 24; step++) {
      final n = seq.length;
      final ids = await OrtValue.fromList(Int64List.fromList(seq), [1, n]);
      final zeros = await OrtValue.fromList(Int64List(n), [1, n]);
      final zerosF = await OrtValue.fromList(Float32List(n), [1, n]);
      final out = await _dec!.run({'input_ids': ids, 'encoder_hidden_states': h, 'targets': zeros, 'mask': zerosF});
      final nid = (await out['next_id']!.asFlattenedList()).first;
      final best = (nid as num).toInt();
      if (best == eot) break;
      seq.add(best);
    }
    final content = seq.sublist(prefix.length);
    final free = normalizeKana(decoder.decode(content).trim());
    final t3 = DateTime.now();

    // 候補の絞り込みと採点 (候補 + 自由認識をまとめて 1 回の forward)
    final hyps = List<Map<String, dynamic>>.from(sets[state]['hyps']);
    final idx = shortlist(free, hyps);
    final lists = <List<int>>[for (final i in idx) List<int>.from(hyps[i]['ids']), content];
    final scored = await _score(h, lists);
    final scores = scored.$1, lens = scored.$2;
    final nFree = lens.last;
    final probs = cal.probs(scores.sublist(0, scores.length - 1), lens.sublist(0, lens.length - 1), scores.last, nFree);
    final noneProb = probs.last;

    // 同じ意味の表層形は足し合わせる
    final agg = <String, List<dynamic>>{};
    for (var i = 0; i < idx.length; i++) {
      final h2 = hyps[idx[i]];
      final key = '${h2['i']}|${h2['s']}';
      final cur = agg[key] ?? [h2, 0.0, -1e30];
      cur[1] = (cur[1] as double) + probs[i];
      if (scores[i] > (cur[2] as double)) { cur[0] = h2; cur[2] = scores[i]; }
      agg[key] = cur;
    }
    final ranked = agg.values.toList()..sort((a, b) => (b[1] as double).compareTo(a[1] as double));
    final t4 = DateTime.now();
    // 従来のやり方 (比較用): 状態内の全候補から、カナの近さが最大のものを選ぶだけ
    var nbSim = -1.0; Map<String, dynamic>? nb;
    for (final h2 in hyps) {
      final sim = similarity(free, h2['k'] as String);
      if (sim > nbSim) { nbSim = sim; nb = h2; }
    }

    final top = ranked.isEmpty ? null : ranked.first;
    final action = top == null ? 'none'
        : decide(top[1] as double, noneProb, (top[0] as Map)['r'] as String? ?? 'low');
    return VoiceResult(
      freeKana: free, action: action,
      text: top == null ? '-' : (top[0] as Map)['t'] as String,
      intent: top == null ? '' : (top[0] as Map)['i'] as String,
      slots: top == null ? const {} : Map<String, dynamic>.from((top[0] as Map)['s'] as Map? ?? const {}),
      prob: top == null ? 0 : top[1] as double, noneProb: noneProb,
      naiveText: nb == null ? '-' : nb['t'] as String,
      naiveIntent: nb == null ? '' : nb['i'] as String,
      naiveSlots: nb == null ? const {} : Map<String, dynamic>.from(nb['s'] as Map? ?? const {}),
      naiveSim: nbSim < 0 ? 0 : nbSim,
      melMs: t1.difference(t0).inMilliseconds, encMs: t2.difference(t1).inMilliseconds,
      genMs: t3.difference(t2).inMilliseconds, scoreMs: t4.difference(t3).inMilliseconds,
      top: [for (final r in ranked.take(5)) MapEntry((r[0] as Map)['t'] as String, r[1] as double)],
    );
  }

  /// 候補ごとの log p(候補|音声) と、採点したトークン数。
  /// 採点はグラフの中で済むので、端末に返るのは候補ごとの 1 つの数字だけ。
  /// encoder の出力もグラフの中で候補数ぶんに広げるため、ここでは複製しない。
  Future<(List<double>, List<int>)> _score(OrtValue h, List<List<int>> lists) async {
    final b = lists.length;
    final seqs = [for (final t in lists) [...prefix, ...t, eot]];
    final inLen = seqs.map((s) => s.length).reduce(max) - 1;
    final ids = Int64List(b * inLen);
    final tgts = Int64List(b * inLen);
    final mask = Float32List(b * inLen);
    final lens = <int>[];
    for (var i = 0; i < b; i++) {
      final s = seqs[i];
      var n = 0;
      for (var j = 0; j < inLen; j++) {
        final k = i * inLen + j;
        ids[k] = j < s.length ? s[j] : eot;
        tgts[k] = j + 1 < s.length ? s[j + 1] : eot;
        // 採点するのは prefix より後ろ、かつ本物のトークンがある位置だけ
        if (j + 1 < s.length && j + 1 >= prefix.length) { mask[k] = 1.0; n++; }
      }
      lens.add(n);
    }
    final out = await _dec!.run({
      'input_ids': await OrtValue.fromList(ids, [b, inLen]),
      'encoder_hidden_states': h,
      'targets': await OrtValue.fromList(tgts, [b, inLen]),
      'mask': await OrtValue.fromList(mask, [b, inLen]),
    });
    final raw = await out['scores']!.asFlattenedList();
    return ([for (final v in raw) (v as num).toDouble()], lens);
  }
}
