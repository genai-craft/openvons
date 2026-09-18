/// 端末内の画像判定: 1 枚の画像を埋め込み、選択肢 (状態) の埋め込みと内積を取って確率にする。
/// 物体検出ではなく「場面の状態」を有限の選択肢で答えさせる。テキスト側はサーバーで先に計算済み。
library;

import 'dart:math';
import 'dart:typed_data';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:image/image.dart' as img;

import 'decision.dart';

class VisionAnswer {
  final String key, title, label, level;
  final double prob;
  final List<MapEntry<String, double>> all;
  /// 前提が崩れていて答えなかった質問 (食べ物が写っていないのに「和食」と言わないため)
  final bool skipped;
  final String skipReason;
  VisionAnswer(this.key, this.title, this.label, this.prob, this.level, this.all,
      {this.skipped = false, this.skipReason = ''});
}

class VisionEngine {
  OrtSession? _img;
  /// 質問セット (general = 室内外の汎用、kasen = 河川カメラの監視)。
  /// 端末に置くのは画像エンコーダだけなので、セットの切り替えは JSON の差し替えだけで済む。
  final Map<String, Map<String, dynamic>> docs = {};
  Map<String, dynamic>? choices;
  late int size;
  late List<double> mean, std;
  double logitScale = 100.0, logitBias = 0.0;

  bool get ready => _img != null && choices != null;

  String setKey = 'general';
  String get setTitle => (choices?['title'] ?? '') as String;

  /// 質問セットを切り替える。画像モデルはそのまま。
  void use(String key) {
    final d = docs[key];
    if (d == null) return;
    setKey = key;
    choices = d;
    logitScale = (d['logit_scale'] as num).toDouble();
    logitBias = (d['logit_bias'] as num).toDouble();
  }

  Future<void> load({required String modelPath, required Map<String, dynamic> preprocess,
      required Map<String, Map<String, dynamic>> choiceDocs, List<OrtProvider>? providers}) async {
    final ort = OnnxRuntime();
    _img = await ort.createSession(modelPath, options: OrtSessionOptions(providers: providers, intraOpNumThreads: 4));
    size = (preprocess['size'] ?? 224) as int;
    mean = List<double>.from(preprocess['mean'].map((e) => (e as num).toDouble()));
    std = List<double>.from(preprocess['std'].map((e) => (e as num).toDouble()));
    docs.clear();
    docs.addAll(choiceDocs);
    use(choiceDocs.containsKey(setKey) ? setKey : choiceDocs.keys.first);
  }

  Float32List preprocess(img.Image src) {
    final im = img.copyResize(src, width: size, height: size, interpolation: img.Interpolation.cubic);
    final out = Float32List(3 * size * size);
    for (var y = 0; y < size; y++) {
      for (var x = 0; x < size; x++) {
        final p = im.getPixel(x, y);
        final rgb = [p.r / 255.0, p.g / 255.0, p.b / 255.0];
        for (var c = 0; c < 3; c++) {
          out[c * size * size + y * size + x] = (rgb[c] - mean[c]) / std[c];
        }
      }
    }
    return out;
  }

  /// 返り値: (答え, 画像を数値にするのにかかった ms, 質問に答えるのにかかった マイクロ秒)。
  /// 「質問を増やしても 2 つ目の数字はほとんど増えない」ことを画面で見せるために分けて測る。
  Future<(List<VisionAnswer>, int, int)> run(img.Image image) async {
    final t0 = DateTime.now();
    final x = preprocess(image);
    final inp = await OrtValue.fromList(x, [1, 3, size, size]);
    final out = await _img!.run({'pixel_values': inp});
    final emb = (await out.values.first.asFlattenedList()).map((e) => (e as num).toDouble()).toList();
    final embedMs = DateTime.now().difference(t0).inMilliseconds;
    final t1 = DateTime.now();
    final answers = <VisionAnswer>[];
    final picked = <String, String>{};      // 質問 key -> 選ばれた選択肢 id (前提の判定に使う)
    for (final q in List<Map<String, dynamic>>.from(choices!['questions'])) {
      final cs = List<Map<String, dynamic>>.from(q['choices']);
      // 前提が満たされていない質問は答えない
      final req = q['requires'] as Map<String, dynamic>?;
      if (req != null) {
        final got = picked[req['key'] as String];
        final want = List<String>.from(req['any'] as List);
        if (got == null || !want.contains(got)) {
          final gate = List<Map<String, dynamic>>.from(choices!['questions'])
              .firstWhere((x) => x['key'] == req['key'], orElse: () => {'title': req['key']});
          answers.add(VisionAnswer(q['key'] as String, q['title'] as String, '—', 0, '対象外', const [],
              skipped: true, skipReason: '${gate['title']} が該当しないため'));
          continue;
        }
      }
      final embs = Map<String, dynamic>.from(q['embeddings']);
      final logits = <double>[];
      for (final c in cs) {
        final v = List<double>.from(embs[c['id']].map((e) => (e as num).toDouble()));
        var dot = 0.0;
        for (var i = 0; i < v.length; i++) dot += v[i] * emb[i];
        logits.add(dot * logitScale + logitBias);
      }
      final mx = logits.reduce(max);
      final ex = logits.map((v) => exp(v - mx)).toList();
      final s = ex.reduce((a, b) => a + b);
      final pr = ex.map((v) => v / s).toList();
      var best = 0;
      for (var i = 1; i < pr.length; i++) if (pr[i] > pr[best]) best = i;
      picked[q['key'] as String] = cs[best]['id'] as String;
      answers.add(VisionAnswer(q['key'] as String, q['title'] as String, cs[best]['label'] as String, pr[best],
          levelLabel(pr[best]), [for (var i = 0; i < cs.length; i++) MapEntry(cs[i]['label'] as String, pr[i])]));
    }
    return (answers, embedMs, DateTime.now().difference(t1).inMicroseconds);
  }
}
