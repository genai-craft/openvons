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

/// 学習済みヘッド (凍結した画像エンコーダ + 小さな head)。
/// 数万パラメータなので ONNX にせず、重みを JSON で受け取って端末側で計算する。
class TrainedHead {
  final String title;
  /// 学習時の実測値 (質問ごと)。画面に出して、どこまで信じてよいかを示す
  Map<String, dynamic> metricsOf(String key) =>
      Map<String, dynamic>.from((tasks[key] as Map)['metrics'] as Map? ?? const {});
  final List<double> normW, normB, b1;
  final List<List<double>> w1;
  final Map<String, dynamic> tasks;
  TrainedHead(this.title, this.normW, this.normB, this.w1, this.b1, this.tasks);

  factory TrainedHead.fromJson(Map<String, dynamic> j) {
    final t = Map<String, dynamic>.from(j['trunk'] as Map);
    List<double> d(dynamic v) => [for (final x in v as List) (x as num).toDouble()];
    List<List<double>> dd(dynamic v) => [for (final r in v as List) d(r)];
    return TrainedHead((j['title'] ?? '学習済み') as String, d(t['norm_w']), d(t['norm_b']),
        dd(t['w1']), d(t['b1']), Map<String, dynamic>.from(j['tasks'] as Map));
  }

  /// LayerNorm → Linear → GELU (質問間で共有する幹)
  List<double> _trunk(List<double> x) {
    var mean = 0.0;
    for (final v in x) {
      mean += v;
    }
    mean /= x.length;
    var varr = 0.0;
    for (final v in x) {
      varr += (v - mean) * (v - mean);
    }
    varr = varr / x.length;
    final inv = 1.0 / sqrt(varr + 1e-5);
    final n = [for (var i = 0; i < x.length; i++) (x[i] - mean) * inv * normW[i] + normB[i]];
    final h = <double>[];
    for (var o = 0; o < w1.length; o++) {
      var s = b1[o];
      final row = w1[o];
      for (var i = 0; i < row.length; i++) {
        s += row[i] * n[i];
      }
      // GELU (tanh 近似)
      h.add(0.5 * s * (1 + _tanh(0.7978845608 * (s + 0.044715 * s * s * s))));
    }
    return h;
  }

  static double _tanh(double x) {
    final e = exp(2 * x);
    return (e - 1) / (e + 1);
  }

  /// 「年齢層 55.7% / 性別 93.1%」のような一行 (テスト集合での正解率)
  String get accuracyLine => tasks.entries.map((e) {
        final m = Map<String, dynamic>.from((e.value as Map)['metrics'] as Map? ?? const {});
        final t = ((e.value as Map)['title'] ?? e.key) as String;
        final a = ((m['test_acc'] ?? 0) as num).toDouble();
        final n = ((e.value as Map)['labels'] as List).length;
        return '$t $n 区分 ${(a * 100).toStringAsFixed(1)}%';
      }).join(' / ');

  /// 画像の埋め込みから、質問ごとの校正済み確率を返す。
  List<VisionAnswer> answer(List<double> emb) {
    final h = _trunk(emb);
    final out = <VisionAnswer>[];
    for (final e in tasks.entries) {
      final t = Map<String, dynamic>.from(e.value as Map);
      final w2 = [for (final r in t['w2'] as List) [for (final x in r as List) (x as num).toDouble()]];
      final b2 = [for (final x in t['b2'] as List) (x as num).toDouble()];
      final temp = ((t['temperature'] ?? 1.0) as num).toDouble();
      final labels = List<Map<String, dynamic>>.from(t['labels'] as List);
      final logits = <double>[];
      for (var o = 0; o < w2.length; o++) {
        var s = b2[o];
        for (var i = 0; i < h.length; i++) {
          s += w2[o][i] * h[i];
        }
        logits.add(s / temp);
      }
      final mx = logits.reduce(max);
      final ex = logits.map((v) => exp(v - mx)).toList();
      final sum = ex.reduce((a, b) => a + b);
      final pr = ex.map((v) => v / sum).toList();
      var best = 0;
      for (var i = 1; i < pr.length; i++) {
        if (pr[i] > pr[best]) best = i;
      }
      out.add(VisionAnswer(e.key, (t['title'] ?? e.key) as String, labels[best]['label'] as String, pr[best],
          levelLabel(pr[best]), [for (var i = 0; i < labels.length; i++) MapEntry(labels[i]['label'] as String, pr[i])]));
    }
    return out;
  }
}

class VisionEngine {
  OrtSession? _img;
  /// 学習済みヘッド (key = "head:fairface" など)
  final Map<String, TrainedHead> heads = {};
  /// 質問セット (general = 室内外の汎用、kasen = 河川カメラの監視)。
  /// 端末に置くのは画像エンコーダだけなので、セットの切り替えは JSON の差し替えだけで済む。
  final Map<String, Map<String, dynamic>> docs = {};
  Map<String, dynamic>? choices;
  late int size;
  late List<double> mean, std;
  double logitScale = 100.0, logitBias = 0.0;

  bool get ready => _img != null && choices != null;

  String setKey = 'general';
  String get setTitle => trained != null ? trained!.title : (choices?['title'] ?? '') as String;

  /// いま選ばれているのが学習済みヘッドならそれを返す
  TrainedHead? get trained => heads[setKey];

  /// 質問セットを切り替える。画像モデルはそのまま。
  void use(String key) {
    if (heads.containsKey(key)) { setKey = key; return; }
    final d = docs[key];
    if (d == null) return;
    setKey = key;
    choices = d;
    logitScale = (d['logit_scale'] as num).toDouble();
    logitBias = (d['logit_bias'] as num).toDouble();
  }

  /// 顔を写す想定なので中央の正方形を切り出す (学習データが顔の切り抜きのため)。
  img.Image centerSquare(img.Image src) {
    final n = min(src.width, src.height);
    return img.copyCrop(src, x: (src.width - n) ~/ 2, y: (src.height - n) ~/ 2, width: n, height: n);
  }

  Future<void> load({required String modelPath, required Map<String, dynamic> preprocess,
      required Map<String, Map<String, dynamic>> choiceDocs,
      Map<String, Map<String, dynamic>> headDocs = const {}, List<OrtProvider>? providers}) async {
    final ort = OnnxRuntime();
    _img = await ort.createSession(modelPath, options: OrtSessionOptions(providers: providers, intraOpNumThreads: 4));
    size = (preprocess['size'] ?? 224) as int;
    mean = List<double>.from(preprocess['mean'].map((e) => (e as num).toDouble()));
    std = List<double>.from(preprocess['std'].map((e) => (e as num).toDouble()));
    docs.clear();
    docs.addAll(choiceDocs);
    heads.clear();
    headDocs.forEach((k, v) => heads[k] = TrainedHead.fromJson(v));
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
    final head = trained;
    final x = preprocess(head != null ? centerSquare(image) : image);
    final inp = await OrtValue.fromList(x, [1, 3, size, size]);
    final out = await _img!.run({'pixel_values': inp});
    final emb = (await out.values.first.asFlattenedList()).map((e) => (e as num).toDouble()).toList();
    final embedMs = DateTime.now().difference(t0).inMilliseconds;
    final t1 = DateTime.now();
    if (head != null) {
      return (head.answer(emb), embedMs, DateTime.now().difference(t1).inMicroseconds);
    }
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
