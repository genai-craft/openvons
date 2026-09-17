/// 校正と判断 (openvons/core/none_calibration.py, decision.py と同じ式)。
library;

import 'dart:math';

class Calibration {
  final double temperature, noneBias, lenBonus, residualPenalty;
  const Calibration({this.temperature = 2.5, this.noneBias = 4.0, this.lenBonus = 1.4, this.residualPenalty = 1.0});
  factory Calibration.fromJson(Map<String, dynamic> j) => Calibration(
        temperature: (j['temperature'] ?? 2.5).toDouble(),
        noneBias: (j['none_bias'] ?? 4.0).toDouble(),
        lenBonus: (j['len_bonus'] ?? 1.4).toDouble(),
        residualPenalty: (j['residual_penalty'] ?? 1.0).toDouble(),
      );

  /// 候補のスコアと、自由認識のスコアから [候補..., 該当なし] の確率を作る。
  List<double> probs(List<double> scores, List<int> lens, double freeScore, int nFree) {
    final z = <double>[];
    for (var i = 0; i < scores.length; i++) {
      final n = lens[i].toDouble();
      z.add((scores[i] + lenBonus * min(n, nFree.toDouble()) - residualPenalty * max(0, nFree - n)) / temperature);
    }
    z.add((freeScore - noneBias) / temperature);
    final mx = z.reduce(max);
    final ex = z.map((v) => exp(v - mx)).toList();
    final s = ex.reduce((a, b) => a + b);
    return ex.map((v) => v / s).toList();
  }
}

/// 3 段階 (実行 / 確認 / 棄却) と、該当なし。
String decide(double pTop, double noneProb, String risk, {double execute = 0.85, double confirm = 0.4, double executeMedium = 0.95}) {
  if (noneProb > pTop) return 'none';
  if (risk == 'high') return pTop >= confirm ? 'confirm' : 'reject';
  if (risk == 'medium') return pTop >= executeMedium ? 'execute' : (pTop >= confirm ? 'confirm' : 'reject');
  if (pTop >= execute) return 'execute';
  if (pTop >= confirm) return 'confirm';
  return 'reject';
}

String actionLabel(String a) => const {'execute': '実行', 'confirm': '確認', 'reject': '棄却', 'none': '該当なし'}[a] ?? a;
String levelLabel(double p, {double hi = 0.7, double mid = 0.45}) => p >= hi ? '確定' : (p >= mid ? '要確認' : '不明');
