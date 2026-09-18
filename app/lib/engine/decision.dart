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

/// 3 段階 (実行 / 確認 / 棄却) と、該当なし。openvons/core/decision.py と同じ規則。
///
/// confirmable=false は「確認し直せない意図」(はい / いいえ)。ここで confirm を返すと
/// 「いいえ でよろしいですか」と聞き返す無限ループになるので、受けるか棄却するかの 2 択にする。
/// positive はその返事が実行側 (はい) か取り消し側 (いいえ) か。
String decide(double pTop, double noneProb, String risk,
    {double execute = 0.85, double confirm = 0.4, double executeMedium = 0.95,
    double clearMin = 0.65, double clearRatio = 3.0, double clearNoneMax = 0.20,
    double answerYes = 0.60, double answerNo = 0.40,
    bool confirmable = true, bool positive = true}) {
  if (noneProb > pTop) return 'none';
  if (!confirmable) {
    return pTop >= (positive ? answerYes : answerNo) ? 'execute' : 'reject';
  }
  if (risk == 'high') return pTop >= confirm ? 'confirm' : 'reject';
  if (risk == 'medium') return pTop >= executeMedium ? 'execute' : (pTop >= confirm ? 'confirm' : 'reject');
  if (pTop >= execute) return 'execute';
  // 他の候補に残った確率が小さく、該当なしも低いなら、迷っていないので確認を省く
  final rest = (1.0 - pTop - noneProb).clamp(0.0, 1.0);
  if (pTop >= clearMin && noneProb <= clearNoneMax && rest <= pTop / clearRatio) return 'execute';
  if (pTop >= confirm) return 'confirm';
  return 'reject';
}

String actionLabel(String a) => const {'execute': '実行', 'confirm': '確認', 'reject': '棄却', 'none': '該当なし'}[a] ?? a;
String levelLabel(double p, {double hi = 0.7, double mid = 0.45}) => p >= hi ? '確定' : (p >= mid ? '要確認' : '不明');
