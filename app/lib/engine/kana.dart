/// カナの正規化と、候補の絞り込み (サーバー側 openvons/voice/kana.py と同じ規則)。
library;

String normalizeKana(String s) {
  final b = StringBuffer();
  for (final r in s.runes) {
    final c = String.fromCharCode(r);
    if ('　 、。，．,.!?！？「」『』・~〜()（）[]【】"\'’‘-–—:;/…'.contains(c)) continue;
    switch (c) {
      case 'ヲ': b.write('オ'); break;
      case 'ヂ': b.write('ジ'); break;
      case 'ヅ': b.write('ズ'); break;
      case 'ヰ': b.write('イ'); break;
      case 'ヱ': b.write('エ'); break;
      default: b.write(c);
    }
  }
  return b.toString();
}

int levenshtein(String a, String b) {
  if (a.isEmpty || b.isEmpty) return a.length > b.length ? a.length : b.length;
  var prev = List<int>.generate(b.length + 1, (j) => j);
  var cur = List<int>.filled(b.length + 1, 0);
  for (var i = 1; i <= a.length; i++) {
    cur[0] = i;
    for (var j = 1; j <= b.length; j++) {
      final cost = a.codeUnitAt(i - 1) == b.codeUnitAt(j - 1) ? 0 : 1;
      var m = prev[j] + 1;
      if (cur[j - 1] + 1 < m) m = cur[j - 1] + 1;
      if (prev[j - 1] + cost < m) m = prev[j - 1] + cost;
      cur[j] = m;
    }
    final t = prev; prev = cur; cur = t;
  }
  return prev[b.length];
}

double similarity(String a, String b) {
  final n = a.length > b.length ? a.length : b.length;
  if (n == 0) return 1;
  return 1 - levenshtein(a, b) / n;
}

/// 自由認識のカナに近い候補を k 個。同じ意味は最大 2 表層形まで。
List<int> shortlist(String free, List<Map<String, dynamic>> hyps, {int k = 12}) {
  final scored = <MapEntry<int, double>>[];
  for (var i = 0; i < hyps.length; i++) {
    scored.add(MapEntry(i, similarity(free, hyps[i]['k'] as String)));
  }
  scored.sort((x, y) => y.value.compareTo(x.value));
  final perMeaning = <String, int>{};
  final out = <int>[];
  for (final e in scored) {
    final h = hyps[e.key];
    final key = '${h['i']}|${h['s']}';
    final c = perMeaning[key] ?? 0;
    if (c >= 2) continue;
    perMeaning[key] = c + 1;
    out.add(e.key);
    if (out.length >= k) break;
  }
  return out;
}
