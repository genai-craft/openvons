/// 河川カメラの指令卓 (端末内)。音声の判断結果を受けて、状態と表示中のカメラを動かす。
///
/// 画面に「いま受け付けている言い方」と「言った結果どう動いたか」が出るようにするための、
/// アプリ側の最小の状態機械。サーバーは地点の一覧と画像の中継をするだけ。
library;

class Site {
  final String id, code, label, reading, river, office, pref;
  final int order;
  final double? lat, lng;
  Site(this.id, this.code, this.label, this.reading, this.river, this.office, this.pref, this.order, this.lat, this.lng);
  factory Site.fromJson(Map<String, dynamic> j) => Site(
      j['id'] as String, (j['code'] ?? '') as String, j['label'] as String, (j['reading'] ?? '') as String,
      (j['river'] ?? '') as String, (j['office'] ?? '') as String, (j['pref'] ?? '') as String,
      (j['order'] ?? 0) as int, (j['lat'] as num?)?.toDouble(), (j['lng'] as num?)?.toDouble());
}

/// 地図をどう動かすか。画面側が実際の地図に渡す。
class MapMove {
  final double dx, dy;      // 画面の何割ぶん動かすか (右と下が正)
  final int zoom;           // +1 拡大 / -1 縮小 / 0 変えない
  const MapMove({this.dx = 0, this.dy = 0, this.zoom = 0});
}

/// 音声 1 回分の出来事。画面の履歴に積む。
class Event {
  final String speech;      // 指令卓が何をしたか
  final bool changed;       // 表示が変わったか
  final MapMove? map;       // 地図を動かす場合
  Event(this.speech, this.changed, {this.map});
}

/// 8 方向の単位ベクトル (画面座標。右と下が正)
const _panVec = {
  'up': [0.0, -1.0], 'down': [0.0, 1.0], 'left': [-1.0, 0.0], 'right': [1.0, 0.0],
  'upleft': [-0.71, -0.71], 'upright': [0.71, -0.71], 'downleft': [-0.71, 0.71], 'downright': [0.71, 0.71],
};

class Console {
  final List<Site> sites;
  final String scopeName;
  Console(this.sites, this.scopeName);

  String state = 'MAP';
  String? cameraId;
  double zoom = 1.0;
  final favorites = <String>{};
  int imageSeq = 0;          // 更新のたびに増やして画像のキャッシュを外す

  // 確認待ちの内容
  String? pendingText, pendingIntent;
  Map<String, dynamic> pendingSlots = const {};

  Site? get camera => cameraId == null ? null : siteOf(cameraId!);
  Site? siteOf(String id) {
    for (final s in sites) {
      if (s.id == id) return s;
    }
    return null;
  }

  /// 確認待ちを取り消して元の画面へ戻す。
  void cancelPending() {
    if (state != 'CONFIRM') return;
    pendingIntent = null;
    pendingText = null;
    pendingSlots = const {};
    state = cameraId == null ? 'MAP' : 'CAMERA';
  }

  String get stateLabel => switch (state) {
        'MAP' => '地図',
        'CAMERA' => 'カメラ表示中',
        _ => '確認待ち',
      };

  String get stateHint => switch (state) {
        'MAP' => '地点名を言うと、そのカメラが出ます',
        'CAMERA' => '上流 / 下流・更新・拡大・縮小・戻る。地点名で別の地点へ',
        _ => 'はい / いいえ のどちらかだけを受け付けます',
      };

  /// いま受け付けている言い方の例。状態で変わることを見せるための短い一覧。
  List<String> get examples => switch (state) {
        'MAP' => [sites.isEmpty ? '地点名' : sites.first.label, '地点名を表示', 'ヘルプ'],
        'CAMERA' => ['ひとつ下流', '上流へ', '更新して', 'もっと寄って', '地図に戻って', 'この地点を登録して'],
        _ => ['はい', 'いいえ'],
      };

  /// 隣のカメラ (同じ河川の上流・下流)。
  Site? neighbor(int step) {
    final c = camera;
    if (c == null) return null;
    final same = sites.where((s) => s.river == c.river).toList()..sort((a, b) => a.order.compareTo(b.order));
    final i = same.indexWhere((s) => s.id == c.id);
    if (i < 0 || i + step < 0 || i + step >= same.length) return null;
    return same[i + step];
  }

  /// 判断が「実行」なら実際に動かす。「確認」なら確認待ちに入る。「却下」は何もしない。
  Event apply(String action, String intent, Map<String, dynamic> slots, String text,
      [Map<String, dynamic> params = const {}]) {
    if (action == 'reject' || action == 'none') return Event('聞き流しました (この端末には指示していないと判断)', false);
    if (action == 'confirm') {
      pendingText = text;
      pendingIntent = intent;
      pendingSlots = slots;
      state = 'CONFIRM';
      return Event('$text ですか？', false);
    }
    return _execute(intent, slots, text, params);
  }

  Event _execute(String intent, Map<String, dynamic> slots, String text, [Map<String, dynamic> params = const {}]) {
    if (state == 'CONFIRM') {
      final pi = pendingIntent, ps = pendingSlots, pt = pendingText;
      state = cameraId == null ? 'MAP' : 'CAMERA';
      pendingIntent = null;
      pendingText = null;
      pendingSlots = const {};
      if (intent == 'yes' && pi != null) return _execute(pi, ps, pt ?? '');
      // 確認待ちからの逃げ道。はい / いいえ だけだと行き止まりに感じる
      if (intent == 'back' || intent == 'help') return _execute(intent, slots, text, params);
      return Event('取り消しました', false);
    }
    // 地図の移動と、地図表示中の拡大縮小は、カメラを選んでいなくても効く
    if (intent.startsWith('pan_')) {
      final v = _panVec[(params['dir'] ?? '') as String];
      if (v == null) return Event('その向きは分かりません', false);
      final f = ((params['amount'] ?? 0.6) as num).toDouble();
      return Event(text, true, map: MapMove(dx: v[0] * f, dy: v[1] * f));
    }
    if (state == 'MAP' && (intent == 'zoom_in' || intent == 'zoom_out')) {
      final zi = intent == 'zoom_in';
      return Event(zi ? '拡大します' : '縮小します', true, map: MapMove(zoom: zi ? 1 : -1));
    }

    switch (intent) {
      case 'select_camera':
        final id = slots['camera'] as String?;
        final s = id == null ? null : siteOf(id);
        if (s == null) return Event('その地点は担当範囲にありません', false);
        cameraId = s.id;
        state = 'CAMERA';
        zoom = 1.0;
        imageSeq++;
        return Event('${s.label} (${s.river})', true);
      case 'back':
        cameraId = null;
        state = 'MAP';
        zoom = 1.0;
        return Event('地図に戻ります', true);
      case 'upstream':
      case 'downstream':
        final nb = neighbor(intent == 'upstream' ? -1 : 1);
        if (nb == null) return Event('この先にカメラはありません', false);
        cameraId = nb.id;
        zoom = 1.0;
        imageSeq++;
        return Event('${intent == 'upstream' ? '上流' : '下流'}へ: ${nb.label}', true);
      case 'refresh':
        imageSeq++;
        return Event('画像を更新しました', true);
      case 'zoom_in':
        zoom = (zoom * 1.5).clamp(1.0, 6.0);
        return Event('拡大 (x${zoom.toStringAsFixed(1)})', true);
      case 'zoom_out':
        zoom = (zoom / 1.5).clamp(1.0, 6.0);
        return Event('縮小 (x${zoom.toStringAsFixed(1)})', true);
      case 'favorite':
        final id = cameraId;
        if (id == null) return Event('地点を選んでください', false);
        favorites.add(id);
        return Event('${siteOf(id)?.label} をお気に入りに登録しました', true);
      case 'help':
        return Event(examples.join('、'), false);
      case 'yes':
      case 'no':
        return Event('いま確認待ちではありません', false);
      default:
        return Event(text, false);
    }
  }
}

/// 画面上部に出す集計。「従来のやり方なら何回まちがえていたか」を数える。
class Tally {
  int executed = 0, confirmed = 0, rejected = 0, naiveMisfire = 0, utterances = 0;
  void add(String action, bool naiveMisfire_) {
    utterances++;
    if (action == 'execute') {
      executed++;
    } else if (action == 'confirm') {
      confirmed++;
    } else {
      rejected++;
    }
    if (naiveMisfire_) naiveMisfire++;
  }
}
