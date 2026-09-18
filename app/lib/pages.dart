/// 画面: 声で操作 (河川カメラの指令卓) / 映像の状態 / 設定
///
/// デモの狙いは「確率で答える判断層が何を防いでいるか」を目で見せること。
/// そのため音声の画面では、同じ音声に対して
///   従来のやり方 (文字起こしに一番近い言い方を選ぶ。棄却できない)
///   openvons    (候補を確率で採点し、該当なしを含めて実行/確認/却下に分ける)
/// を毎回並べ、指令卓が実際に動く様子と、従来なら誤作動していた回数を数える。
library;

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:latlong2/latlong.dart';
import 'package:http/http.dart' as http;
import 'package:image/image.dart' as img;
import 'package:record/record.dart';

import 'engine/console.dart';
import 'engine/decision.dart';
import 'engine/vad.dart';
import 'engine/vision.dart';
import 'engine/voice.dart';
import 'main.dart' show acc, bad, ok, panel, warn;

/// ---------------------------------------------------------------- 声で操作
class VoicePage extends StatefulWidget {
  final VoiceEngine engine;
  final String status;
  final String server;
  final Console? console;
  final bool speak;      // 端末の合成音声で返すか (設定タブで切り替え)
  const VoicePage({super.key, required this.engine, required this.status, required this.server,
    required this.console, required this.speak});
  @override
  State<VoicePage> createState() => _VoicePageState();
}

class _Turn {
  final VoiceResult r;
  final String effect;
  _Turn(this.r, this.effect);
}

class _VoicePageState extends State<VoicePage> {
  final _rec = AudioRecorder();
  StreamSubscription<Uint8List>? _sub;
  Vad? _vad;
  final _queue = <Float32List>[];
  bool _listening = false, _testing = false, _draining = false, _hearing = false;
  final _turns = <_Turn>[];
  final _tally = Tally();
  String _msg = '';
  bool _showMap = true;          // 地図 / 一覧 の切り替え
  String _filter = '';           // 一覧の絞り込み
  Site? _picked;                 // 地図で押した地点 (読み方の確認用)
  double _zoom = 9.5;            // ラベルを出すかどうかの判断に使う
  final _mapCtl = MapController();
  final _tts = FlutterTts();
  bool _speaking = false;        // 読み上げ中はマイクを止める (自分の声を拾わないため)
  Timer? _confirmTimer, _unmute;

  @override
  void initState() {
    super.initState();
    _tts.setLanguage('ja-JP');
    _tts.setSpeechRate(0.55);
    // 読み終わった通知が来ても、余韻が残るので少し待ってからマイクを戻す
    _tts.setCompletionHandler(() => _unmuteAfter(const Duration(milliseconds: 500)));
    _tts.setCancelHandler(() => _unmuteAfter(const Duration(milliseconds: 300)));
    _tts.setErrorHandler((_) => _unmuteAfter(const Duration(milliseconds: 300)));
  }

  @override
  void dispose() { _unmute?.cancel(); _confirmTimer?.cancel(); _sub?.cancel(); _rec.dispose(); _tts.stop(); super.dispose(); }

  Console? get _c => widget.console;

  /// 端末の合成音声で返す。読み上げている間と、その直後の余韻のあいだはマイクを止める。
  /// (端末のエコー抑制だけでは自分の声が残り、「〜でよろしいですか」を指示として拾ってしまう)
  Future<void> _say(String text) async {
    if (!widget.speak || text.isEmpty) return;
    _muteMic();
    await _tts.stop();
    await _tts.speak(text);
    // 完了通知が来ない端末があるので、長さから見積もった時間でも必ず解除する
    _unmuteAfter(Duration(milliseconds: 900 + text.length * 200));
  }

  void _muteMic() {
    _speaking = true;
    _unmute?.cancel();
    _vad?.reset();        // 途中まで溜まっていた音は捨てる
  }

  /// 読み上げが終わってから少し置いてマイクを戻す。戻すときに VAD を初期化して、
  /// 余韻を発話の続きとして拾わないようにする。
  void _unmuteAfter(Duration d) {
    _unmute?.cancel();
    _unmute = Timer(d, () {
      _vad?.reset();
      _speaking = false;
    });
  }

  /// ハンズフリー待受。一度押したら入れっぱなしで、話し終わりを自分で見つけて判断し、
  /// そのまま次の指示を待つ。手が塞がっている現場を想定しているので、押しっぱなしにはしない。
  Future<void> _toggle() async {
    if (!widget.engine.ready) { setState(() => _msg = '設定タブでモデルを取得してください'); return; }
    if (_listening) {
      await _sub?.cancel();
      await _rec.stop();
      _vad?.reset();
      _queue.clear();
      setState(() { _listening = false; _hearing = false; _msg = ''; });
      return;
    }
    if (!await _rec.hasPermission()) { setState(() => _msg = 'マイクの許可が要ります'); return; }
    _vad = Vad(
      onUtterance: (audio) {
        if (_speaking) return;          // 自分の読み上げ由来は捨てる
        _queue.add(audio);
        _drain();
      },
      onState: (sp) { if (mounted && _listening) setState(() => _hearing = sp); },
    );
    final stream = await _rec.startStream(const RecordConfig(
        encoder: AudioEncoder.pcm16bits, sampleRate: 16000, numChannels: 1,
        // 既定は全て false。入れないと自分の読み上げをそのまま拾って、確認が延々と続く
        echoCancel: true, noiseSuppress: true, autoGain: true));
    _sub = stream.listen((bytes) {
      if (_speaking) return;              // 自分の読み上げは聞かない
      final pcm = Int16List.view(Uint8List.fromList(bytes).buffer);
      final f = Float32List(pcm.length);
      for (var i = 0; i < pcm.length; i++) {
        f[i] = pcm[i] / 32768.0;
      }
      _vad?.feed(f);
    });
    setState(() { _listening = true; _msg = ''; });
  }

  /// 溜まった発話を順に処理する (判断中に話されたぶんも取りこぼさない)。
  Future<void> _drain() async {
    if (_draining) return;
    _draining = true;
    while (_queue.isNotEmpty) {
      final audio = _queue.removeAt(0);
      if (mounted) setState(() => _msg = '端末の中で判断しています…');
      await _handle(audio);
    }
    _draining = false;
    if (mounted) setState(() => _msg = '');
  }

  /// 声で地図を動かす。画面の何割ぶんかで平行移動し、拡大縮小は 1 段ずつ。
  void _moveMap(MapMove mv) {
    final cam = _mapCtl.camera;
    if (mv.zoom != 0) {
      _mapCtl.move(cam.center, (cam.zoom + mv.zoom).clamp(4.0, 16.0));
      return;
    }
    final b = cam.visibleBounds;
    final dLat = (b.north - b.south) * -mv.dy;     // 画面の下が南
    final dLng = (b.east - b.west) * mv.dx;
    _mapCtl.move(LatLng(cam.center.latitude + dLat, cam.center.longitude + dLng), cam.zoom);
  }

  /// 確認待ちのまま黙っていたら自分で取り消す (行き止まりにしない)。
  void _armConfirmTimeout(Console? c) {
    _confirmTimer?.cancel();
    if (c == null || c.state != 'CONFIRM') return;
    _confirmTimer = Timer(const Duration(seconds: 12), () {
      if (!mounted || c.state != 'CONFIRM') return;
      c.cancelPending();
      setState(() => _msg = '確認の返事が無かったので取り消しました');
      _say('取り消しました');
    });
  }

  /// 音声 1 回分。判断して、指令卓に流して、集計する。
  Future<void> _handle(Float32List audio) async {
    final c = _c;
    try {
      if (c != null) widget.engine.state = c.state;
      final r = await widget.engine.run(audio);
      final ev = c?.apply(r.action, r.intent, r.slots, r.text, r.params);
      _tally.add(r.action, r.naiveWouldMisfire);
      if (ev?.map != null) _moveMap(ev!.map!);
      _armConfirmTimeout(c);
      if (!mounted) return;
      setState(() {
        _turns.insert(0, _Turn(r, ev?.speech ?? ''));
        _msg = '';
      });
      // 確認は必ず声で聞き返す。実行は短く復唱し、聞き流したときは黙る
      if (r.action == 'confirm') {
        await _say('${r.text} でよろしいですか');
      } else if (r.action == 'execute' && (ev?.speech ?? '').isNotEmpty) {
        await _say(ev!.speech);
      }
    } catch (e) {
      if (mounted) setState(() => _msg = 'エラー: $e');
    }
  }

  /// マイクを使わずに経路を確かめる。サーバーの検証用音声 (合成) を落として端末内で判断する。
  /// 電話中の雑談を模した音声も混ざっていて、それを「該当なし」に倒せるかが見どころ。
  Future<void> _selfTest() async {
    if (!widget.engine.ready) { setState(() => _msg = '設定タブでモデルを取得してください'); return; }
    setState(() { _testing = true; _msg = '検証用の音声を取得中…'; });
    try {
      final list = jsonDecode(utf8.decode((await http.get(Uri.parse('${widget.server}/api/samples'))).bodyBytes)) as List;
      for (final raw in list) {
        final s = Map<String, dynamic>.from(raw as Map);
        setState(() => _msg = '「${s["show"]}」と言ってみています…');
        final wav = (await http.get(Uri.parse('${widget.server}/api/sample/${s["id"]}.wav'))).bodyBytes;
        // 状態は指令卓に任せる (地点名 → カメラ表示 → 操作、と自然に遷移する)
        await _handle(_decodeWav(wav));
      }
      setState(() => _msg = '4 件を再生しました。最後の 1 件は電話中の雑談で、はねるのが正解です');
    } catch (err) {
      setState(() => _msg = 'エラー: $err');
    } finally {
      setState(() => _testing = false);
    }
  }

  /// 16kHz mono の WAV から PCM を取り出す (data チャンクを探すだけ)。
  static Float32List _decodeWav(Uint8List b) {
    var off = 12;
    while (off + 8 <= b.length) {
      final id = String.fromCharCodes(b.sublist(off, off + 4));
      final len = b.buffer.asByteData().getUint32(off + 4, Endian.little);
      if (id == 'data') {
        final pcm = Int16List.view(b.buffer, b.offsetInBytes + off + 8, len ~/ 2);
        final out = Float32List(pcm.length < 16000 ? 16000 : pcm.length);
        for (var i = 0; i < pcm.length; i++) {
          out[i] = pcm[i] / 32768.0;
        }
        return out;
      }
      off += 8 + len + (len.isOdd ? 1 : 0);
    }
    throw 'wav の data チャンクが見つかりません';
  }

  @override
  Widget build(BuildContext context) {
    final c = _c;
    final e = widget.engine;
    final n = e.ready && c != null ? (e.sets[c.state]?['n'] ?? 0) as int : 0;
    return ListView(padding: const EdgeInsets.all(12), children: [
      // 状態 — 受け付ける言い方がここで決まる
      Row(children: [
        _chip(c?.stateLabel ?? '地図', acc),
        const SizedBox(width: 8),
        Expanded(child: Text(c == null ? '地点一覧を取得中…' : '${c.scopeName} · ${c.sites.length} 地点',
            style: const TextStyle(color: Colors.white70, fontSize: 12))),
        if (n > 0) Text('$n 通りの言い方', style: const TextStyle(color: Colors.white38, fontSize: 11, fontFamily: 'monospace')),
      ]),
      const SizedBox(height: 4),
      Text(c?.stateHint ?? '', style: const TextStyle(color: Colors.white38, fontSize: 11)),
      const SizedBox(height: 8),

      // 指令卓
      _consoleView(c),
      const SizedBox(height: 8),
      _sayable(c),
      const SizedBox(height: 8),

      // 話しかける
      FilledButton.icon(
        onPressed: _testing ? null : _toggle,
        icon: Icon(_listening ? Icons.stop : Icons.mic),
        label: Text(_listening
            ? (_hearing ? '聞いています…' : '待受中 (押すと止める)')
            : 'ハンズフリー待受を始める'),
        style: FilledButton.styleFrom(
            backgroundColor: _listening ? (_hearing ? ok : bad) : acc,
            foregroundColor: Colors.black, minimumSize: const Size.fromHeight(52)),
      ),
      if (_listening) const Padding(padding: EdgeInsets.only(top: 6),
          child: Text('話し終わって少し黙ると、そこまでを 1 つの指示として判断します。そのまま次の指示を続けられます。',
              style: TextStyle(color: Colors.white38, fontSize: 11))),
      const SizedBox(height: 6),
      OutlinedButton.icon(
        onPressed: _testing || _listening ? null : _selfTest,
        icon: const Icon(Icons.play_circle_outline, size: 18),
        label: const Text('マイク無しで試す (合成音声 4 件)'),
        style: OutlinedButton.styleFrom(foregroundColor: acc, minimumSize: const Size.fromHeight(42)),
      ),
      if (_msg.isNotEmpty) Padding(padding: const EdgeInsets.only(top: 8),
          child: Text(_msg, style: const TextStyle(color: Colors.white70, fontSize: 13))),

      // 集計 — この画面の主張そのもの
      if (_tally.utterances > 0) ...[
        const SizedBox(height: 12),
        _tallyBar(_tally),
      ],

      const SizedBox(height: 10),
      for (final t in _turns.take(6)) _turnCard(t),

      if (_turns.isEmpty) ...[
        const SizedBox(height: 10),
        Card(child: Padding(padding: const EdgeInsets.all(12), child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: const [
          Text('やってみること', style: TextStyle(fontWeight: FontWeight.w700)),
          SizedBox(height: 6),
          Text('1. 地点名を言う (例: 栗橋水位)。カメラの実画像が出ます。\n'
              '2. 「ひとつ下流」「更新して」「もっと寄って」で動かす。\n'
              '3. 電話をしているつもりで関係ない話をする。\n'
              '   → 従来のやり方なら何か実行してしまう場面で、openvons は「該当なし」ではねます。',
              style: TextStyle(color: Colors.white70, fontSize: 13, height: 1.5)),
          SizedBox(height: 8),
          Text('認識も判断も端末の中で動いています。音声はこの端末から出ません。',
              style: TextStyle(color: Colors.white38, fontSize: 11)),
        ]))),
      ],
    ]);
  }

  // ------------------------------------------------------------- 指令卓の絵
  Widget _consoleView(Console? c) {
    if (c == null) {
      return Container(height: 180, decoration: BoxDecoration(color: panel, borderRadius: BorderRadius.circular(10)),
          child: const Center(child: CircularProgressIndicator()));
    }
    final cam = c.camera;
    if (cam == null) return _mapOrList(c);
    final url = '${widget.server}/api/image/${cam.id}?v=${c.imageSeq}';
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      ClipRRect(
        borderRadius: BorderRadius.circular(10),
        child: AspectRatio(
          aspectRatio: 4 / 3,
          child: Container(color: Colors.black,
            child: ClipRect(child: Transform.scale(scale: c.zoom,
              child: Image.network(url, fit: BoxFit.contain, gaplessPlayback: true,
                errorBuilder: (_, __, ___) => const Center(child: Text('画像を取得できません', style: TextStyle(color: Colors.white38))),
                loadingBuilder: (ctx, w, p) => p == null ? w : const Center(child: CircularProgressIndicator()))))),
        ),
      ),
      const SizedBox(height: 6),
      Row(children: [
        Expanded(child: Text(cam.label, style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w700))),
        if (c.favorites.contains(cam.id)) const Icon(Icons.star, color: warn, size: 18),
        if (c.zoom > 1.01) Text(' x${c.zoom.toStringAsFixed(1)}', style: const TextStyle(color: Colors.white54, fontSize: 12)),
      ]),
      Text('${cam.reading} · ${cam.river} · ${cam.pref}', style: const TextStyle(color: Colors.white38, fontSize: 11)),
      const Text('出典: 国土交通省 関東地方整備局 (PDL1.0)。画像は表示のたびに取得し、保存しません。',
          style: TextStyle(color: Colors.white24, fontSize: 10)),
    ]);
  }

  /// 地図 / 一覧。「何と言えるか」を覚えなくて済むように、地点は常にどちらかで見えている。
  Widget _mapOrList(Console c) {
    final sites = c.sites;
    return Container(
      decoration: BoxDecoration(color: panel, borderRadius: BorderRadius.circular(10), border: Border.all(color: Colors.white10)),
      padding: const EdgeInsets.all(8),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(children: [
          const Expanded(child: Text('この中の地点名を声で言ってください', style: TextStyle(color: Colors.white70, fontSize: 12))),
          ToggleButtons(
            isSelected: [_showMap, !_showMap],
            onPressed: (i) => setState(() => _showMap = i == 0),
            borderRadius: BorderRadius.circular(8),
            constraints: const BoxConstraints(minHeight: 28, minWidth: 48),
            selectedColor: Colors.black, fillColor: acc, color: Colors.white70,
            children: const [Text('地図', style: TextStyle(fontSize: 12)), Text('一覧', style: TextStyle(fontSize: 12))],
          ),
        ]),
        const SizedBox(height: 6),
        if (_showMap) ...[
          SizedBox(height: 260, child: ClipRRect(borderRadius: BorderRadius.circular(8), child: _map(c))),
          if (_picked != null)
            Padding(padding: const EdgeInsets.only(top: 6), child: Row(children: [
              Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text(_picked!.label, style: const TextStyle(fontWeight: FontWeight.w700)),
                Text('${_picked!.reading} · ${_picked!.river}', style: const TextStyle(color: Colors.white38, fontSize: 11, fontFamily: 'monospace')),
              ])),
              TextButton(onPressed: () => setState(() { c.cameraId = _picked!.id; c.state = 'CAMERA'; c.imageSeq++; }),
                  child: const Text('開く')),
            ]))
          else const Padding(padding: EdgeInsets.only(top: 6),
              child: Text('印を押すと読み方が出ます。拡大すると名前が並びます。', style: TextStyle(color: Colors.white24, fontSize: 10))),
        ]
        else ...[
          TextField(
            onChanged: (v) => setState(() => _filter = v),
            style: const TextStyle(fontSize: 13),
            decoration: const InputDecoration(isDense: true, prefixIcon: Icon(Icons.search, size: 18),
              hintText: '地点名でしぼる', border: OutlineInputBorder()),
          ),
          const SizedBox(height: 6),
          SizedBox(height: 220, child: ListView(children: [
            for (final s in sites.where((s) => _filter.isEmpty || s.label.contains(_filter) || s.reading.contains(_filter)))
              InkWell(
                onTap: () => setState(() { c.cameraId = s.id; c.state = 'CAMERA'; c.imageSeq++; }),
                child: Padding(padding: const EdgeInsets.symmetric(vertical: 4), child: Row(children: [
                  Expanded(child: Text(s.label, style: const TextStyle(fontSize: 14))),
                  Text(s.reading, style: const TextStyle(color: Colors.white38, fontSize: 11, fontFamily: 'monospace')),
                ])),
              ),
          ])),
          const Text('タップでも開けますが、本来は声だけで届きます', style: TextStyle(color: Colors.white24, fontSize: 10)),
        ],
      ]),
    );
  }

  Widget _map(Console c) {
    final pts = [for (final s in c.sites) if (s.lat != null && s.lng != null) s];
    if (pts.isEmpty) return const Center(child: Text('位置情報がありません', style: TextStyle(color: Colors.white38)));
    final lat = pts.map((s) => s.lat!).reduce((a, b) => a + b) / pts.length;
    final lng = pts.map((s) => s.lng!).reduce((a, b) => a + b) / pts.length;
    return FlutterMap(
      mapController: _mapCtl,
      options: MapOptions(initialCenter: LatLng(lat, lng), initialZoom: 9.5,
          onPositionChanged: (pos, _) { if ((pos.zoom - _zoom).abs() > .4) setState(() => _zoom = pos.zoom); }),
      children: [
        TileLayer(urlTemplate: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png', userAgentPackageName: 'com.genaicraft.openvons',
            tileProvider: NetworkTileProvider()),
        MarkerLayer(markers: [
          for (final s in pts)
            Marker(point: LatLng(s.lat!, s.lng!), width: _zoom >= 11 ? 130 : 22, height: _zoom >= 11 ? 44 : 22,
              child: GestureDetector(
                onTap: () => setState(() => _picked = s),
                child: Column(mainAxisSize: MainAxisSize.min, children: [
                  Icon(Icons.place, size: 18, color: _picked?.id == s.id ? warn : acc),
                  // 名前と読みは拡大したときだけ (縮小時に重なって読めなくなるため)。
                  // 読み方が分からないと声に出せないので、地図でも併記する
                  if (_zoom >= 11) Container(
                    padding: const EdgeInsets.symmetric(horizontal: 3, vertical: 1),
                    color: Colors.black.withValues(alpha: .62),
                    child: Column(mainAxisSize: MainAxisSize.min, children: [
                      Text(s.label, maxLines: 1, overflow: TextOverflow.ellipsis,
                          style: const TextStyle(fontSize: 9, height: 1.1, color: Colors.white)),
                      if (s.reading.isNotEmpty)
                        Text(s.reading, maxLines: 1, overflow: TextOverflow.ellipsis,
                            style: const TextStyle(fontSize: 8, height: 1.1, color: Color(0xFF9FD8FF),
                                fontFamily: 'monospace')),
                    ]),
                  ),
                ]),
              )),
        ]),
        const RichAttributionWidget(attributions: [TextSourceAttribution('OpenStreetMap contributors')]),
      ],
    );
  }

  /// いま受け付けている言い方の一覧。覚えなくても画面を見れば分かるように常に出す。
  Widget _sayable(Console? c) {
    final e = widget.engine;
    if (!e.ready || c == null) return const SizedBox.shrink();
    final hyps = List<Map<String, dynamic>>.from((e.sets[c.state]?['hyps'] ?? const []) as List);
    final byIntent = <String, List<String>>{};
    for (final h in hyps) {
      var i = h['i'] as String;
      if (i == 'select_camera') continue;
      if (i.startsWith('pan_')) i = 'pan';    // 方向 8 × 量 3 を 1 行にまとめる
      final t = h['t'] as String;
      final list = byIntent[i] ??= [];
      if (!list.contains(t)) list.add(t);     // 同じ表示文が複数の読みを持つので重複を落とす
    }
    for (final v in byIntent.values) {
      v.sort((a, b) => a.length.compareTo(b.length));
    }
    const names = {
      'upstream': '上流へ移る', 'downstream': '下流へ移る', 'refresh': '画像を更新', 'zoom_in': '拡大',
      'zoom_out': '縮小', 'back': '地図に戻る (いつでも)', 'favorite': 'お気に入り登録 (要確認)', 'help': 'ヘルプ',
      'pan': '地図を動かす',
      'yes': 'はい', 'no': 'いいえ', 'select_camera': '地点を選ぶ',
    };
    return Card(child: Padding(padding: const EdgeInsets.all(12), child: Column(
      crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(children: [
          const Icon(Icons.checklist, size: 16, color: Colors.white54),
          const SizedBox(width: 6),
          Text('いま言えること (${c.stateLabel})', style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 13)),
        ]),
        const SizedBox(height: 6),
        if (c.state == 'MAP')
          Padding(padding: const EdgeInsets.only(bottom: 4), child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
            const SizedBox(width: 108, child: Text('地点を選ぶ', style: TextStyle(fontSize: 12, color: Colors.white70))),
            Expanded(child: Text('上の地図か一覧にある ${c.sites.length} 地点の名前', style: const TextStyle(fontSize: 12))),
          ])),
        for (final e2 in byIntent.entries)
          Padding(padding: const EdgeInsets.symmetric(vertical: 2), child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
            SizedBox(width: 108, child: Text(names[e2.key] ?? e2.key, style: const TextStyle(fontSize: 12, color: Colors.white70))),
            Expanded(child: Text(
                e2.key == 'pan'
                    ? '「ちょっと右に動かして」「大きく上へ」 上下左右と斜め × ちょっと / 大きく'
                    : e2.value.take(3).map((t) => '「$t」').join(' '),
                style: const TextStyle(fontSize: 12))),
          ])),
        const SizedBox(height: 4),
        Text('この一覧が、いま音声が選べる全ての選択肢です。状態が変わると中身も変わります。',
            style: const TextStyle(color: Colors.white38, fontSize: 10)),
      ])));
  }

  // ------------------------------------------------------------- 集計
  Widget _tallyBar(Tally t) => Card(
    color: panel,
    child: Padding(padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10), child: Column(
      crossAxisAlignment: CrossAxisAlignment.start, children: [
        Text('${t.utterances} 回 話しかけた結果', style: const TextStyle(fontSize: 13, color: Colors.white70)),
        const SizedBox(height: 6),
        Wrap(spacing: 8, runSpacing: 6, children: [
          _chip('実行 ${t.executed}', ok),
          _chip('確認 ${t.confirmed}', warn),
          _chip('はねた ${t.rejected}', Colors.white38),
          _chip('従来なら誤作動 ${t.naiveMisfire}', t.naiveMisfire > 0 ? bad : Colors.white24),
        ]),
        if (t.naiveMisfire > 0) ...[
          const SizedBox(height: 8),
          Text('「文字起こしに一番近いものを選ぶ」やり方だと、この ${t.naiveMisfire} 回は意図しない操作が走っていました。',
              style: const TextStyle(color: Colors.white54, fontSize: 11, height: 1.4)),
        ],
      ])),
  );

  // ------------------------------------------------------------- 1 発話のカード
  Widget _turnCard(_Turn t) {
    final r = t.r;
    final lvl = r.action == 'execute' ? ok : r.action == 'confirm' ? warn : bad;
    return Card(child: Padding(padding: const EdgeInsets.all(12), child: Column(
      crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(children: [
          const Text('聞こえた音: ', style: TextStyle(color: Colors.white38, fontSize: 11)),
          Expanded(child: Text(r.freeKana.isEmpty ? '(無音)' : r.freeKana,
              style: const TextStyle(color: Colors.white70, fontSize: 12, fontFamily: 'monospace'))),
          Text('${r.totalMs}ms', style: const TextStyle(color: Colors.white24, fontSize: 11, fontFamily: 'monospace')),
        ]),
        const SizedBox(height: 8),
        // 2 列の比較
        IntrinsicHeight(child: Row(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
          Expanded(child: _column('従来のやり方', '一番近い言い方を選ぶ',
              r.naiveText, r.naiveWouldMisfire ? '誤作動' : '同じ結果',
              r.naiveWouldMisfire ? bad : Colors.white38,
              '近さ ${(r.naiveSim * 100).toStringAsFixed(0)}%')),
          const VerticalDivider(width: 18, color: Colors.white12),
          Expanded(child: _column('openvons', '確率で答える (該当なしを含む)',
              r.action == 'reject' || r.action == 'none' ? '該当なし' : r.text,
              actionLabel(r.action), lvl,
              '${((r.action == 'reject' || r.action == 'none' ? r.noneProb : r.prob) * 100).toStringAsFixed(1)}%')),
        ])),
        const SizedBox(height: 10),
        for (final p in r.top.take(3)) _bar(p.key, p.value, acc),
        _bar('該当なし', r.noneProb, Colors.white24),
        if (t.effect.isNotEmpty) ...[
          const SizedBox(height: 8),
          Row(children: [
            const Icon(Icons.subdirectory_arrow_right, size: 14, color: Colors.white38),
            const SizedBox(width: 4),
            Expanded(child: Text(t.effect, style: const TextStyle(color: Colors.white70, fontSize: 13))),
          ]),
        ],
      ])));
  }

  Widget _column(String title, String sub, String result, String verdict, Color c, String detail) =>
    Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Text(title, style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w700, color: Colors.white70)),
      Text(sub, style: const TextStyle(fontSize: 10, color: Colors.white24, height: 1.3)),
      const SizedBox(height: 6),
      Text(result, style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w700)),
      const SizedBox(height: 4),
      Wrap(spacing: 6, runSpacing: 2, crossAxisAlignment: WrapCrossAlignment.center, children: [
        _chip(verdict, c),
        Text(detail, style: const TextStyle(fontSize: 10, color: Colors.white38, fontFamily: 'monospace')),
      ]),
    ]);
}

Widget _chip(String t, Color c) => Container(
  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 3),
  decoration: BoxDecoration(color: c.withValues(alpha: .18), borderRadius: BorderRadius.circular(99), border: Border.all(color: c)),
  child: Text(t, style: TextStyle(color: c, fontWeight: FontWeight.w700, fontSize: 12)));

Widget _bar(String label, double p, Color c) => Padding(
  padding: const EdgeInsets.symmetric(vertical: 2),
  child: Row(children: [
    Expanded(child: Stack(children: [
      Container(height: 18, decoration: BoxDecoration(color: Colors.black26, borderRadius: BorderRadius.circular(4))),
      FractionallySizedBox(widthFactor: p.clamp(0, 1),
        child: Container(height: 18, decoration: BoxDecoration(color: c.withValues(alpha: .75), borderRadius: BorderRadius.circular(4)))),
      Positioned(left: 6, top: 0, bottom: 0, child: Align(child: Text(label, style: const TextStyle(fontSize: 12)))),
    ])),
    SizedBox(width: 52, child: Text('${(p * 100).toStringAsFixed(1)}%', textAlign: TextAlign.right,
      style: const TextStyle(fontSize: 12, color: Colors.white70, fontFamily: 'monospace'))),
  ]));

/// ---------------------------------------------------------------- 映像の状態
class VisionPage extends StatefulWidget {
  final VisionEngine engine;
  final String server;
  final Console? console;
  const VisionPage({super.key, required this.engine, required this.server, required this.console});
  @override
  State<VisionPage> createState() => _VisionPageState();
}

class _VisionPageState extends State<VisionPage> {
  CameraController? _cam;
  Timer? _timer;
  List<VisionAnswer> _answers = [];
  int _embedMs = 0, _matchUs = 0, _nQuestions = 0;
  bool _busy = false, _auto = false, _live = false;
  String _msg = '';
  int _siteIdx = 0;
  Uint8List? _liveBytes;

  @override
  void dispose() { _timer?.cancel(); _cam?.dispose(); super.dispose(); }

  Future<void> _startCamera() async {
    try {
      final cams = await availableCameras();
      if (cams.isEmpty) { setState(() => _msg = 'カメラが見つかりません'); return; }
      final c = CameraController(cams.first, ResolutionPreset.medium, enableAudio: false);
      await c.initialize();
      setState(() { _cam = c; _msg = ''; });
    } catch (e) { setState(() => _msg = 'カメラを開けません: $e'); }
  }

  /// 河川ライブカメラの画像を、同じ質問セットで判定する (監視カメラの状態分類そのもの)。
  Future<void> _analyzeLive() async {
    final sites = widget.console?.sites ?? [];
    if (sites.isEmpty) { setState(() => _msg = '地点一覧がまだ取れていません'); return; }
    if (_busy || !widget.engine.ready) {
      if (!widget.engine.ready) setState(() => _msg = '設定タブでモデルを取得してください');
      return;
    }
    _busy = true;
    try {
      final s = sites[_siteIdx % sites.length];
      setState(() => _msg = '${s.label} の画像を取得中…');
      final bytes = (await http.get(Uri.parse('${widget.server}/api/image/${s.id}'))).bodyBytes;
      final im = img.decodeImage(bytes);
      if (im == null) { setState(() => _msg = '画像を読めません'); return; }
      final res = await widget.engine.run(im);
      if (mounted) {
        setState(() {
          _liveBytes = bytes;
          _answers = res.$1; _embedMs = res.$2; _matchUs = res.$3; _nQuestions = res.$1.length;
          _msg = '${s.label} (${s.river})';
        });
      }
    } catch (e) {
      if (mounted) setState(() => _msg = 'エラー: $e');
    } finally { _busy = false; }
  }

  Future<void> _analyze() async {
    if (_live) return _analyzeLive();
    if (_busy || _cam == null || !widget.engine.ready) {
      if (!widget.engine.ready) setState(() => _msg = '設定タブでモデルを取得してください');
      return;
    }
    _busy = true;
    try {
      final shot = await _cam!.takePicture();
      final bytes = await File(shot.path).readAsBytes();
      final im = img.decodeImage(bytes);
      if (im == null) return;
      final res = await widget.engine.run(im);
      if (mounted) setState(() { _answers = res.$1; _embedMs = res.$2; _matchUs = res.$3; _nQuestions = res.$1.length; _msg = ''; });
      await File(shot.path).delete();
    } catch (e) {
      if (mounted) setState(() => _msg = 'エラー: $e');
    } finally { _busy = false; }
  }

  void _toggleAuto(bool v) {
    setState(() => _auto = v);
    _timer?.cancel();
    if (v) _timer = Timer.periodic(const Duration(seconds: 4), (_) => _analyze());
  }

  @override
  Widget build(BuildContext context) {
    final sites = widget.console?.sites ?? [];
    return ListView(padding: const EdgeInsets.all(12), children: [
      // どの映像を見るか
      SegmentedButton<bool>(
        segments: const [
          ButtonSegment(value: false, label: Text('この端末のカメラ')),
          ButtonSegment(value: true, label: Text('河川ライブカメラ')),
        ],
        selected: {_live},
        onSelectionChanged: (v) => setState(() {
          _live = v.first; _answers = []; _msg = '';
          widget.engine.use(_live ? 'kasen' : 'general');   // 見る映像に合わせて質問セットを替える
          if (_auto) _toggleAuto(false);
        }),
      ),
      const SizedBox(height: 10),

      if (_live) ...[
        if (_liveBytes != null)
          ClipRRect(borderRadius: BorderRadius.circular(10), child: Image.memory(_liveBytes!, fit: BoxFit.contain))
        else
          Container(height: 150, decoration: BoxDecoration(color: panel, borderRadius: BorderRadius.circular(10)),
            child: const Center(child: Text('「いま判定する」で最新の画像を取り込みます', style: TextStyle(color: Colors.white38, fontSize: 12)))),
        const SizedBox(height: 8),
        if (sites.isNotEmpty)
          DropdownButton<int>(
            isExpanded: true, value: _siteIdx.clamp(0, sites.length - 1), dropdownColor: panel,
            items: [for (var i = 0; i < sites.length; i++) DropdownMenuItem(value: i, child: Text(sites[i].label, overflow: TextOverflow.ellipsis))],
            onChanged: (v) => setState(() => _siteIdx = v ?? 0)),
      ] else if (_cam == null)
        FilledButton.icon(onPressed: _startCamera, icon: const Icon(Icons.videocam), label: const Text('カメラを開始'),
          style: FilledButton.styleFrom(backgroundColor: acc, foregroundColor: Colors.black, minimumSize: const Size.fromHeight(52)))
      else
        AspectRatio(aspectRatio: 4 / 3, child: ClipRRect(borderRadius: BorderRadius.circular(10), child: CameraPreview(_cam!))),

      if (_live || _cam != null) ...[
        const SizedBox(height: 8),
        Row(children: [
          FilledButton(onPressed: _analyze, style: FilledButton.styleFrom(backgroundColor: acc, foregroundColor: Colors.black),
            child: const Text('いま判定する')),
          const SizedBox(width: 12),
          Switch(value: _auto, onChanged: _toggleAuto, activeThumbColor: acc),
          const Text('4 秒ごと', style: TextStyle(color: Colors.white70)),
        ]),
      ],
      if (_msg.isNotEmpty) Padding(padding: const EdgeInsets.only(top: 8),
          child: Text(_msg, style: const TextStyle(color: Colors.white70, fontSize: 13))),

      // 内訳 — 「質問を増やしてもほぼタダ」がこの行で見える
      if (_nQuestions > 0) ...[
        const SizedBox(height: 10),
        Card(color: panel, child: Padding(padding: const EdgeInsets.all(10),
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text('質問セット: ${widget.engine.setTitle}', style: const TextStyle(fontSize: 12, color: Colors.white70)),
            const SizedBox(height: 2),
            Text('画像を数値にする $_embedMs ms  +  $_nQuestions 個の質問に答える ${(_matchUs / 1000).toStringAsFixed(1)} ms',
                style: const TextStyle(fontSize: 13, fontFamily: 'monospace')),
            const SizedBox(height: 4),
            const Text('画像を数値にするのは 1 回だけ。質問は、その数値と選択肢を照らし合わせるだけなので、'
                '何個増やしてもほとんど時間が増えません。文章で答えさせる方式との差はここに出ます。',
                style: TextStyle(color: Colors.white54, fontSize: 11, height: 1.4)),
          ]))),
      ],

      const SizedBox(height: 10),
      for (final a in _answers) Card(child: Padding(padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Row(children: [
            SizedBox(width: 92, child: Text(a.title, style: const TextStyle(color: Colors.white70, fontSize: 13))),
            Expanded(child: Text(a.label, style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 16))),
            _chip(a.level, a.level == '確定' ? ok : a.level == '要確認' ? warn : bad),
          ]),
          const SizedBox(height: 4),
          for (final c in a.all) _bar(c.key, c.value, acc),
        ]))),

      if (_answers.isEmpty) const Padding(padding: EdgeInsets.only(top: 12),
        child: Text('物体の検出ではなく、場面の「状態」を有限の選択肢で答えます。'
            '片付き具合・照明・混み具合・路面・扉の開閉・画面の正常/エラーなど。\n'
            '自信が無いときは「不明」と言います。答えを作らずに、選択肢の中から確率で選ぶからです。',
          style: TextStyle(color: Colors.white54, fontSize: 12, height: 1.5))),
    ]);
  }
}

/// ---------------------------------------------------------------- 設定
class SettingsPage extends StatelessWidget {
  final String server, status;
  final double progress;
  final bool loading, useGpu, speak;
  final void Function(String) onServer;
  final void Function(bool) onGpu;
  final void Function(bool) onSpeak;
  final Future<void> Function() onDownload;
  const SettingsPage({super.key, required this.server, required this.status, required this.progress,
    required this.loading, required this.useGpu, required this.onServer, required this.onGpu,
    required this.onDownload, required this.speak, required this.onSpeak});

  @override
  Widget build(BuildContext context) => ListView(padding: const EdgeInsets.all(14), children: [
    const Text('サーバー', style: TextStyle(color: Colors.white70)),
    TextFormField(initialValue: server, onFieldSubmitted: onServer,
      decoration: const InputDecoration(border: OutlineInputBorder(), isDense: true, helperText: 'モデルと選択肢の配布元 (推論は端末内)')),
    const SizedBox(height: 16),
    SwitchListTile(value: useGpu, onChanged: onGpu, activeThumbColor: acc,
      title: const Text('端末の加速器を使う'), subtitle: const Text('Android: NNAPI / iOS: Core ML。切ると CPU のみ'),
      contentPadding: EdgeInsets.zero),
    SwitchListTile(value: speak, onChanged: onSpeak, activeThumbColor: acc,
      title: const Text('声で返す'), subtitle: const Text('確認は「〜でよろしいですか」と聞き返します。読み上げ中はマイクを止めます'),
      contentPadding: EdgeInsets.zero),
    const SizedBox(height: 8),
    FilledButton.icon(onPressed: loading ? null : onDownload, icon: const Icon(Icons.download),
      label: const Text('モデルを取得して読み込む'),
      style: FilledButton.styleFrom(backgroundColor: acc, foregroundColor: Colors.black, minimumSize: const Size.fromHeight(52))),
    const SizedBox(height: 10),
    LinearProgressIndicator(value: progress == 0 ? null : progress, backgroundColor: Colors.black26, color: acc),
    const SizedBox(height: 8),
    Text(status, style: const TextStyle(color: Colors.white70)),
    const SizedBox(height: 20),
    const Text('この端末で動くもの', style: TextStyle(color: Colors.white70)),
    const Text('・音声: 前処理 (log-mel) → 認識 → 候補の採点 → 校正 → 判断\n'
        '・映像: 画像の埋め込み → 選択肢との内積 → 校正 → 判断\n'
        'サーバーに送るのはモデルと選択肢の要求だけで、音声も映像も端末から出ません。\n'
        'モデルを取り込んだ後は、機内モードでも音声の判断は動きます (河川カメラの画像取得だけは通信が要ります)。',
      style: TextStyle(color: Colors.white54, fontSize: 12, height: 1.5)),
  ]);
}
