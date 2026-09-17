/// openvons — 声と映像を、端末の中で有限の選択肢に確率で答えるアプリ。
/// サーバーはモデルと「選べるものの一覧」を配るだけで、推論は端末内で完結する。
library;

import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'engine/console.dart';
import 'engine/server.dart';
import 'engine/vision.dart';
import 'engine/voice.dart';
import 'pages.dart';

const kDefaultServer = 'https://ondevice.openvons.com';
const bg = Color(0xFF0B0E13), panel = Color(0xFF141B24), line = Color(0xFF222C38);
const acc = Color(0xFF4CC2FF), ok = Color(0xFF3FB950), warn = Color(0xFFD29922), bad = Color(0xFFF85149);

void main() => runApp(const OpenvonsApp());

class OpenvonsApp extends StatelessWidget {
  const OpenvonsApp({super.key});
  @override
  Widget build(BuildContext context) => MaterialApp(
        title: 'openvons',
        debugShowCheckedModeBanner: false,
        theme: ThemeData(
          brightness: Brightness.dark,
          scaffoldBackgroundColor: bg,
          colorScheme: const ColorScheme.dark(primary: acc, surface: panel),
          cardColor: panel,
          appBarTheme: const AppBarTheme(backgroundColor: panel, elevation: 0),
        ),
        home: const HomePage(),
      );
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});
  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> with SingleTickerProviderStateMixin {
  late TabController _tabs;
  final voice = VoiceEngine();
  final vision = VisionEngine();
  String server = kDefaultServer;
  Console? console;
  String status = 'モデル未取得';
  double progress = 0;
  bool loading = false, useGpu = true;

  @override
  void initState() {
    super.initState();
    _tabs = TabController(length: 3, vsync: this);
    SharedPreferences.getInstance().then((p) {
      setState(() => server = p.getString('server') ?? kDefaultServer);
      _loadSites();
      _download();   // 取得済みなら読み込むだけで終わる (毎回ボタンを押させない)
    });
  }

  List<OrtProvider>? get providers => useGpu
      ? (Platform.isAndroid ? [OrtProvider.NNAPI, OrtProvider.XNNPACK, OrtProvider.CPU]
                            : [OrtProvider.CORE_ML, OrtProvider.XNNPACK, OrtProvider.CPU])
      : [OrtProvider.CPU];

  /// 指令卓に出す地点一覧 (モデルとは別に、すぐ取れる)。
  Future<void> _loadSites() async {
    try {
      final j = await Api(server).sites();
      final list = [for (final s in List<Map<String, dynamic>>.from(j['sites'])) Site.fromJson(s)];
      if (mounted) setState(() => console = Console(list, (j['scope'] ?? '') as String));
    } catch (_) {
      // 通信できない場合は指令卓なしで動く (音声の判断そのものは端末内で完結する)
    }
  }

  Future<void> _download() async {
    setState(() { loading = true; status = 'サーバーに問い合わせ中…'; progress = 0; });
    try {
      final api = Api(server);
      if (console == null) await _loadSites();
      final cfg = await api.config();
      final man = await api.manifest();
      final cmds = await api.commands();
      final toks = await api.tokens();
      final vsets = await api.visionSets();
      final vch = <String, Map<String, dynamic>>{};
      for (final v in vsets) {
        final key = (v as Map)['key'] as String;
        vch[key] = await api.visionChoices(set: key);
      }
      final voiceFiles = List<Map<String, dynamic>>.from(man['voice']['files']);
      final vname = man['voice']['name'] as String;
      String pick(String suffix) => voiceFiles.firstWhere((f) => (f['path'] as String).endsWith(suffix))['path'] as String;
      int size(String p) => voiceFiles.firstWhere((f) => f['path'] == p)['bytes'] as int;
      final melP = pick('mel.onnx');
      final encP = '$vname/onnx/encoder_model_quantized.onnx';
      final decP = '$vname/onnx/decoder_head_quantized.onnx';
      final visionFiles = List<Map<String, dynamic>>.from(man['vision']['files']);
      final imgP = visionFiles.firstWhere((f) => (f['path'] as String).endsWith('image_encoder_quantized.onnx'))['path'] as String;
      final imgSize = visionFiles.firstWhere((f) => f['path'] == imgP)['bytes'] as int;

      setState(() => status = '音声モデルを取得中 (${(size(encP) + size(decP)) ~/ 1000000}MB)');
      final mel = await api.ensureFile(melP, size(melP), (p) => setState(() => progress = p * .05));
      final enc = await api.ensureFile(encP, size(encP), (p) => setState(() => progress = .05 + p * .30));
      final dec = await api.ensureFile(decP, size(decP), (p) => setState(() => progress = .35 + p * .30));
      setState(() => status = '画像モデルを取得中 (${imgSize ~/ 1000000}MB)');
      final imgModel = await api.ensureFile(imgP, imgSize, (p) => setState(() => progress = .65 + p * .30));
      final pre = await api.ensureFile('siglip2-base-img/preprocess.json', 0, (_) {});

      setState(() => status = 'モデルを読み込み中…');
      await voice.load(melPath: mel, encPath: enc, decPath: dec, config: cfg,
          tokenDecoder: TokenDecoder.fromJson(toks), commandSets: cmds, providers: providers);
      await vision.load(modelPath: imgModel, preprocess: await _readJson(pre), choiceDocs: vch, providers: providers);
      setState(() { status = '準備完了 (${useGpu ? "GPU/NNAPI" : "CPU"})'; progress = 1; loading = false; });
    } catch (e) {
      setState(() { status = 'エラー: $e'; loading = false; });
    }
  }

  Future<Map<String, dynamic>> _readJson(String path) async => _json(await File(path).readAsString());

  @override
  Widget build(BuildContext context) => Scaffold(
        appBar: AppBar(
          title: const Text('openvons', style: TextStyle(fontWeight: FontWeight.w800)),
          bottom: TabBar(controller: _tabs, indicatorColor: acc, tabs: const [
            Tab(text: '声で操作'), Tab(text: '映像の状態'), Tab(text: '設定'),
          ]),
        ),
        body: TabBarView(controller: _tabs, children: [
          VoicePage(engine: voice, status: status, server: server, console: console),
          VisionPage(engine: vision, server: server, console: console),
          SettingsPage(
            server: server, status: status, progress: progress, loading: loading, useGpu: useGpu,
            onServer: (v) async { setState(() => server = v); (await SharedPreferences.getInstance()).setString('server', v); await _loadSites(); },
            onGpu: (v) => setState(() => useGpu = v),
            onDownload: _download,
          ),
        ]),
      );
}

Map<String, dynamic> _json(String s) => Map<String, dynamic>.from(jsonDecode(s) as Map);
