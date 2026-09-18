/// サーバーから取るもの: モデルの一覧と実体、状態ごとのコマンド集合、復号表、画像の質問セット。
/// 推論そのものは端末で行う。サーバーは「何を選べるか」を配るだけ。
library;

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';

class Api {
  String base;
  Api(this.base);

  Uri _u(String p) => Uri.parse('$base$p');

  /// 圏外のまま無期限に待たない。待たせるより、はっきり失敗させて理由を出す
  static const timeout = Duration(seconds: 20);

  String _unreachable(String why) =>
      'サーバーに繋がりません: ${Uri.parse(base).host} ($why)。設定タブのサーバー欄と通信の状態を確認してください';

  Future<dynamic> _get(String path) async {
    try {
      final r = await http.get(_u(path)).timeout(timeout);
      if (r.statusCode != 200) throw '$path が ${r.statusCode} を返しました';
      return jsonDecode(utf8.decode(r.bodyBytes));
    } on TimeoutException {
      throw _unreachable('応答がありません');
    } on SocketException catch (e) {
      throw _unreachable(e.osError?.message ?? e.message);
    } on http.ClientException catch (e) {
      // http パッケージは SocketException を包むので、名前解決の失敗もここに来る
      throw _unreachable(e.message.split(',').first);
    }
  }

  Future<Map<String, dynamic>> config() async => Map<String, dynamic>.from(await _get('/api/config'));
  Future<Map<String, dynamic>> manifest() async => Map<String, dynamic>.from(await _get('/api/manifest'));
  Future<Map<String, dynamic>> commands({String app = 'kasen'}) async =>
      Map<String, dynamic>.from(await _get('/api/commands?app_name=$app'));
  Future<Map<String, dynamic>> sites() async => Map<String, dynamic>.from(await _get('/api/sites'));
  Future<Map<String, dynamic>> tokens() async => Map<String, dynamic>.from(await _get('/api/tokens'));
  Future<Map<String, dynamic>> visionChoices({String set = 'general'}) async =>
      Map<String, dynamic>.from(await _get('/api/vision/choices?set=$set'));
  Future<Map<String, dynamic>> visionHead(String task) async =>
      Map<String, dynamic>.from(await _get('/api/vision/head/$task'));
  Future<List<dynamic>> visionSets() async => await _get('/api/vision/sets') as List;

  /// モデルファイルを端末に落として、そのパスを返す (2 回目からはキャッシュ)。
  Future<String> ensureFile(String relPath, int expectedBytes, void Function(double) onProgress) async {
    final dir = await getApplicationSupportDirectory();
    final f = File('${dir.path}/models/$relPath');
    if (await f.exists() && (await f.length()) == expectedBytes) return f.path;
    await f.parent.create(recursive: true);
    final req = http.Request('GET', _u('/model/$relPath'));
    final res = await req.send().timeout(timeout);
    final sink = f.openWrite();
    var got = 0;
    await for (final chunk in res.stream) {
      got += chunk.length;
      sink.add(chunk);
      if (expectedBytes > 0) onProgress(got / expectedBytes);
    }
    await sink.close();
    return f.path;
  }
}

/// トークン id → バイト列。生成された id を繋いで UTF-8 に戻す (byte-level BPE なので途中で切れる)。
class TokenDecoder {
  final Map<int, List<int>> bytes;
  TokenDecoder(this.bytes);
  factory TokenDecoder.fromJson(Map<String, dynamic> j) =>
      TokenDecoder({for (final e in j.entries) int.parse(e.key): List<int>.from(e.value)});

  String decode(List<int> ids) {
    final buf = <int>[];
    for (final id in ids) {
      final b = bytes[id];
      if (b != null) buf.addAll(b);
    }
    try {
      return utf8.decode(buf);
    } catch (_) {
      return utf8.decode(buf, allowMalformed: true);
    }
  }
}
