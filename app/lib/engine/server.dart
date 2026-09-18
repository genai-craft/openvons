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

  Future<Map<String, dynamic>> config() async => jsonDecode(utf8.decode((await http.get(_u('/api/config'))).bodyBytes));
  Future<Map<String, dynamic>> manifest() async => jsonDecode(utf8.decode((await http.get(_u('/api/manifest'))).bodyBytes));
  Future<Map<String, dynamic>> commands({String app = 'kasen'}) async =>
      jsonDecode(utf8.decode((await http.get(_u('/api/commands?app_name=$app'))).bodyBytes));
  Future<Map<String, dynamic>> sites() async => jsonDecode(utf8.decode((await http.get(_u('/api/sites'))).bodyBytes));
  Future<Map<String, dynamic>> tokens() async => jsonDecode(utf8.decode((await http.get(_u('/api/tokens'))).bodyBytes));
  Future<Map<String, dynamic>> visionChoices({String set = 'general'}) async =>
      jsonDecode(utf8.decode((await http.get(_u('/api/vision/choices?set=$set'))).bodyBytes));
  Future<Map<String, dynamic>> visionHead(String task) async =>
      jsonDecode(utf8.decode((await http.get(_u('/api/vision/head/$task'))).bodyBytes));
  Future<List<dynamic>> visionSets() async => jsonDecode(utf8.decode((await http.get(_u('/api/vision/sets'))).bodyBytes)) as List;

  /// モデルファイルを端末に落として、そのパスを返す (2 回目からはキャッシュ)。
  Future<String> ensureFile(String relPath, int expectedBytes, void Function(double) onProgress) async {
    final dir = await getApplicationSupportDirectory();
    final f = File('${dir.path}/models/$relPath');
    if (await f.exists() && (await f.length()) == expectedBytes) return f.path;
    await f.parent.create(recursive: true);
    final req = http.Request('GET', _u('/model/$relPath'));
    final res = await req.send();
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
