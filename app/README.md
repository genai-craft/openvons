# openvons on-device app (Android / iOS)

A Flutter app that runs the whole openvons pipeline **inside the phone**: log-mel, recognition,
candidate scoring, calibration and the decision for voice; image embedding and choice matching for
vision. The server only hands out the models and the list of things that can be chosen.
No audio and no image leaves the device.

Three screens:

- **声で操作 (voice)** — a river-camera console. Pick a site by saying its name (map or list), then
  move upstream / downstream, refresh, zoom. Every utterance is shown twice: what the conventional
  approach would do (closest phrase to the transcription, no way to refuse) next to what openvons
  does (probabilities including "none of the above" → execute / confirm / reject), with a running
  count of how often the conventional approach would have fired the wrong command.
  "マイク無しで試す" plays four synthesized utterances so you can see the flow without speaking.
- **映像の状態 (vision)** — the phone camera or a live MLIT river camera, answered as scene *state*
  (not object detection) with calibrated probabilities. The timing is split into "embed the image"
  and "answer N questions" to show that questions are nearly free.
- **設定 (settings)** — which server to fetch models from, and whether to use the device accelerator.

## Run it

```bash
flutter pub get
flutter run                     # a connected Android device or iOS simulator/device
flutter build apk --release     # Android (111 MB; --split-per-abi gives ~40 MB per ABI)
flutter build ios --release     # iOS (needs macOS and an Apple signing identity)
```

The release build is signed with the debug key (the Flutter template default); add your own signing
config in `android/app/build.gradle.kts` before distributing it.

Requires Flutter 3.35 or newer, and for Android a JDK 17+ with `ANDROID_HOME` set.

On first launch the app downloads about 300 MB of models from the server in the settings tab
(default `https://ondevice.openvons.com`, the public demo) and caches them. After that the voice
decision works with no network at all; only the river camera images need one.

To serve the models yourself, run the on-device server from the repository root and point the
app at it:

```bash
scripts/serve_ondevice.sh start 0 8606      # /api/config /api/manifest /api/commands /api/sites /model/...
```

## Layout

```
lib/engine/kana.dart      normalisation and shortlisting by kana edit distance
lib/engine/decision.dart  4-parameter calibration and the execute / confirm / reject policy
lib/engine/server.dart    fetching models, command sets, site list, token table
lib/engine/voice.dart     mel → encoder → greedy → candidate scoring → calibration → decision
lib/engine/vision.dart    preprocess → image embedding → dot product with choice embeddings
lib/engine/console.dart   the river-camera state machine (MAP / CAMERA / CONFIRM)
lib/pages.dart            the three screens
```

Design notes, the ONNX export scripts and the traps we hit (handing the full vocabulary logits to
the app crashes it) are in [../docs/ondevice_app.md](../docs/ondevice_app.md) (Japanese).
