# openvons (open-Jev) — a decision layer that answers finite choices with probabilities

**日本語の README はこちら → [README_ja.md](README_ja.md)**

Instead of asking an LLM, VLM or ASR model to *generate text*, openvons makes it **answer a finite set of options with a probability for each**.
"None of the above" is always one of the options, and the calibrated probabilities are split into **execute / confirm / reject**.

The name comes from Jevons (the marginal-utility economist). The project started as *open-Jev*; the package is `openvons`, and `import jev` still works as an alias.

The same idea is implemented for three input types:

| Module | What it chooses | How | Measured (see docs/) |
|---|---|---|---|
| `openvons.lm` | intent, tool selection, scores (Noul / Choice / Score) | frozen LLM + a trained output head | 4B frozen + head 0.916 vs 27B zero-shot 0.875; 8 questions in 22.6 ms ([lm_benchmark](docs/lm_benchmark.md)) |
| `openvons.vision` | image attributes (age, gender, orientation, baggage, …) | frozen vision encoder (407M) + a 25k-parameter head | beats 27B zero-shot at 1/34 the VRAM and 36x the speed ([vision_summary](docs/vision_summary.md)) |
| `openvons.voice` | tens of thousands of proper nouns + state-dependent commands | batch scoring of candidates with kana-whisper + calibration with a "none" option | 50 ms, 99–100% after calibration, 100% rejection of phone chatter ([voice_evaluation](docs/voice_evaluation.md)) |

The shared layer `openvons.core` provides the Noul / Choice / Score representation (`Question`), the decision policy (`decide`, confidence gating),
calibration (temperature, isotonic, and **calibration with an explicit "none" option**) and metrics (ECE / Brier / NLL / macro-F1).

## JevPick — the same "pick from a finite menu" idea, applied to speeding up text generation

[docs/jevpick/README.md](docs/jevpick/README.md). Instead of a draft model, JevPick builds a **menu of likely continuations** (from the tool
definitions, past outputs, or the checkpoint's own MTP head), picks one from the target model's hidden state, and lets the target verify it in
one pass. Measured on tool calling (Qwen3-4B / Qwen3.8-27B, bf16 and 4-bit): the menu contains the true continuation 90% of the time, JevPick
picks the right one 88% (vs 64% for a frequency rule), **3.2–4.8x faster decode with byte-identical output**, and a picker trained on bf16
transfers to FP8 / NF4 hidden states. Reports: [phase 1](docs/jevpick/phase1_oracle_report.md), [phase 2](docs/jevpick/phase2_report.md),
[phase 3 (27B, quantized, MTP / DFlash2, Flash-Next)](docs/jevpick/phase3_27b_report.md), [all results](docs/jevpick/summary_table.md).

## Getting started

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev]"   # install torch from the CUDA index that matches your GPU
.venv/bin/python -m pytest -q tests/test_voice_core.py
```

### Text decision demo

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507 --served-model-name qwen3-4b --port 8300 --logprobs-mode processed_logprobs   # any OpenAI-compatible server
scripts/serve_text.sh start 8604
```

### Voice command demos

```bash
docker run -d --name voicevox -p 127.0.0.1:50021:50021 voicevox/voicevox_engine:cpu-latest   # TTS used for pre-training
JEV_APP=examples.stations.app     scripts/serve_demo.sh start 0 8601   # railway map driven by station names (general audience)
JEV_APP=examples.kasen.app        scripts/serve_demo.sh start 0 8603   # river live cameras (MLIT Kanto, real images)
JEV_APP=examples.road_cameras.app scripts/serve_demo.sh start 0 8600   # road camera monitoring (synthetic catalog, industrial example)
```

The microphone requires https or localhost (`ssh -L 8601:localhost:8601 <host>`). On a phone, "Add to Home Screen" opens it as a PWA.
Utterances captured through the microphone are stored under `state/<app>/utts/` (`JEV_DUMP_UTTS=1`, the default) for recalibration and measurement.

| Demo | Live | What it does |
|---|---|---|
| Text decisions | https://text.openvons.com | Paste a support email / inspection note / meeting memo, ask several Noul / Choice / Score questions at once, get probabilities per question. Compare "probabilities" vs "JSON generation" vs "one pick", and measure live how latency changes as you add questions (8 questions: 40 ms vs 900 ms) |
| Railway map by voice | https://eki.openvons.com | 8,987 stations (station_database, CC BY 4.0). Say a station to zoom; next / previous station; switch line; add favorite (asks for confirmation) |
| River live cameras (kasen) | https://kasen.openvons.com | 199 live cameras of MLIT Kanto Regional Development Bureau (Tone, Arakawa, Naka, Kuji, Watarase, Kasumigaura). Say a site name to show its live image; move upstream / downstream; refresh; zoom; favorite (asks for confirmation) |
| Video judge | https://judge.openvons.com | Slice a 15 s – 1 min video into windows and answer 10 checks (fight, shoplifting, no harness at height, no hard hat, double dribble, person collapsed, intrusion, fire/smoke, phone while driving, abandoned bag) as yes / no / cannot tell, shown on a timeline. Live camera mode judges frames in real time, including robot navigation (left / right / straight / stop) |
| Face and body attributes (vision) | https://kao.openvons.com | Webcam or upload. Age (9 bins) and gender per face (FairFace head), gender / age group / orientation / baggage for the body (PA-100K head), each with calibrated probabilities and a decided / check / unknown level. Images are not stored |

An application is just `examples/<name>/app.py` (intents, states, entities) plus `static/`; recognition, calibration, pre-training and the server are shared.
Design and evaluation: [docs/voice_design.md](docs/voice_design.md) / [docs/voice_evaluation.md](docs/voice_evaluation.md) (Japanese).

How the voice pipeline works:

1. kana-whisper transcribes freely into katakana (a reading, not spelling), giving a free hypothesis and its log-likelihood.
2. The current UI state selects the command set (e.g. in the *confirm* state only "yes / no" are valid). Candidates are shortlisted by kana edit distance and partial match.
3. Each candidate is scored by teacher forcing, `log p(candidate | audio)`, in one batched decoder pass (about 10 ms for 16 candidates). Candidates embedded in the free hypothesis (filler words around the command) are scored too.
4. Candidates and the free hypothesis are combined into one calibrated distribution; the free hypothesis acts as the "none of the above" option.
5. A policy turns the distribution into execute / confirm / reject, with risk levels per intent.
6. Registering a scope (e.g. a set of lines or cameras) triggers *pre-training*: names are synthesized with TTS in several voices, run through the recognizer, and the calibration, the realized readings and the confusable pairs are stored with the scope.

### Vision demo

```bash
scripts/serve_vision.sh start 0 8602   # OPENVONS_FACE_CKPT / OPENVONS_BODY_CKPT point at trained heads; YuNet onnx under state/models/
```

Both heads share a frozen Qwen3-VL-2B vision encoder (407M) and add tens of thousands of trainable parameters. Face detection uses OpenCV YuNet (Apache-2.0).

### Text and vision

```bash
.venv/bin/python scripts/lm_prepare_datasets.py               # public datasets -> JSONL
.venv/bin/python scripts/lm_train_head.py --task massive_scenario_en --model Qwen/Qwen3-4B-Instruct-2507
DM_LLM_URL=http://127.0.0.1:8300/v1 .venv/bin/python -m openvons.lm.api.server   # POST /v1/decision, and POST /v1/systemone (TypeSafe Jev wire format)
```

`/v1/systemone` accepts the same request shape as the typesafe-sdk (`state` + `questions{type, instructions, criteria}` → `answers`),
so pointing the SDK's base URL at this server makes your local decision model answer. The mapping to the public spec, and the licensing
caveats, are in [docs/jev_api.md](docs/jev_api.md).

### Native app (Android / iOS)

`app/` is a Flutter app that runs the whole pipeline **on the device**: log-mel, recognition, candidate
scoring, calibration and the decision for voice; image embedding and choice matching for vision.
The server only hands out the models and the list of things that can be chosen; no audio or image leaves the phone.

The voice screen is a river-camera console: say a site name to show its live image, then move upstream /
downstream, refresh or zoom. Every utterance is shown in two columns, **the conventional way** (pick the
closest phrase from the transcription, no way to refuse) next to **openvons** (probabilities including
"none of the above", split into execute / confirm / reject), with a running count of how often the
conventional way would have fired the wrong command. Measured on a Nothing Phone 3 (NNAPI, int8):
2.0-2.2 s per utterance, 260-290 ms to embed an image and 11 ms to answer 9 questions about it.

**[Download the Android APK](https://github.com/genai-craft/openvons/releases/latest)** (39 MB for arm64),
or build it yourself:

```bash
cd app && flutter pub get
flutter run                     # a connected Android device, or an iOS device/simulator
flutter build apk --release     # Android (111 MB; --split-per-abi gives ~40 MB per ABI)
flutter build ios --release     # iOS (needs macOS and an Apple signing identity)
```

On first launch it downloads about 300 MB of models (default source `https://ondevice.openvons.com`,
the public demo) and caches them; after that the voice decision needs no network.
To serve the models yourself: `scripts/serve_ondevice.sh start 0 8606`, then change the server in the
settings tab. [app/README.md](app/README.md) has the details, and [docs/ondevice_app.md](docs/ondevice_app.md)
(Japanese) covers the export scripts and the traps (handing the full vocabulary logits to the app crashes
it, so the decoder is re-wrapped to return only the scores and the next token).

### Small model for on-device use (in progress)

whisper-small is being distilled into a 2-layer-decoder katakana model using pseudo labels from kana-whisper (809M) (`openvons/voice/distill/`).
Candidate scoring is tolerant of free-transcription errors, which makes it a good match for small models. The goal is in-browser inference (transformers.js, WebGPU).

## Layout

```
openvons/core/      primitives (Question), formats (Sample), decision (policy), temperature/isotonic/none_calibration (calibration), metrics
openvons/lm/        models (backbone/heads/pooling/decision_model/hybrid_cache), training, backends (LLM baseline), teacher, api, benchmark
openvons/vision/    vision_model (encoder only), vlm_decision_model (small VLM), train_vision/train_vlm, server
openvons/voice/     kana, lexicon (entities and scopes), grammar (state-dependent command sets), asr (kana-whisper), engine, state, vad, synth (pre-training)
openvons/tts/       TTS backends (voicevox:// default, irodori://, openai://)
examples/           text_decision (text questions), stations (railway map), kasen (river live cameras), road_cameras (synthetic cameras)
app/                Flutter app for Android / iOS (everything runs on the device)
openvons/voice/demo_server.py   shared demo server (--app selects the application); openvons/voice/distill/ distillation of the small kana model
scripts/            lm_* (text / vision experiments), build_catalog / build_stations / eval_synthetic / refit_calibration (voice), serve_demo.sh
docs/               design, evaluation and research notes (lm_*, vision_*, voice_*), licensing.md, jev_api.md, roadmap.md
```

Most documents under `docs/` are in Japanese for now.

## Commercial use, customization and deployment

openvons is an independent open-source project. For help applying it to your own domain (your own entity catalog, states and commands,
on-premise deployment, model tuning, evaluation on your own audio), please get in touch through **https://genai-craft.com**.

GitHub Issues are for bugs and questions about the code itself.

## License

Code is Apache-2.0. Third-party models and data are listed in [docs/licensing.md](docs/licensing.md): kana-whisper (MIT), Silero VAD (MIT),
pyopenjtalk (MIT), station_database (CC BY 4.0), Japan Post postal-code data (no copyright claimed), Qwen3 (Apache-2.0),
river camera list and images from the MLIT Kanto Regional Development Bureau website (Public Data License 1.0, edited; images fetched live and not stored), OpenStreetMap tiles (ODbL).

openvons (open-Jev) is an independent implementation unrelated to TypeSafe AI and its product Jev. No output of their API is used anywhere in this project.
