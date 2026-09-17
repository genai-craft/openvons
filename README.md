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

## Getting started

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev]"   # install torch from the CUDA index that matches your GPU
.venv/bin/python -m pytest -q tests/test_voice_core.py
```

### Voice command demos

```bash
docker run -d --name voicevox -p 127.0.0.1:50021:50021 voicevox/voicevox_engine:cpu-latest   # TTS used for pre-training
JEV_APP=examples.stations.app     scripts/serve_demo.sh start 0 8601   # railway map driven by station names (general audience)
JEV_APP=examples.road_cameras.app scripts/serve_demo.sh start 0 8600   # road camera monitoring (industrial example)
```

The microphone requires https or localhost (`ssh -L 8601:localhost:8601 <host>`). On a phone, "Add to Home Screen" opens it as a PWA.
Utterances captured through the microphone are stored under `state/<app>/utts/` (`JEV_DUMP_UTTS=1`, the default) for recalibration and measurement.

| Demo | Live | What it does |
|---|---|---|
| Railway map by voice | https://eki.aunvox.com | 8,987 stations (station_database, CC BY 4.0). Say a station to zoom; next / previous station; switch line; add favorite (asks for confirmation) |
| Road camera monitoring | https://sashizu.aunvox.com | 2,932 synthetic cameras. Say a camera name to enlarge; pan / tilt / zoom; save preset (asks for confirmation) |

An application is just `examples/<name>/app.py` (intents, states, entities) plus `static/`; recognition, calibration, pre-training and the server are shared.
Design and evaluation: [docs/voice_design.md](docs/voice_design.md) / [docs/voice_evaluation.md](docs/voice_evaluation.md) (Japanese).

How the voice pipeline works:

1. kana-whisper transcribes freely into katakana (a reading, not spelling), giving a free hypothesis and its log-likelihood.
2. The current UI state selects the command set (e.g. in the *confirm* state only "yes / no" are valid). Candidates are shortlisted by kana edit distance and partial match.
3. Each candidate is scored by teacher forcing, `log p(candidate | audio)`, in one batched decoder pass (about 10 ms for 16 candidates). Candidates embedded in the free hypothesis (filler words around the command) are scored too.
4. Candidates and the free hypothesis are combined into one calibrated distribution; the free hypothesis acts as the "none of the above" option.
5. A policy turns the distribution into execute / confirm / reject, with risk levels per intent.
6. Registering a scope (e.g. a set of lines or cameras) triggers *pre-training*: names are synthesized with TTS in several voices, run through the recognizer, and the calibration, the realized readings and the confusable pairs are stored with the scope.

### Text and vision

```bash
.venv/bin/python scripts/lm_prepare_datasets.py               # public datasets -> JSONL
.venv/bin/python scripts/lm_train_head.py --task massive_scenario_en --model Qwen/Qwen3-4B-Instruct-2507
DM_LLM_URL=http://127.0.0.1:8300/v1 .venv/bin/python -m openvons.lm.api.server   # POST /v1/decision, and POST /v1/systemone (TypeSafe Jev wire format)
```

`/v1/systemone` accepts the same request shape as the typesafe-sdk (`state` + `questions{type, instructions, criteria}` → `answers`),
so pointing the SDK's base URL at this server makes your local decision model answer. The mapping to the public spec, and the licensing
caveats, are in [docs/jev_api.md](docs/jev_api.md).

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
examples/           stations (railway map), road_cameras (camera monitoring) — each is app.py + static/
openvons/voice/demo_server.py   shared demo server (--app selects the application); openvons/voice/distill/ distillation of the small kana model
scripts/            lm_* (text / vision experiments), build_catalog / build_stations / eval_synthetic / refit_calibration (voice), serve_demo.sh
docs/               design, evaluation and research notes (lm_*, vision_*, voice_*), licensing.md, jev_api.md, roadmap.md
```

Most documents under `docs/` are in Japanese for now.

## License

Code is Apache-2.0. Third-party models and data are listed in [docs/licensing.md](docs/licensing.md): kana-whisper (MIT), Silero VAD (MIT),
pyopenjtalk (MIT), station_database (CC BY 4.0), Japan Post postal-code data (no copyright claimed), Qwen3 (Apache-2.0).

openvons (open-Jev) is an independent implementation unrelated to TypeSafe AI and its product Jev. No output of their API is used anywhere in this project.
