"""互換用の別名パッケージ。openvons は open-Jev として始まったので `import jev` / `from jev.voice import ...` も動くようにしておく。"""
import importlib as _il, sys as _sys
import openvons as _ov
_sys.modules[__name__] = _ov
for _sub in ("core", "voice", "lm", "vision", "tts"):
    try:
        _sys.modules[f"{__name__}.{_sub}"] = _il.import_module(f"openvons.{_sub}")
    except Exception:
        pass
