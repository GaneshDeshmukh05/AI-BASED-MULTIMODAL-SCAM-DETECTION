"""Tests for audio engine selection (AUDIO_ENGINE) and the capped audio decoding.

These tests need no model files and no TensorFlow: the NumPy predictor is replaced
by a tiny fake, and `keras` is replaced or hidden with monkeypatch.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is in sys.path when executed directly or via pytest
WORKSPACE = Path(__file__).resolve().parent.parent
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from adapters import audio_adapter as audio_module
from adapters.audio_adapter import AudioAdapter


class _FakePredictor:
    built: list = []

    def __init__(self, path, has_conv_bias=False):
        type(self).built.append((path, has_conv_bias))


@pytest.fixture
def fake_numpy_engine(monkeypatch):
    pytest.importorskip("h5py")  # adapters.numpy_cnn imports h5py at import time
    from adapters import numpy_cnn

    _FakePredictor.built = []
    monkeypatch.setattr(numpy_cnn, "NumPyKerasCNNPredictor", _FakePredictor)
    return _FakePredictor


def test_numpy_engine_never_touches_keras(monkeypatch, fake_numpy_engine):
    class ExplodingKeras(types.ModuleType):
        def __getattr__(self, name):
            raise AssertionError("keras must not be used when AUDIO_ENGINE=numpy")

    monkeypatch.setenv("AUDIO_ENGINE", "numpy")
    monkeypatch.setitem(sys.modules, "keras", ExplodingKeras("keras"))

    AudioAdapter._build_model("model.keras", has_conv_bias=True)

    assert fake_numpy_engine.built == [("model.keras", True)]


def test_auto_engine_falls_back_to_numpy_without_keras(monkeypatch, fake_numpy_engine):
    monkeypatch.setenv("AUDIO_ENGINE", "auto")
    monkeypatch.setitem(sys.modules, "keras", None)  # makes `import keras` fail

    AudioAdapter._build_model("model.keras", has_conv_bias=False)

    assert fake_numpy_engine.built == [("model.keras", False)]


def test_keras_engine_is_strict(monkeypatch):
    monkeypatch.setenv("AUDIO_ENGINE", "keras")
    monkeypatch.setitem(sys.modules, "keras", None)

    with pytest.raises(ImportError):
        AudioAdapter._build_model("model.keras", has_conv_bias=False)


def test_invalid_engine_value_behaves_like_auto(monkeypatch):
    monkeypatch.setenv("AUDIO_ENGINE", "banana")
    assert AudioAdapter._engine_preference() == "auto"


def test_models_are_cached_and_use_correct_bias_flags(monkeypatch, fake_numpy_engine):
    monkeypatch.setenv("AUDIO_ENGINE", "numpy")
    monkeypatch.setattr(AudioAdapter, "_scam_model", None)
    monkeypatch.setattr(AudioAdapter, "_spoof_model", None)

    first = AudioAdapter.load_scam_model("scam.keras")
    second = AudioAdapter.load_scam_model()
    AudioAdapter.load_spoof_model("spoof.keras")

    assert first is second
    assert fake_numpy_engine.built == [("scam.keras", False), ("spoof.keras", True)]


def test_audio_decoding_is_capped(monkeypatch, tmp_path):
    sf = pytest.importorskip("soundfile")
    pytest.importorskip("librosa")
    monkeypatch.setattr(audio_module, "MAX_AUDIO_SECONDS", 10.0)

    wav_path = tmp_path / "long_call.wav"
    sf.write(str(wav_path), np.zeros(8000 * 30, dtype=np.float32), 8000)  # 30 s at 8 kHz

    samples = AudioAdapter._load_audio_robust(str(wav_path), target_sr=16000)

    # About 10 s at 16 kHz, not the full 30 s (which would be 480,000 samples).
    assert 159_000 <= len(samples) <= 161_000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
