"""Focused regression tests for the approved Phase 1 repairs."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path when executed directly or via pytest
WORKSPACE = Path(__file__).resolve().parent.parent
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from typing import Any

import numpy as np

from adapters.audio_adapter import AudioAdapter
from domain_intelligence.src.feature_engineering import extract_features
from domain_intelligence.src.lookalike import scan_domain


def test_spoof_preprocessing_matches_training_configuration(monkeypatch):
    observed: dict[str, object] = {}

    class FeatureModule:
        @staticmethod
        def melspectrogram(**kwargs):
            observed.update(kwargs)
            return np.ones((64, 4), dtype=np.float32)

    class LibrosaModule:
        feature = FeatureModule

        @staticmethod
        def power_to_db(values, ref):
            assert ref is np.max
            return values

    monkeypatch.setitem(__import__("sys").modules, "librosa", LibrosaModule)

    audio = np.ones(80000, dtype=np.float32)
    result = AudioAdapter._compute_spoof_logmel(audio, sr=16000)

    assert observed == {
        "y": audio,
        "sr": 16000,
        "n_fft": 1024,
        "hop_length": 256,
        "n_mels": 64,
        "fmin": 20,
        "fmax": 8000,
        "power": 2.0,
    }
    assert result.shape == (64, 4)
    assert result.dtype == np.float32
    assert np.allclose(result, 0.0)


def test_spoof_inference_uses_training_feature_shape(monkeypatch):
    captured: dict[str, Any] = {}

    class FakeModel:
        def predict(self, value, verbose=0):
            _ = verbose
            captured["input"] = value
            return np.array([[0.25]], dtype=np.float32)

    monkeypatch.setattr(AudioAdapter, "load_spoof_model", classmethod(lambda cls: FakeModel()))
    monkeypatch.setattr(
        AudioAdapter,
        "_load_audio_robust",
        staticmethod(lambda audio_path, target_sr=16000: np.ones(80000, dtype=np.float32)),
    )
    result = AudioAdapter.predict_spoof_file("fixture.wav")

    assert result.status.value == "success"
    assert captured["input"].shape == (1, 64, 313, 1)


def test_exact_brand_apex_domain_is_not_its_own_lookalike():
    for hostname in ("google.com", "www.google.com", "paypal.com"):
        result = scan_domain(hostname)
        assert result.is_lookalike is False
        assert result.matched_brand is None


def test_genuine_lookalikes_remain_detected():
    for hostname in ("paypa1.com", "g00gle-login.example"):
        result = scan_domain(hostname)
        assert result.is_lookalike is True
        assert result.matched_brand in {"paypal", "google"}
        assert result.edit_distance != 0


def test_apex_www_and_benign_subdomain_have_distinct_expected_features():
    apex = extract_features("https://example.com")
    www = extract_features("https://www.example.com")
    subdomain = extract_features("https://mail.example.com")

    assert apex["subdomain_count"] == 0
    assert www["subdomain_count"] == 1
    assert subdomain["subdomain_count"] == 1


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])


def test_compound_brand_lookalike_detection():
    """Verify that deceptive compound brand domains are caught while preserving legitimate domains."""
    deceptive_cases = [
        ("paypal.com-verify.online", "paypal", "compound_brand"),
        ("google.com.security-check.example.invalid", "google", "compound_brand"),
        ("paypal-login.com", "paypal", "compound_brand"),
        ("chase-banking-update.net", "chase", "compound_brand"),
    ]
    for hostname, expected_brand, expected_tech in deceptive_cases:
        res = scan_domain(hostname)
        assert res.is_lookalike is True, f"Failed to flag compound lookalike: {hostname}"
        assert res.matched_brand == expected_brand, f"Wrong brand for {hostname}: {res.matched_brand}"
        assert res.technique == expected_tech, f"Wrong technique for {hostname}: {res.technique}"

    genuine_brand_cases = [
        "paypal.com",
        "www.paypal.com",
        "login.paypal.com",
        "verify.paypal.com",
        "google.com",
        "www.google.com",
        "mail.google.com",
    ]
    for hostname in genuine_brand_cases:
        res = scan_domain(hostname)
        assert res.is_lookalike is False, f"False positive on genuine brand domain: {hostname}"

    benign_mentions = [
        "tech-apple-reviews.com",
        "blog.paypal-community.com",
        "wikipedia.org",
    ]
    for hostname in benign_mentions:
        res = scan_domain(hostname)
        assert res.is_lookalike is False, f"False positive on benign brand mention: {hostname}"


def test_url_normalization_consistency():
    """Verify normalization consistency across URL variants while preserving genuine feature differences."""
    from domain_intelligence.src.feature_engineering import normalise_url

    assert normalise_url("https://www.wikipedia.org/") == "https://www.wikipedia.org"
    assert normalise_url("https://wikipedia.org/") == "https://wikipedia.org"

    f_slash = extract_features("https://www.wikipedia.org/")
    f_noslash = extract_features("https://www.wikipedia.org")
    assert f_slash == f_noslash

    assert normalise_url("wikipedia.org") == "https://wikipedia.org"
    assert normalise_url("www.wikipedia.org") == "https://www.wikipedia.org"
    assert extract_features("wikipedia.org") == extract_features("https://wikipedia.org")

    assert normalise_url("http://wikipedia.org") == "http://wikipedia.org"
    assert extract_features("http://wikipedia.org")["uses_https"] == 0

    f_apex = extract_features("https://wikipedia.org")
    f_www = extract_features("https://www.wikipedia.org")
    assert f_apex["subdomain_count"] == 0
    assert f_www["subdomain_count"] == 1
    assert f_apex["dot_count"] == 1
    assert f_www["dot_count"] == 2
