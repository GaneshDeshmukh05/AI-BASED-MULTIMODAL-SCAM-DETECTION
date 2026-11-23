"""compare_audio_engines.py
Checks that the lightweight NumPy audio engine gives the same scores as Keras.

Run this ONCE on your own computer (where TensorFlow is installed) before you
deploy with AUDIO_ENGINE=numpy:

    python scripts/compare_audio_engines.py
    python scripts/compare_audio_engines.py path/to/some_call.wav another.wav

For every WAV file it runs both audio models twice - first with Keras, then with
the NumPy engine - and prints both scores. The script exits with an error if the
scores differ by more than 0.001.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adapters.audio_adapter import AudioAdapter  # noqa: E402

TOLERANCE = 1e-3


def score_with_engine(engine: str, wav_path: str):
    """Returns (scam_score, spoof_score) using the chosen engine, or an error string."""
    os.environ["AUDIO_ENGINE"] = engine
    AudioAdapter._scam_model = None  # force the models to be rebuilt for this engine
    AudioAdapter._spoof_model = None

    scam = AudioAdapter.predict_scam_call_file(wav_path)
    spoof = AudioAdapter.predict_spoof_file(wav_path)

    def read(result):
        if result.status.value != "success":
            return f"FAILED ({result.error_message})"
        return float(result.risk_score)

    return read(scam), read(spoof)


def main() -> int:
    wavs = sys.argv[1:] or [
        str(ROOT / "audio_model" / "dataset" / "SCAM_CALLS" / "scam_22.wav"),
        str(ROOT / "audio_model" / "dataset" / "NORMAL_CALLS" / "normal_17.wav"),
    ]
    wavs = [w for w in wavs if Path(w).exists()]
    if not wavs:
        print("No WAV files found. Pass the path of a .wav file as an argument.")
        return 2

    all_ok = True
    for wav in wavs:
        print(f"\n=== {wav}")
        keras_scores = score_with_engine("keras", wav)
        numpy_scores = score_with_engine("numpy", wav)

        for name, k, n in zip(("scam model", "spoof model"), keras_scores, numpy_scores):
            if isinstance(k, str) or isinstance(n, str):
                print(f"  {name:12s} keras={k}  numpy={n}  -> could not compare")
                all_ok = False
                continue
            diff = abs(k - n)
            verdict = "OK" if diff <= TOLERANCE else "MISMATCH"
            all_ok &= diff <= TOLERANCE
            print(f"  {name:12s} keras={k:.6f}  numpy={n:.6f}  diff={diff:.2e}  {verdict}")

    print("\nRESULT:", "the NumPy engine matches Keras." if all_ok else "PROBLEM - do not deploy with AUDIO_ENGINE=numpy yet.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
