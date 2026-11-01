"""verify_installation.py
Post-setup health check for the Multimodal Scam Detection System.

Verifies, in order:
1. Core package imports (pydantic, numpy, pandas, sklearn, xgboost, shap, joblib,
   librosa, soundfile, PIL, tldextract, ...).
2. Project package imports (fusion, adapters) resolve from the repository root.
3. Model artifacts exist at their documented locations.
4. Model artifacts actually load (Keras audio models, XGBoost + SHAP, text classifier).
   Pass --fast to skip the slow TensorFlow/Keras loads.
5. A quick offline URL inference + fusion + explanation smoke test.

Run from the repository root:  python scripts/verify_installation.py
Exit code 0 = all checks passed, 1 = at least one check failed.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_results: list[tuple[str, bool, str]] = []


def check(name: str, fn, fast_skippable: bool = False, fast: bool = False) -> None:
    if fast and fast_skippable:
        _results.append((name, True, "SKIPPED (--fast)"))
        print(f"  [ SKIP ] {name}")
        return
    try:
        detail = fn()
        _results.append((name, True, str(detail)))
        print(f"  [ OK   ] {name}" + (f" — {detail}" if detail else ""))
    except Exception as exc:
        _results.append((name, False, str(exc)))
        print(f"  [ FAIL ] {name} — {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the installation health.")
    parser.add_argument("--fast", action="store_true", help="Skip slow Keras/TF model loads")
    args = parser.parse_args()

    print("=" * 70)
    print("  INSTALLATION VERIFICATION — Multimodal Scam Detection System")
    print("=" * 70)

    # --- 1. Third-party imports -------------------------------------------
    print("\n[1/5] Third-party package imports")
    for mod in ["pydantic", "numpy", "pandas", "sklearn", "xgboost", "shap",
                "joblib", "librosa", "soundfile", "PIL", "tldextract",
                "requests", "dns", "whois"]:
        check(f"import {mod}", lambda m=mod: __import__(m).__name__)
    check("import keras / tensorflow (slow first time)",
          lambda: (lambda k: f"keras {k.__version__}")(__import__("keras")),
          fast_skippable=True, fast=args.fast)

    # --- 2. Project package imports ---------------------------------------
    print("\n[2/5] Project package imports (run from repo root)")
    check("import fusion (full public API)",
          lambda: f"{len(__import__('fusion').__all__)} public symbols")
    check("import adapters", lambda: f"{len(__import__('adapters').__all__)} adapters")
    check("import main / demo modules compile",
          lambda: "ok" if (__import__("py_compile").compile(str(ROOT / "main.py"), doraise=True)
                           and __import__("py_compile").compile(str(ROOT / "demo.py"), doraise=True))
          else "")

    # --- 3. Artifact existence --------------------------------------------
    print("\n[3/5] Model artifact locations")
    artifacts = {
        "Scam-call audio model": ROOT / "audio_model/dataset/best_audio_scam_model.keras",
        "Anti-spoof audio model": ROOT / "audio_model/LA/best_model.keras",
        "XGBoost URL model": ROOT / "domain_intelligence/models/xgboost_url_model.joblib",
        "SHAP URL explainer": ROOT / "domain_intelligence/models/shap_url_explainer.joblib",
        "URL feature manifest": ROOT / "domain_intelligence/models/feature_manifest.json",
        "Text scam classifier": ROOT / "text_and_screenshot_module/models/text_scam_classifier.joblib",
        "Sample scam audio": ROOT / "audio_model/dataset/SCAM_CALLS/scam_22.wav",
    }
    for name, path in artifacts.items():
        check(f"artifact: {name}",
              lambda p=path: f"{p.stat().st_size / 1e6:.1f} MB" if p.is_file() else (_ for _ in ()).throw(FileNotFoundError(p)))

    # --- 4. Artifact loading ----------------------------------------------
    print("\n[4/5] Model artifact loading")
    check("Keras scam-call model loads",
          lambda: "loaded" if __import__("adapters").AudioAdapter.load_scam_model() is not None else "missing",
          fast_skippable=True, fast=args.fast)
    check("Keras anti-spoof model loads",
          lambda: "loaded" if __import__("adapters").AudioAdapter.load_spoof_model() is not None else "missing",
          fast_skippable=True, fast=args.fast)
    check("XGBoost + SHAP URL model service loads",
          lambda: "loaded" if __import__("adapters").URLAdapter.load_model_service() is not None else "missing")
    check("Text classifier loads",
          lambda: "loaded" if __import__("adapters").TextAdapter.load_model() is not None
          else "NOT FOUND (text adapter will fall back to heuristics)")

    # --- 5. End-to-end smoke test (offline, no network) ---------------------
    print("\n[5/5] Fusion pipeline smoke test (offline URL mode)")
    def smoke():
        from fusion import analyze_multimodal
        result = analyze_multimodal(
            url="http://paypal.com-verify.online/login",
            include_domain_intelligence=False,
        )
        payload = result.to_dict()
        import json
        json.dumps(payload, default=str)  # raises if not JSON-serialisable
        return f"verdict={result.verdict.value}, risk={result.overall_risk_score:.2f}, JSON ok"
    check("analyze_multimodal + to_dict() JSON", smoke)

    # --- Summary -------------------------------------------------------------
    failed = [r for r in _results if not r[1]]
    skipped = [r for r in _results if r[1] and r[2].startswith("SKIPPED")]
    print("\n" + "=" * 70)
    print(f"  RESULT: {len(_results) - len(failed) - len(skipped)} passed, "
          f"{len(failed)} failed, {len(skipped)} skipped")
    if failed:
        print("\n  Failed checks:")
        for name, _, err in failed:
            print(f"    - {name}: {err}")
    print("=" * 70)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
