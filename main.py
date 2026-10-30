"""main.py
Main entrypoint integrating:
1. Audio Detection -> audio_model/
2. Domain Intelligence -> domain_intelligence/
3. LLM Multimodal Fusion & Defense Layer (My Integration) -> fusion/ & adapters/

Usage:
    # 1. Run live multimodal analysis on real audio and URL:
    python main.py --audio audio_model/dataset/SCAM_CALLS/scam_22.wav --url "http://paypal.com-verify.online/login"

    # 2. Run with Google Gemini / OpenAI grounded explanation:
    python main.py --audio audio_model/dataset/SCAM_CALLS/scam_22.wav --url "http://paypal.com-verify.online/login" --provider gemini

    # 3. Output structured JSON:
    python main.py --audio audio_model/dataset/SCAM_CALLS/scam_22.wav --url "http://paypal.com-verify.online/login" --json

    # 4. Run automated live test walkthrough across all modules:
    python main.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Silence low-level TF / oneDNN messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Configure UTF-8 for Windows console
if sys.platform == "win32":
    try:
        reconfig_out = getattr(sys.stdout, "reconfigure", None)
        if callable(reconfig_out):
            reconfig_out(encoding="utf-8", errors="replace")
        reconfig_err = getattr(sys.stderr, "reconfigure", None)
        if callable(reconfig_err):
            reconfig_err(encoding="utf-8", errors="replace")
    except Exception:
        pass

from fusion import MultimodalPipeline, ExplainerConfig


def print_banner():
    print("""
========================================================================
   [+] MULTIMODAL SCAM & DEEPFAKE DETECTION FUSION SYSTEM [+]
   Integrated: Audio Models + 26-Feature XGBoost & SHAP URL OSINT
========================================================================
""")


def main():
    parser = argparse.ArgumentParser(
        description="Multimodal Scam & Deepfake Detection Engine (Audio + URL + OSINT)"
    )
    parser.add_argument(
        "--audio",
        "-a",
        default=None,
        help="Path to audio file (.wav) for scam call and anti-spoofing analysis",
    )
    parser.add_argument(
        "--url",
        "-u",
        default=None,
        help="Target URL or domain for XGBoost and Domain OSINT analysis",
    )
    parser.add_argument(
        "--text",
        "-t",
        default=None,
        help="Optional SMS or conversation transcript text",
    )
    parser.add_argument(
        "--screenshot",
        "-s",
        default=None,
        help="Path to screenshot image (.png, .jpg) for visual OCR analysis",
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "gemini", "openai", "deterministic"],
        default="deterministic",
        help="LLM provider for explanation generation (default: deterministic)",
    )
    parser.add_argument(
        "--no-intel",
        action="store_true",
        help="Disable online DNS/WHOIS/TLS lookups (fast offline XGBoost mode)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON results instead of formatted report",
    )

    args = parser.parse_args()

    # If user provided specific inputs, run targeted analysis
    if args.audio is not None or args.url is not None or args.text is not None or args.screenshot is not None:
        pipeline = MultimodalPipeline(
            explainer_config=ExplainerConfig(provider=args.provider)
        )
        result = pipeline.analyze(
            audio_path=args.audio,
            url=args.url,
            text=args.text,
            screenshot_path=args.screenshot,
            include_domain_intelligence=not args.no_intel,
        )

        if args.json:
            print(json.dumps(result.to_dict(), indent=2, default=str))
        else:
            print_banner()
            result.print_summary()
        return

    # If run with no arguments, execute full live multi-scenario integration demonstration
    print_banner()
    print("[*] No custom inputs provided. Running live multimodal integration demonstration...\n")

    default_scam_audio = str(Path("audio_model") / "dataset" / "SCAM_CALLS" / "scam_22.wav")
    default_normal_audio = str(Path("audio_model") / "dataset" / "NORMAL_CALLS" / "normal_17.wav")
    phishing_url = "http://paypal.com-account-verify.online/login"
    legitimate_url = "https://google.com"

    pipeline = MultimodalPipeline(
        explainer_config=ExplainerConfig(provider=args.provider)
    )

    # Scenario 1: Threat Convergence (Live Audio + Live Phishing URL)
    print("------------------------------------------------------------------------")
    print("  SCENARIO 1: Live Multimodal Threat Convergence")
    print(f"  Audio: {default_scam_audio}")
    print(f"  URL  : {phishing_url}")
    print("------------------------------------------------------------------------")
    res1 = pipeline.analyze(
        audio_path=default_scam_audio,
        url=phishing_url,
        include_domain_intelligence=False,
    )
    res1.print_summary()

    # Scenario 2: Contradictory Signals (Live Suspicious Audio + Live Safe URL)
    print("------------------------------------------------------------------------")
    print("  SCENARIO 2: Conflicting Signals & Defense-in-Depth Resolution")
    print(f"  Audio: {default_scam_audio} (Deepfake threat)")
    print(f"  URL  : {legitimate_url} (Legitimate URL)")
    print("------------------------------------------------------------------------")
    res2 = pipeline.analyze(
        audio_path=default_scam_audio,
        url=legitimate_url,
        include_domain_intelligence=True,
    )
    res2.print_summary()

    # Scenario 3: Single Modality Live Audio Evaluation
    print("------------------------------------------------------------------------")
    print("  SCENARIO 3: Live Audio Modality Evaluation")
    print(f"  Audio: {default_normal_audio}")
    print("------------------------------------------------------------------------")
    res3 = pipeline.analyze(
        audio_path=default_normal_audio,
    )
    res3.print_summary()


if __name__ == "__main__":
    main()
