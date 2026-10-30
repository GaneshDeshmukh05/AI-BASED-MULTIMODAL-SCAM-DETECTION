"""Main Entrypoint for LLM-Powered URL & Domain Intelligence.

Usage as a Python module:
    from main import analyze_with_llm, URLThreatAnalyst

    report = analyze_with_llm("https://example.com/login?ref=mail")
    print(report["llm_analysis"])

Usage from Command Line:
    python main.py https://example.com/login
    python main.py https://paypal.com-verify.online --provider gemini
    python main.py https://google.com --provider openai
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure local imports work regardless of cwd
_cur_dir = Path(__file__).resolve().parent
if str(_cur_dir) not in sys.path:
    sys.path.insert(0, str(_cur_dir))

from src.llm_analyst import URLThreatAnalyst, analyze_with_llm

__all__ = ["analyze_with_llm", "URLThreatAnalyst", "get_tool_schema"]


def get_tool_schema() -> Dict[str, Any]:
    """Returns an OpenAI / LangChain compatible tool schema for function calling."""
    return {
        "type": "function",
        "function": {
            "name": "analyze_url_threat",
            "description": (
                "Performs deep machine learning inference (26 lexical/structural features, "
                "SHAP explainability) and live domain OSINT (DNS, WHOIS age, SSL certificate, "
                "lookalike brand detection) to evaluate if a URL is phishing or malicious."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The full HTTP/HTTPS URL or domain to inspect.",
                    },
                    "include_domain_intelligence": {
                        "type": "boolean",
                        "description": "Whether to perform live DNS, WHOIS, and TLS lookups (default: True).",
                        "default": True,
                    },
                },
                "required": ["url"],
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CYBERLENS: LLM-Integrated Phishing URL & Domain Threat Analyst"
    )
    parser.add_argument(
        "url",
        nargs="?",
        default="https://paypal.com-account-verify.online/login",
        help="Target URL to inspect (default: test sample)",
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "gemini", "openai", "claude", "ollama", "deterministic"],
        default="auto",
        help="LLM provider to use (default: auto)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Custom model name (e.g., gemini-1.5-flash, gpt-4o-mini, llama3)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of formatted markdown",
    )
    parser.add_argument(
        "--no-intel",
        action="store_true",
        help="Disable online DNS/WHOIS/TLS domain intelligence",
    )

    args = parser.parse_args()

    print(f"\n[+] Analyzing Target URL: {args.url}")
    print(f"[+] Provider: {args.provider}")
    print("[+] Extracting 26 features, running XGBoost + SHAP & domain telemetry...\n")

    report = analyze_with_llm(
        url=args.url,
        provider=args.provider,
        model=args.model,
        include_domain_intelligence=not args.no_intel,
    )

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print("=" * 70)
        print(f"VERDICT: {report['prediction'].upper()} | RISK SCORE: {report['risk_score']}/100 | LEVEL: {report['risk_level'].upper()}")
        print(f"ANALYST ENGINE: {report['provider_used']}")
        print("=" * 70)
        print("\n" + report["llm_analysis"] + "\n")
        print("=" * 70)


if __name__ == "__main__":
    main()
