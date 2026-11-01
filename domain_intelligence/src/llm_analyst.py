"""Multi-provider LLM Threat Analyst for Explainable URL & Domain Intelligence.

Integrates with:
- Google Gemini (via GEMINI_API_KEY or GOOGLE_API_KEY)
- OpenAI & Compatible (via OPENAI_API_KEY, custom OPENAI_BASE_URL for Groq/DeepSeek/vLLM)
- Anthropic Claude (via ANTHROPIC_API_KEY)
- Local Ollama (via OLLAMA_HOST, default http://localhost:11434)
- Built-in Deterministic Analyst (High-fidelity offline fallback when no API keys are present)
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
import urllib.error
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from src.config import DOMAIN_INTEL_ENABLED
from src.domain_intelligence import collect_intelligence
from src.evidence import SourceEvidence
from src.model_service import get_model_service
from src.risk_scoring import merge_intelligence
from src.threat_feeds import lookup_url, lookup_urlhaus

log = logging.getLogger(__name__)


def make_json_serializable(obj: Any) -> Any:
    """Recursively convert datetime/date/bytes/sets to standard JSON-serializable types."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(x) for x in obj]
    if isinstance(obj, set):
        return [make_json_serializable(x) for x in sorted(obj, key=str)]
    return obj


SYSTEM_PROMPT = """You are an elite Cybersecurity Threat Intelligence Analyst specializing in phishing forensics and explainable machine learning.
Your task is to analyze the technical evidence gathered for a given URL and provide an authoritative, plain-language security assessment for a SOC (Security Operations Center) analyst or end user.

You will be given:
1. Target URL
2. XGBoost ML Verdict & Risk Score
3. SHAP Feature Attribution (the exact lexical and structural signals that drove the model's decision)
4. Domain Intelligence (WHOIS registration age, DNS resolution, SSL/TLS certificate details, Lookalike brand matches)
5. Threat Feed status (URLhaus & OpenPhish)

Generate a structured security assessment in clean markdown with the following sections:
- ### Executive Verdict: Clear status (Malicious Phishing / High Suspicion / Benign Legitimate) and brief summary.
- ### Threat Vector & Deception Tactics: Explain specifically what the attacker or site is doing (e.g., typosquatting, credential harvesting, brand impersonation, deceptive subdomain chaining).
- ### Key Technical Evidence: Highlight the top 3-4 concrete red flags or safety indicators from the SHAP analysis and domain telemetry (e.g. domain age < 30 days, missing HTTPS, high entropy, suspicious keyword).
- ### Recommended Actions: Actionable next steps (e.g. block domain at perimeter firewall, add to SIEM blocklist, notify affected users, or allow traffic).
"""


class URLThreatAnalyst:
    """Orchestrates ML inference, passive domain intelligence, and LLM reasoning."""

    def __init__(
        self,
        provider: str = "auto",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """Initialize the analyst with a selected provider.

        provider: 'auto', 'gemini', 'openai', 'claude', 'ollama', or 'deterministic'
        """
        self.provider = provider.lower()
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

        if self.provider == "auto":
            self.provider = self._detect_provider()

    def _detect_provider(self) -> str:
        """Auto-detect available provider based on environment variables."""
        if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            return "gemini"
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        if os.getenv("ANTHROPIC_API_KEY"):
            return "claude"
        if os.getenv("OLLAMA_HOST"):
            return "ollama"
        # Check if local Ollama port is responding
        try:
            req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=0.8):
                return "ollama"
        except Exception:
            pass
        return "deterministic"

    def gather_telemetry(
        self,
        url: str,
        include_domain_intelligence: bool = True,
        include_redirects: bool = False,
    ) -> Dict[str, Any]:
        """Run ML inference and gather all technical evidence for the URL."""
        model_result = get_model_service().analyze(url)
        threat_intel = lookup_url(url)

        domain_evidence = None
        use_intel = DOMAIN_INTEL_ENABLED if include_domain_intelligence is None else include_domain_intelligence
        if use_intel:
            try:
                domain_evidence = collect_intelligence(url)
                try:
                    urlhaus_res = lookup_urlhaus(url, timeout=3.0)
                    stale = urlhaus_res.get("is_stale", False)
                    ev = (
                        SourceEvidence.stale_feed("urlhaus", urlhaus_res)
                        if stale
                        else SourceEvidence.ok("urlhaus", urlhaus_res, stale_after_hours=6.0)
                    )
                    domain_evidence.sources.append(ev)
                except Exception:
                    pass
            except Exception as e:
                log.warning(f"Domain intelligence collection failed: {e}")

        merged = merge_intelligence(model_result, domain_evidence)
        merged["threat_intelligence"] = threat_intel

        if threat_intel.get("is_known_phishing"):
            merged["prediction"] = "phishing"
            merged["composite_risk_score"] = 100
            merged["risk_score"] = 100
            merged["risk_level"] = "critical"

        if include_redirects:
            try:
                from src.redirect_resolver import resolve_redirects
                chain = resolve_redirects(url)
                merged["redirect_chain"] = chain.to_dict()
            except Exception:
                merged["redirect_chain"] = None

        return merged

    def _call_gemini(self, prompt: str) -> str:
        """Call Google Gemini REST API."""
        api_key = self.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable is missing.")

        model = self.model or "gemini-1.5-flash"
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": f"{SYSTEM_PROMPT}\n\nEvidence to analyze:\n{prompt}"}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 800,
            },
        }

        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["candidates"][0]["content"]["parts"][0]["text"]

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI or compatible endpoint (Groq, DeepSeek, Together, LM Studio)."""
        api_key = self.api_key or os.getenv("OPENAI_API_KEY") or "dummy-key"
        base_url = (self.base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        endpoint = f"{base_url}/chat/completions"
        model = self.model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Evidence to analyze:\n{prompt}"},
            ],
            "temperature": 0.2,
            "max_tokens": 800,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]

    def _call_claude(self, prompt: str) -> str:
        """Call Anthropic Claude API."""
        api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is missing.")

        endpoint = "https://api.anthropic.com/v1/messages"
        model = self.model or "claude-3-5-sonnet-20241022"

        payload = {
            "model": model,
            "system": SYSTEM_PROMPT,
            "messages": [
                {"role": "user", "content": f"Evidence to analyze:\n{prompt}"}
            ],
            "max_tokens": 800,
            "temperature": 0.2,
        }

        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }

        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["content"][0]["text"]

    def _call_ollama(self, prompt: str) -> str:
        """Call local Ollama server."""
        host = (self.base_url or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        endpoint = f"{host}/api/generate"
        model = self.model or os.getenv("OLLAMA_MODEL", "llama3")

        payload = {
            "model": model,
            "system": SYSTEM_PROMPT,
            "prompt": f"Evidence to analyze:\n{prompt}",
            "stream": False,
        }

        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["response"]

    def _deterministic_analysis(self, evidence: Dict[str, Any]) -> str:
        """Synthesize expert security reasoning without external API calls."""
        url = evidence.get("url", "")
        prediction = evidence.get("prediction", "unknown")
        composite_score = evidence.get("composite_risk_score", evidence.get("risk_score", 0))
        prob = evidence.get("phishing_probability", 0.0) * 100
        risk_level = evidence.get("risk_level", "medium").upper()
        intel = evidence.get("domain_intelligence") or {}
        ssl_info = evidence.get("ssl_analysis") or {}
        top_risk = evidence.get("top_risk_factors", [])
        adjustments = evidence.get("risk_adjustments", [])
        threat_feed = evidence.get("threat_intelligence") or {}

        verdict_str = "MALICIOUS PHISHING THREAT" if prediction == "phishing" else "BENIGN / LEGITIMATE"

        findings = []
        for factor in top_risk:
            feat = factor.get("feature", "")
            shap_val = factor.get("shap_value", 0.0)
            findings.append(f"- **High Risk Feature ({feat}):** Model attribution SHAP +{shap_val:.3f} indicates significant deceptive lexical pattern.")

        age_days = intel.get("age_days")
        if age_days is not None:
            if age_days < 30:
                findings.append(f"- **Critically New Domain Registration:** Domain is only {age_days} days old, a hallmark of disposable phishing infrastructure.")
            else:
                findings.append(f"- **Domain Longevity:** Domain age is {age_days} days ({(age_days / 365):.1f} years), typical of established sites.")

        if ssl_info.get("has_ssl") is False:
            findings.append("- **Insecure Transport:** URL does not utilize TLS/SSL encryption.")
        elif ssl_info.get("valid"):
            issuer = ssl_info.get("issuer", "Unknown CA")
            findings.append(f"- **TLS Certificate:** Valid certificate issued by {issuer}.")

        for adj in adjustments:
            findings.append(f"- **Domain Intelligence Adjustment:** {adj.get('reason', '')} (+{adj.get('points', 0)} pts).")

        if threat_feed.get("is_known_phishing"):
            findings.append("- **Threat Intelligence Feed Hit:** Verified match on active community threat feeds.")

        findings_text = "\n".join(findings) if findings else "- No severe anomalies detected."

        tactics = []
        if intel.get("lookalike", {}).get("matched_brand"):
            brand = intel["lookalike"]["matched_brand"]
            tactics.append(f"Domain closely resembles brand '{brand}', indicating targeted typosquatting or brand spoofing.")
        if any(f.get("feature") == "suspicious_keyword_count" for f in top_risk):
            tactics.append("Embedded authentication keywords (e.g., login, verify, account, secure) designed to elicit credential submission.")
        if any(f.get("feature") in ("subdomain_count", "dot_count") for f in top_risk):
            tactics.append("Deep subdomain nesting structured to obscure the actual apex domain.")
        if not tactics:
            tactics.append("No active deception tactics detected in URL structure.")

        tactics_text = "\n".join(f"- {t}" for t in tactics)

        if prediction == "phishing":
            actions = (
                "- **Block URL:** Ingest domain and host into perimeter firewalls and DNS sinkholes.\n"
                "- **Credential Reset:** Force credential change for any user who accessed this link within the last 24h.\n"
                "- **Report Abuse:** Submit domain details to registrar abuse contact and threat feeds."
            )
        else:
            actions = (
                "- **Allow Traffic:** No anomalous indicators found. Safe to proceed under standard monitoring.\n"
                "- **Standard Hygiene:** Continue enforcing multifactor authentication across all external links."
            )

        return f"""### Executive Verdict: {verdict_str} (Risk Score: {composite_score}/100, Level: {risk_level})
Target URL `{url}` has been evaluated with an ML probability of {prob:.2f}% and classified as **{prediction}**.

### Threat Vector & Deception Tactics:
{tactics_text}

### Key Technical Evidence:
{findings_text}

### Recommended Actions:
{actions}
"""

    def analyze(
        self,
        url: str,
        include_domain_intelligence: bool = True,
        include_redirects: bool = False,
    ) -> Dict[str, Any]:
        """Perform end-to-end analysis: ML + Domain OSINT + LLM Reasoning."""
        # 1. Gather all technical evidence
        evidence = self.gather_telemetry(
            url,
            include_domain_intelligence=include_domain_intelligence,
            include_redirects=include_redirects,
        )
        evidence = make_json_serializable(evidence)

        # 2. Build concise prompt representation
        prompt_data = make_json_serializable({
            "target_url": evidence.get("url"),
            "ml_prediction": evidence.get("prediction"),
            "composite_risk_score": evidence.get("composite_risk_score"),
            "ml_phishing_probability": round(evidence.get("phishing_probability", 0.0), 4),
            "risk_level": evidence.get("risk_level"),
            "top_phishing_shap_factors": evidence.get("top_risk_factors"),
            "top_legitimate_shap_factors": evidence.get("top_legitimate_factors"),
            "domain_intelligence": {
                "registered_domain": evidence.get("domain_intelligence", {}).get("registered_domain") if evidence.get("domain_intelligence") else None,
                "domain_age_days": evidence.get("domain_intelligence", {}).get("age_days") if evidence.get("domain_intelligence") else None,
                "registrar": evidence.get("domain_intelligence", {}).get("registrar") if evidence.get("domain_intelligence") else None,
                "dns_ips": evidence.get("domain_intelligence", {}).get("dns_a_records", []) if evidence.get("domain_intelligence") else [],
                "lookalike_match": evidence.get("domain_intelligence", {}).get("lookalike", {}).get("matched_brand") if evidence.get("domain_intelligence") else None,
            },
            "ssl_certificate": {
                "has_ssl": evidence.get("ssl_analysis", {}).get("has_ssl") if evidence.get("ssl_analysis") else None,
                "valid": evidence.get("ssl_analysis", {}).get("valid") if evidence.get("ssl_analysis") else None,
                "issuer": evidence.get("ssl_analysis", {}).get("issuer") if evidence.get("ssl_analysis") else None,
            },
            "threat_feeds": {
                "is_known_phishing": evidence.get("threat_intelligence", {}).get("is_known_phishing"),
            },
        })

        evidence_json = json.dumps(prompt_data, indent=2, default=str)

        # 3. Call LLM provider
        analysis_text = ""
        provider_used = self.provider

        try:
            if self.provider == "gemini":
                analysis_text = self._call_gemini(evidence_json)
            elif self.provider == "openai":
                analysis_text = self._call_openai(evidence_json)
            elif self.provider == "claude":
                analysis_text = self._call_claude(evidence_json)
            elif self.provider == "ollama":
                analysis_text = self._call_ollama(evidence_json)
            else:
                analysis_text = self._deterministic_analysis(evidence)
                provider_used = "deterministic_analyst"
        except Exception as e:
            log.warning(f"LLM call to {self.provider} failed ({e}); falling back to deterministic reasoning.")
            analysis_text = self._deterministic_analysis(evidence)
            provider_used = f"deterministic_analyst (fallback from {self.provider})"

        # 4. Return consolidated response
        return make_json_serializable({
            "url": url,
            "prediction": evidence.get("prediction"),
            "risk_score": evidence.get("composite_risk_score", evidence.get("risk_score")),
            "phishing_probability": evidence.get("phishing_probability"),
            "risk_level": evidence.get("risk_level"),
            "llm_analysis": analysis_text,
            "provider_used": provider_used,
            "technical_evidence": prompt_data,
            "raw_evidence": evidence,
        })


def analyze_with_llm(
    url: str,
    provider: str = "auto",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    include_domain_intelligence: bool = True,
) -> Dict[str, Any]:
    """Top-level convenience function for analyzing a URL with LLM intelligence.

    Example:
        from src.llm_analyst import analyze_with_llm
        report = analyze_with_llm("https://paypal.com-security-check.online/login")
        print(report["llm_analysis"])
    """
    analyst = URLThreatAnalyst(provider=provider, api_key=api_key, model=model)
    return analyst.analyze(url, include_domain_intelligence=include_domain_intelligence)
