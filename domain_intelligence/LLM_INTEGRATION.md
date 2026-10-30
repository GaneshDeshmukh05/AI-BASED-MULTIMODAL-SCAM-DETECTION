# LLM Threat Analyst Integration Guide

This module connects the **26-feature XGBoost & SHAP URL Classifier** and **live Domain OSINT** (DNS, WHOIS age, TLS certs, brand lookalikes) directly to Large Language Models.

---

## 1. Quick Python Usage (Import into Your Code)

```python
from main import analyze_with_llm

# Analyze any URL
result = analyze_with_llm("https://paypal.com-account-verify.online/login")

print("Verdict:", result["prediction"])
print("Risk Score:", result["risk_score"])
print("Provider:", result["provider_used"])
print("\n--- LLM Forensic Report ---")
print(result["llm_analysis"])
```

---

## 2. Supported LLM Providers

The integration auto-detects your provider from environment variables, or you can specify it directly:

### A. Google Gemini (Recommended)
```bash
set GEMINI_API_KEY="your-google-ai-api-key"
```
```python
from main import analyze_with_llm

report = analyze_with_llm("https://example.com", provider="gemini", model="gemini-1.5-flash")
```

### B. OpenAI / Groq / DeepSeek
```bash
set OPENAI_API_KEY="your-openai-api-key"
# Optional: point to Groq, DeepSeek, or LM Studio
# set OPENAI_BASE_URL="https://api.groq.com/openai/v1"
```
```python
report = analyze_with_llm("https://example.com", provider="openai", model="gpt-4o-mini")
```

### C. Anthropic Claude
```bash
set ANTHROPIC_API_KEY="your-anthropic-key"
```
```python
report = analyze_with_llm("https://example.com", provider="claude", model="claude-3-5-sonnet-20241022")
```

### D. Local Ollama (No API Key Required)
Run Ollama locally (`ollama run llama3`):
```python
report = analyze_with_llm("https://example.com", provider="ollama", model="llama3")
```

### E. Offline Deterministic Analyst (Zero Setup)
If no API key or local model is active, the engine automatically uses its built-in cybersecurity reasoner to generate human-readable verdicts from SHAP and OSINT signals.

---

## 3. Command Line Interface (CLI)

Run directly from PowerShell or Terminal:

```powershell
# Auto-detect provider or use offline reasoner
python main.py https://google.com

# Explicit provider
python main.py https://paypal.com-verify.online --provider gemini

# Output raw structured JSON
python main.py https://example.com --json

# Fast offline-only mode (skips live DNS/WHOIS queries)
python main.py https://example.com --no-intel
```

---

## 4. Agent Tool / Function Calling (LangChain / OpenAI)

You can pass the tool schema directly to an LLM Agent:

```python
from main import get_tool_schema, analyze_with_llm

# Tool schema for OpenAI / LangChain / CrewAI
tool_schema = get_tool_schema()

# Function to execute when LLM invokes the tool
def run_tool(arguments: dict):
    return analyze_with_llm(
        url=arguments["url"],
        include_domain_intelligence=arguments.get("include_domain_intelligence", True)
    )
```

---

## 5. Result Structure

The returned dictionary contains:
* `url`: Scanned URL
* `prediction`: `'phishing'` or `'legitimate'`
* `risk_score`: `0` to `100` composite threat score
* `phishing_probability`: Exact XGBoost model probability
* `risk_level`: `'low'`, `'medium'`, `'high'`, or `'critical'`
* `llm_analysis`: Generated markdown report with:
  * **Executive Verdict**
  * **Threat Vector & Deception Tactics**
  * **Key Technical Evidence** (SHAP attribution, Domain age, SSL status)
  * **Recommended Actions** for SOC analysts or users
* `provider_used`: The active model/provider
* `technical_evidence`: Compact technical JSON payload sent to the LLM
* `raw_evidence`: Full raw ML feature values, SHAP explanation array, and DNS records
