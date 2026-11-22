"""app.py
Interactive Streamlit Web Frontend for the Multimodal Scam Detection System.
Enhanced with:
- @st.cache_resource pipeline caching
- Plotly interactive Risk Speedometer / Gauge
- SHAP Feature Attribution bar charts for URLs
- Real-time Session Scan History log
- Multi-modality forensic details and raw API JSON export
"""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fusion import ExplainerConfig, MultimodalPipeline

st.set_page_config(
    page_title="Multimodal Scam & Deepfake Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .verdict-box {
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
        color: white;
    }
    .verdict-scam {
        background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%);
        box-shadow: 0 4px 20px rgba(239, 68, 68, 0.35);
    }
    .verdict-suspicious {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        box-shadow: 0 4px 20px rgba(245, 158, 11, 0.35);
    }
    .verdict-safe {
        background: linear-gradient(135deg, #10b981 0%, #047857 100%);
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.35);
    }
    .verdict-inconclusive {
        background: linear-gradient(135deg, #64748b 0%, #475569 100%);
        box-shadow: 0 4px 20px rgba(100, 116, 139, 0.35);
    }
    
    .metric-card {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_pipeline(provider: str) -> MultimodalPipeline:
    """Caches pipeline instance to prevent repeated model re-initialization."""
    return MultimodalPipeline(explainer_config=ExplainerConfig(provider=provider))


# Initialize session state for scan history
if "scan_history" not in st.session_state:
    st.session_state.scan_history = []

# Title & Description
st.title("🛡️ Multimodal Scam & Phishing Detection System")
st.markdown(
    "Analyze potential threats across **Phone Calls (Audio)**, **Web Links (URL)**, "
    "**SMS Messages (Text)**, and **Screenshots (OCR)** with AI-powered multi-model fusion."
)

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    explainer_provider = st.selectbox(
        "Explanation Provider",
        options=["gemini", "deterministic", "openai"],
        index=0,
        help="Gemini uses your integrated API key for natural AI explanations. Deterministic is offline fallback.",
    )
    include_osint = st.checkbox(
        "Live Domain Intelligence (DNS/WHOIS/TLS)",
        value=False,
        help="Check to perform real-time network lookups. Leave unchecked for fast offline mode.",
    )

    st.markdown("---")
    st.markdown("### 🧪 Quick-Load Sample Scenarios")
    scenario = st.selectbox(
        "Load Preset Demo Case",
        options=[
            "Custom Input",
            "Threat Convergence (Scam Call + Phishing URL)",
            "Urgent SMS Phishing + Malicious Link",
            "Normal Phone Call",
            "Phishing Screenshot",
            "Safe Communication",
        ],
    )

    st.markdown("---")
    st.markdown("### 🕒 Scan History")
    if st.session_state.scan_history:
        for idx, item in enumerate(st.session_state.scan_history[:5]):
            st.markdown(
                f"**#{idx+1} [{item['time']}]** `{item['verdict']}` ({item['risk']})\n\n"
                f"*{item['headline']}*"
            )
            st.divider()
        if st.button("Clear History", use_container_width=True):
            st.session_state.scan_history = []
            st.rerun()
    else:
        st.caption("No scans performed yet in this session.")

# Preset values based on selection
preset_audio = None
preset_url = ""
preset_text = ""
preset_screenshot = None

sample_scam_wav = ROOT / "audio_model" / "dataset" / "SCAM_CALLS" / "scam_22.wav"
sample_normal_wav = ROOT / "audio_model" / "dataset" / "NORMAL_CALLS" / "normal_17.wav"
sample_scam_png = ROOT / "test_scam.png"
if not sample_scam_png.exists():
    sample_scam_png = ROOT / "text_and_screenshot_module" / "test_scam.png"
sample_normal_png = ROOT / "test_normal.png"
if not sample_normal_png.exists():
    sample_normal_png = ROOT / "text_and_screenshot_module" / "test_normal.png"

if scenario == "Threat Convergence (Scam Call + Phishing URL)":
    if sample_scam_wav.exists():
        preset_audio = str(sample_scam_wav)
    preset_url = "http://paypal.com-account-verify.online/login"
elif scenario == "Urgent SMS Phishing + Malicious Link":
    preset_text = "URGENT: Your Chase bank account will be suspended in 24 hours due to unauthorized activity. Please verify your credentials immediately at http://paypal.com-account-verify.online/login"
    preset_url = "http://paypal.com-account-verify.online/login"
elif scenario == "Normal Phone Call":
    if sample_normal_wav.exists():
        preset_audio = str(sample_normal_wav)
elif scenario == "Phishing Screenshot":
    if sample_scam_png.exists():
        preset_screenshot = str(sample_scam_png)
    preset_url = "http://paypal.com-account-verify.online/login"
elif scenario == "Safe Communication":
    preset_text = "Hi Mom, I will be home by 6 PM for dinner. Let me know if you need any groceries on my way back."
    preset_url = "https://www.google.com"
    if sample_normal_png.exists():
        preset_screenshot = str(sample_normal_png)

# Input Modalities Section
st.markdown("### 📥 Input Threat Channels")
col1, col2 = st.columns(2)

with col1:
    st.subheader("📞 1. Audio Call (.wav)")
    audio_file = st.file_uploader("Upload Audio Call Recording", type=["wav", "mp3", "ogg"])
    selected_sample_audio = None

    if not audio_file:
        audio_options = ["None"]
        if sample_scam_wav.exists():
            audio_options.append("Sample Scam Call (scam_22.wav)")
        if sample_normal_wav.exists():
            audio_options.append("Sample Normal Call (normal_17.wav)")

        default_idx = 0
        if preset_audio == str(sample_scam_wav):
            default_idx = 1
        elif preset_audio == str(sample_normal_wav):
            default_idx = 2

        chosen_audio = st.selectbox(
            "Or choose a sample audio file:",
            audio_options,
            index=default_idx,
            key="audio_sample_select",
        )
        if chosen_audio == "Sample Scam Call (scam_22.wav)":
            selected_sample_audio = str(sample_scam_wav)
        elif chosen_audio == "Sample Normal Call (normal_17.wav)":
            selected_sample_audio = str(sample_normal_wav)

    if audio_file:
        st.audio(audio_file)
    elif selected_sample_audio:
        st.audio(selected_sample_audio)

    st.subheader("💬 2. SMS / Text Message")
    input_text = st.text_area(
        "Enter SMS, Email, or Chat Transcript:",
        value=preset_text,
        placeholder="e.g. URGENT: Your bank account will be closed in 24 hours. Click link to verify...",
        height=100,
    )

with col2:
    st.subheader("🔗 3. Target URL / Domain")
    input_url = st.text_input(
        "Enter suspicious link or domain:",
        value=preset_url,
        placeholder="e.g. http://paypal.com-account-verify.online/login",
    )

    st.subheader("🖼️ 4. Screenshot Image")
    screenshot_file = st.file_uploader(
        "Upload Screenshot (PNG, JPG):",
        type=["png", "jpg", "jpeg"],
        key="screenshot_uploader",
    )
    selected_sample_shot = None

    if not screenshot_file:
        shot_options = ["None"]
        if sample_scam_png.exists():
            shot_options.append("Sample Scam Screenshot (test_scam.png)")
        if sample_normal_png.exists():
            shot_options.append("Sample Normal Screenshot (test_normal.png)")

        shot_idx = 0
        if preset_screenshot == str(sample_scam_png):
            shot_idx = 1
        elif preset_screenshot == str(sample_normal_png):
            shot_idx = 2

        chosen_shot = st.selectbox(
            "Or choose a sample screenshot:",
            shot_options,
            index=shot_idx,
            key="screenshot_sample_select",
        )
        if chosen_shot == "Sample Scam Screenshot (test_scam.png)":
            selected_sample_shot = str(sample_scam_png)
        elif chosen_shot == "Sample Normal Screenshot (test_normal.png)":
            selected_sample_shot = str(sample_normal_png)

    if screenshot_file:
        st.image(screenshot_file, caption="Uploaded Screenshot", use_container_width=True)
    elif selected_sample_shot:
        st.image(selected_sample_shot, caption=f"Selected: {chosen_shot}", use_container_width=True)

st.markdown("---")

# Execution Button
run_analysis = st.button("🚀 Run Multimodal Scam Analysis", type="primary", use_container_width=True)

if run_analysis:
    # Resolve audio path
    effective_audio_path = None
    temp_audio_file = None
    if audio_file is not None:
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_audio.write(audio_file.read())
        temp_audio.flush()
        temp_audio.close()
        effective_audio_path = temp_audio.name
        temp_audio_file = effective_audio_path
    elif selected_sample_audio:
        effective_audio_path = selected_sample_audio

    # Resolve screenshot path
    effective_shot_path = None
    temp_shot_file = None
    if screenshot_file is not None:
        temp_shot = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        temp_shot.write(screenshot_file.read())
        temp_shot.flush()
        temp_shot.close()
        effective_shot_path = temp_shot.name
        temp_shot_file = effective_shot_path
    elif selected_sample_shot:
        effective_shot_path = selected_sample_shot

    effective_url = input_url.strip() if input_url and input_url.strip() else None
    effective_text = input_text.strip() if input_text and input_text.strip() else None

    if not any([effective_audio_path, effective_url, effective_text, effective_shot_path]):
        st.warning("⚠️ Please provide at least one input modality (Audio, URL, Text, or Screenshot) to run analysis.")
    else:
        with st.spinner("🔍 Performing forensic analysis across modalities and executing decision fusion..."):
            try:
                pipeline = get_pipeline(explainer_provider)
                result = pipeline.analyze(
                    audio_path=effective_audio_path,
                    url=effective_url,
                    text=effective_text,
                    screenshot_path=effective_shot_path,
                    include_domain_intelligence=include_osint,
                )

                st.markdown("## 📊 Analysis Results")

                verdict_val = result.verdict.value.upper()
                risk_pct = int(result.overall_risk_score * 100)

                # Add to scan history
                st.session_state.scan_history.insert(
                    0,
                    {
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "verdict": verdict_val,
                        "risk": f"{risk_pct}%",
                        "headline": result.headline,
                    },
                )

                if verdict_val == "SCAM":
                    verdict_class = "verdict-scam"
                    badge_icon = "🚨"
                elif verdict_val == "SUSPICIOUS":
                    verdict_class = "verdict-suspicious"
                    badge_icon = "⚠️"
                elif verdict_val == "SAFE":
                    verdict_class = "verdict-safe"
                    badge_icon = "✅"
                else:
                    verdict_class = "verdict-inconclusive"
                    badge_icon = "ℹ️"

                # Verdict Banner
                st.markdown(
                    f"""
                    <div class="verdict-box {verdict_class}">
                        <h1 style="margin:0; font-size: 2.5rem;">{badge_icon} VERDICT: {verdict_val}</h1>
                        <p style="margin: 0.5rem 0 0 0; font-size: 1.25rem; font-weight: 600;">
                            Aggregated Risk: {risk_pct}% &nbsp;|&nbsp; Threat Level: {result.risk_level.value}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Dashboard Layout: Gauge on Left, Metrics on Right
                res_col1, res_col2 = st.columns([1, 1])

                with res_col1:
                    # Plotly Speedometer Gauge
                    gauge_color = "#ef4444" if risk_pct >= 65 else ("#f59e0b" if risk_pct >= 35 else "#10b981")
                    fig_gauge = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=risk_pct,
                            domain={"x": [0, 1], "y": [0, 1]},
                            title={"text": "Overall Threat Risk Index", "font": {"size": 18, "color": "white"}},
                            number={"suffix": "%", "font": {"size": 40, "color": "white"}},
                            gauge={
                                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "white"},
                                "bar": {"color": gauge_color, "thickness": 0.4},
                                "bgcolor": "rgba(255,255,255,0.05)",
                                "borderwidth": 1,
                                "bordercolor": "rgba(255,255,255,0.2)",
                                "steps": [
                                    {"range": [0, 35], "color": "rgba(16, 185, 129, 0.2)"},
                                    {"range": [35, 65], "color": "rgba(245, 158, 11, 0.2)"},
                                    {"range": [65, 100], "color": "rgba(239, 68, 68, 0.3)"},
                                ],
                                "threshold": {
                                    "line": {"color": "white", "width": 3},
                                    "thickness": 0.8,
                                    "value": risk_pct,
                                },
                            },
                        )
                    )
                    fig_gauge.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        height=260,
                        margin=dict(l=20, r=20, t=30, b=20),
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)

                with res_col2:
                    st.markdown("#### Key Threat Indicators")
                    m1, m2 = st.columns(2)
                    with m1:
                        st.metric("Model Confidence", f"{result.overall_confidence:.2f}")
                        st.metric("Active Channels", len(result.active_modalities))
                    with m2:
                        st.metric(
                            "Channel Conflict",
                            "YES (Contradiction)" if result.has_conflict else "No Conflict",
                        )
                        st.metric(
                            "Defense Escalation",
                            "Active" if result.overall_risk_score >= 0.7 else "Standard",
                        )

                # SHAP Feature Attribution Chart (if URL was analyzed)
                url_res = result.modality_results.get("url")
                if url_res and getattr(url_res, "raw_output", None):
                    top_risks = url_res.raw_output.get("top_risk_factors", [])
                    if top_risks and any(isinstance(r, dict) for r in top_risks):
                        feats = [r["feature"] for r in top_risks if isinstance(r, dict)]
                        vals = [float(r.get("shap_value", 0.0)) for r in top_risks if isinstance(r, dict)]
                        if feats and vals:
                            st.markdown("---")
                            st.subheader("📈 URL Phishing Risk Attribution (SHAP Analysis)")
                            fig_shap = go.Figure(
                                go.Bar(
                                    x=vals[::-1],
                                    y=feats[::-1],
                                    orientation="h",
                                    marker=dict(
                                        color=["#ef4444" if v > 0 else "#10b981" for v in vals[::-1]],
                                    ),
                                )
                            )
                            fig_shap.update_layout(
                                title="Feature Contribution Toward Phishing Risk (+ pushes toward Scam)",
                                xaxis_title="SHAP Value Contribution",
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                font=dict(color="white"),
                                height=240,
                                margin=dict(l=20, r=20, t=30, b=20),
                            )
                            st.plotly_chart(fig_shap, use_container_width=True)

                st.markdown("---")

                # Explanation Section
                st.subheader("💡 Plain-Language Threat Assessment")
                st.info(f"**{result.headline}**\n\n{result.why}")

                # Grounded Evidence Citations
                st.subheader("🔎 Forensic Evidence Citations")
                if result.evidence_citations:
                    for ev in result.evidence_citations:
                        st.markdown(f"- {ev}")
                else:
                    st.write("No specific threats detected.")

                if result.explanation.conflict_warning:
                    st.warning(f"**Cross-Channel Conflict Warning:**\n{result.explanation.conflict_warning}")

                if result.explanation.missing_data_disclaimer:
                    st.caption(f"ℹ️ {result.explanation.missing_data_disclaimer}")

                # Recommended Actions
                st.subheader("🛡️ Recommended Security Actions")
                for act in result.recommended_actions:
                    st.markdown(f"- ✅ **{act}**")

                # Detailed Modality Breakdown
                st.markdown("---")
                st.subheader("🔬 Modality Details Breakdown")
                mod_tabs = st.tabs(["Active Modalities Summary", "Raw API JSON Contract"])

                with mod_tabs[0]:
                    for mod_name, mod_res in result.modality_results.items():
                        if mod_res is not None:
                            with st.expander(
                                f"Modality: {mod_name.upper()} (Risk: {mod_res.risk_score:.2f})",
                                expanded=True,
                            ):
                                st.write(f"- **Status**: {mod_res.status.value}")
                                st.write(f"- **Risk Score**: {mod_res.risk_score:.4f}")
                                st.write(f"- **Confidence**: {mod_res.confidence:.4f}")
                                st.write(f"- **Is Threat**: {mod_res.is_threat}")
                                if getattr(mod_res, "evidence", None):
                                    st.write("**Evidence Statements:**")
                                    for ev_item in mod_res.evidence:
                                        st.write(f"  • {ev_item}")
                                if getattr(mod_res, "raw_output", None):
                                    st.caption("Raw Detector Telemetry:")
                                    st.json(mod_res.raw_output)

                with mod_tabs[1]:
                    st.json(result.to_dict())

            except Exception as e:
                st.error(f"Error during analysis: {e}")
            finally:
                # Clean up temporary files if created
                if temp_audio_file and os.path.exists(temp_audio_file):
                    try:
                        os.remove(temp_audio_file)
                    except Exception:
                        pass
                if temp_shot_file and os.path.exists(temp_shot_file):
                    try:
                        os.remove(temp_shot_file)
                    except Exception:
                        pass
