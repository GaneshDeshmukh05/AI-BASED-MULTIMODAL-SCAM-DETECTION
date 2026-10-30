"""test_text_screenshot_regression.py
Regression and validation test suite for Text and Screenshot modules
integrated with the Multimodal Fusion Pipeline.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

# Ensure project root is in sys.path when executed directly or via pytest
WORKSPACE = Path(__file__).resolve().parent.parent
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

import pytest
from PIL import Image

from adapters.screenshot_adapter import ScreenshotAdapter
from adapters.text_adapter import TextAdapter
from fusion import ExplainerConfig, MultimodalPipeline
from fusion.schemas import DetectorStatus, OverallVerdict

SCREENSHOT_ZIP = (
    WORKSPACE
    / "text_and_screenshot_module"
    / "Screenshot_Dataset-20261023T174631Z-1-001"
    / "Screenshot_Dataset"
    / "FINAL_DATASET_1_3.zip"
)


# ==============================================================================
# SCREENSHOT DETECTION TESTS (Genuine OCR Content Analysis)
# ==============================================================================


@pytest.fixture(scope="module")
def screenshot_assets():
    """Extract representative test screenshots into a temporary directory."""
    temp_dir = tempfile.mkdtemp(prefix="test_screenshot_assets_")
    phish_path = None
    legit_path = None

    if SCREENSHOT_ZIP.is_file():
        with zipfile.ZipFile(SCREENSHOT_ZIP, "r") as z:
            # Chase phishing login portal
            if "FINAL_DATASET_1_3/phishing/phishing_1000.png" in z.namelist():
                phish_path = z.extract("FINAL_DATASET_1_3/phishing/phishing_1000.png", temp_dir)
            # Legitimate article
            if "FINAL_DATASET_1_3/legitimate/legitimate_0.png" in z.namelist():
                legit_path = z.extract("FINAL_DATASET_1_3/legitimate/legitimate_0.png", temp_dir)

    # Renamed phishing screenshot (neutral name, must produce identical analysis)
    renamed_phish = os.path.join(temp_dir, "screen_capture_9921.png")
    if phish_path and os.path.exists(phish_path):
        shutil.copyfile(phish_path, renamed_phish)

    # Blank image with no extractable text
    blank_path = os.path.join(temp_dir, "blank_canvas.png")
    Image.new("RGB", (400, 300), color=(255, 255, 255)).save(blank_path)

    # Corrupted image file
    corrupt_path = os.path.join(temp_dir, "corrupted.png")
    with open(corrupt_path, "wb") as f:
        f.write(b"CORRUPTED_NON_IMAGE_DATA_BYTES_123456789")

    # Missing file path
    missing_path = os.path.join(temp_dir, "non_existent_file.png")

    yield {
        "phish_original": phish_path,
        "phish_renamed": renamed_phish,
        "legitimate": legit_path,
        "blank": blank_path,
        "corrupted": corrupt_path,
        "missing": missing_path,
    }

    shutil.rmtree(temp_dir, ignore_errors=True)


def test_screenshot_phishing_detection(screenshot_assets):
    """Verifies that an actual phishing screenshot is detected from its OCR text."""
    path = screenshot_assets["phish_original"]
    if not path or not os.path.exists(path):
        pytest.skip("Phishing sample screenshot not found in dataset")

    res = ScreenshotAdapter.predict_screenshot(path)
    assert res.status == DetectorStatus.SUCCESS
    assert res.is_threat is True
    assert res.risk_score is not None
    score = res.risk_score if res.risk_score is not None else 0.0
    assert score >= 0.70
    assert res.label == "fake_login"
    assert "Chase Bank" in res.raw_output.get("detected_brands", [])


def test_screenshot_rename_invariance(screenshot_assets):
    """Essential regression test: renaming the file must not alter predictions."""
    orig_path = screenshot_assets["phish_original"]
    renamed_path = screenshot_assets["phish_renamed"]
    if not orig_path or not os.path.exists(orig_path):
        pytest.skip("Phishing sample screenshot not found in dataset")

    res_orig = ScreenshotAdapter.predict_screenshot(orig_path)
    res_renamed = ScreenshotAdapter.predict_screenshot(renamed_path)

    assert res_orig.risk_score == res_renamed.risk_score
    assert res_orig.is_threat == res_renamed.is_threat
    assert res_orig.label == res_renamed.label
    assert res_orig.raw_output.get("detected_brands") == res_renamed.raw_output.get("detected_brands")


def test_screenshot_legitimate_content(screenshot_assets):
    """Verifies that legitimate non-phishing content is not falsely flagged."""
    path = screenshot_assets["legitimate"]
    if not path or not os.path.exists(path):
        pytest.skip("Legitimate sample screenshot not found in dataset")

    res = ScreenshotAdapter.predict_screenshot(path)
    assert res.status == DetectorStatus.SUCCESS
    assert res.is_threat is False
    assert res.risk_score is not None
    score = res.risk_score if res.risk_score is not None else 1.0
    assert score < 0.35
    assert res.label == "legitimate_portal"


def test_screenshot_blank_image_inconclusive(screenshot_assets):
    """Verifies that an image with no readable text returns an inconclusive result."""
    path = screenshot_assets["blank"]
    res = ScreenshotAdapter.predict_screenshot(path)

    assert res.status == DetectorStatus.SUCCESS
    assert res.label == "inconclusive"
    assert res.risk_score is None  # Excluded from math, not assumed safe
    assert res.is_threat is False
    assert "inconclusive" in res.evidence[0].lower()


def test_screenshot_corrupted_file_handled_safely(screenshot_assets):
    """Verifies that a corrupted image file returns a safe FAILED status."""
    path = screenshot_assets["corrupted"]
    res = ScreenshotAdapter.predict_screenshot(path)

    assert res.status == DetectorStatus.FAILED
    assert res.risk_score is None
    assert res.is_threat is False
    assert res.error_message is not None


def test_screenshot_missing_file_handled_safely(screenshot_assets):
    """Verifies that a missing file returns a safe FAILED status."""
    path = screenshot_assets["missing"]
    res = ScreenshotAdapter.predict_screenshot(path)

    assert res.status == DetectorStatus.FAILED
    assert res.risk_score is None
    assert res.is_threat is False


# ==============================================================================
# TEXT DETECTION TESTS (Calibrated Scoring & Semantic Detection)
# ==============================================================================


def test_text_bank_suspension_scam():
    """Verifies fake bank suspension lure is detected as a threat."""
    text = (
        "URGENT: Your Wells Fargo account has been suspended due to unauthorized access. "
        "Click here immediately to verify your identity: http://wf-sec.com/login"
    )
    res = TextAdapter.predict_text(text)
    assert res.status == DetectorStatus.SUCCESS
    assert res.is_threat is True
    assert res.risk_score is not None
    score = res.risk_score if res.risk_score is not None else 0.0
    assert score >= 0.70
    assert res.label == "scam"
    assert "account_takeover_threat" in res.raw_output.get("detected_tactics", [])


def test_text_irs_tax_warrant_scam():
    """Verifies IRS back taxes and arrest warrant scam is detected as a threat."""
    text = (
        "IRS Notice: You owe $4,850 in back taxes. A federal warrant is issued. "
        "Pay now via prepaid card to avoid arrest: 1-800-555-0199"
    )
    res = TextAdapter.predict_text(text)
    assert res.status == DetectorStatus.SUCCESS
    assert res.is_threat is True
    assert res.risk_score is not None
    score = res.risk_score if res.risk_score is not None else 0.0
    assert score >= 0.70
    assert res.label == "scam"
    assert "financial_legal_coercion" in res.raw_output.get("detected_tactics", [])


def test_text_amazon_gift_card_scam():
    """Verifies prize/gift card smishing lure is detected as a threat."""
    text = "Congratulations! You've been selected to win a $500 Amazon Gift Card. Enter code 9921 to claim your reward."
    res = TextAdapter.predict_text(text)
    assert res.status == DetectorStatus.SUCCESS
    assert res.is_threat is True
    assert res.risk_score is not None
    score = res.risk_score if res.risk_score is not None else 0.0
    assert score >= 0.70
    assert res.label == "scam"
    assert "prize_lottery_lure" in res.raw_output.get("detected_tactics", [])


def test_text_grocery_list_benign():
    """Verifies benign grocery list is not misclassified as scam."""
    text = "Here is the grocery list: milk, eggs, sourdough bread, apples, and olive oil."
    res = TextAdapter.predict_text(text)
    assert res.status == DetectorStatus.SUCCESS
    assert res.is_threat is False
    assert res.risk_score is not None
    score = res.risk_score if res.risk_score is not None else 1.0
    assert score < 0.35
    assert res.label == "legitimate"


def test_text_doctor_appointment_benign():
    """Verifies confirmation message is classified as legitimate."""
    text = "Your appointment with Dr. Smith is confirmed for Thursday, Oct 28 at 2:00 PM. Reply 1 to confirm."
    res = TextAdapter.predict_text(text)
    assert res.status == DetectorStatus.SUCCESS
    assert res.is_threat is False
    assert res.risk_score is not None
    score = res.risk_score if res.risk_score is not None else 1.0
    assert score < 0.20
    assert res.label == "legitimate"


def test_text_casual_dinner_benign():
    """Verifies casual personal text is classified as legitimate."""
    text = "Hey Sarah, dinner tonight is moved to 7:30pm at the Italian restaurant on Main St."
    res = TextAdapter.predict_text(text)
    assert res.status == DetectorStatus.SUCCESS
    assert res.is_threat is False
    assert res.risk_score is not None
    score = res.risk_score if res.risk_score is not None else 1.0
    assert score < 0.20
    assert res.label == "legitimate"


def test_text_benign_urgency_not_flagged():
    """Verifies that the presence of the word 'urgent' alone does not cause a scam verdict."""
    text = "Urgent: please remember to buy milk on your way home today!"
    res = TextAdapter.predict_text(text)
    assert res.status == DetectorStatus.SUCCESS
    assert res.is_threat is False
    assert res.risk_score is not None
    score = res.risk_score if res.risk_score is not None else 1.0
    assert score < 0.35
    assert res.label == "legitimate"


def test_text_ambiguous_invoice_benign():
    """Verifies ambiguous business text without fraud lures remains benign."""
    text = "Please review the attached invoice from our meeting last Tuesday."
    res = TextAdapter.predict_text(text)
    assert res.status == DetectorStatus.SUCCESS
    assert res.is_threat is False
    assert res.risk_score is not None
    score = res.risk_score if res.risk_score is not None else 1.0
    assert score < 0.35
    assert res.label == "legitimate"


def test_text_empty_input_handled_safely():
    """Verifies empty text input returns a safe FAILED status."""
    res = TextAdapter.predict_text("")
    assert res.status == DetectorStatus.FAILED
    assert res.risk_score is None
    assert res.is_threat is False


# ==============================================================================
# PIPELINE AND FUSION INTEGRATION TESTS
# ==============================================================================


def test_pipeline_text_only():
    """Verifies pipeline execution with text modality alone."""
    pipeline = MultimodalPipeline(explainer_config=ExplainerConfig(provider="deterministic"))
    result = pipeline.analyze(
        text="IRS Notice: You owe $4,850 in back taxes. Pay now via prepaid card to avoid arrest: 1-800-555-0199"
    )
    assert "text" in result.fusion.active_modalities
    assert result.verdict in (OverallVerdict.SCAM, OverallVerdict.SUSPICIOUS)
    assert result.overall_risk_score >= 0.65
    assert any("irs notice" in ev.lower() or "coercion" in ev.lower() for ev in result.evidence_citations)


def test_pipeline_screenshot_only(screenshot_assets):
    """Verifies pipeline execution with screenshot modality alone."""
    path = screenshot_assets["phish_renamed"]
    if not path or not os.path.exists(path):
        pytest.skip("Phishing sample screenshot not found")

    pipeline = MultimodalPipeline(explainer_config=ExplainerConfig(provider="deterministic"))
    result = pipeline.analyze(screenshot_path=path)
    assert "screenshot" in result.fusion.active_modalities
    assert result.verdict in (OverallVerdict.SCAM, OverallVerdict.SUSPICIOUS)
    assert result.overall_risk_score >= 0.65


def test_pipeline_text_and_screenshot(screenshot_assets):
    """Verifies threat convergence across text and screenshot modalities."""
    path = screenshot_assets["phish_renamed"]
    if not path or not os.path.exists(path):
        pytest.skip("Phishing sample screenshot not found")

    pipeline = MultimodalPipeline(explainer_config=ExplainerConfig(provider="deterministic"))
    result = pipeline.analyze(
        text="URGENT: Your Wells Fargo account has been suspended due to unauthorized access.",
        screenshot_path=path,
    )
    assert "text" in result.fusion.active_modalities
    assert "screenshot" in result.fusion.active_modalities
    assert result.verdict in (OverallVerdict.SCAM, OverallVerdict.SUSPICIOUS)
    assert result.overall_risk_score >= 0.75


def test_pipeline_inconclusive_screenshot_does_not_pull_down_threat():
    """Verifies that an unreadable or blank screenshot does not dilute a high text threat."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        blank_path = f.name
    try:
        Image.new("RGB", (300, 300), color=(255, 255, 255)).save(blank_path)
        pipeline = MultimodalPipeline(explainer_config=ExplainerConfig(provider="deterministic"))
        result = pipeline.analyze(
            text="IRS Notice: You owe $4,850 in back taxes. Pay now via prepaid card: 1-800-555-0199",
            screenshot_path=blank_path,
        )
        # Screenshot is inconclusive, text is high risk
        assert "text" in result.fusion.active_modalities
        assert "screenshot" not in result.fusion.active_modalities
        assert result.verdict in (OverallVerdict.SCAM, OverallVerdict.SUSPICIOUS)
        assert result.overall_risk_score >= 0.65
    finally:
        if os.path.exists(blank_path):
            os.remove(blank_path)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

