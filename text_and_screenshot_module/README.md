# Text and Screenshot Threat Datasets (Text and Screenshot Module)

This directory contains the labeled datasets for conversational text/SMS scam detection and visual computer vision screenshot phishing analysis.

---

## 📱 Modules & Datasets

1. **Screenshot Visual Threat Dataset (`Screenshot_Dataset-20261023T174631Z-1-001/`)**
   - **Archive**: `FINAL_DATASET_1_3.zip` (~1.8 GB)
   - **Task**: Computer vision classification of deceptive web pages, fake banking dialogs, and phishing login portals.
   - **Classes**:
     - `phishing/`: Screenshots of brand-spoofed interfaces, fake credential harvest forms, and malicious pages.
     - `legitimate/`: Screenshots of legitimate websites, official login dialogs, and standard applications.
   - **Adapter Integration**: Handled by [`adapters/screenshot_adapter.py`](file:///c:/Users/Abhishek/OneDrive/Desktop/LLM%20integration/adapters/screenshot_adapter.py).

2. **Text & SMS Scam Dataset (`Text_Dataset-20261023T174336Z-1-001/`)**
   - **Archive**: `FINAL_TEXT_DATASET_1_3.zip` (~8.7 MB)
   - **Data File**: `FINAL_TEXT_DATASET_1_3.csv`
   - **Task**: NLP text classification of smishing, urgency tactics, fraudulent OTP requests, and impersonation messages.
   - **Adapter Integration**: Handled by [`adapters/text_adapter.py`](file:///c:/Users/Abhishek/OneDrive/Desktop/LLM%20integration/adapters/text_adapter.py).

---

## 📂 Directory Structure

```
text and screenshot/
├── Screenshot_Dataset-20261023T174631Z-1-001/
│   └── Screenshot_Dataset/
│       └── FINAL_DATASET_1_3.zip         # 1.8GB labeled screenshots (legitimate vs phishing)
├── Text_Dataset-20261023T174336Z-1-001/
│   └── Text_Dataset/
│       └── FINAL_TEXT_DATASET_1_3.zip    # FINAL_TEXT_DATASET_1_3.csv (SMS & scam text records)
└── README.md                             # Text & screenshot module documentation
```
