# Audio Threat Detection Models

This module encapsulates the dual acoustic models trained for scam call and synthetic voice deepfake detection.

---

## 🎧 Models

1. **Scam Call Detection (`dataset/best_audio_scam_model.keras`)**
   - **Task**: Conversational scam vs normal call detection.
   - **Feature Extraction**: 64-band Log-Mel Spectrogram (16kHz mono, 5-second sample window).
   - **Classes**: `scam` vs `normal`.

2. **Voice Anti-Spoofing (`LA/best_model.keras`)**
   - **Task**: ASVspoof 2019 logical access voice clone / speech synthesis detection.
   - **Feature Extraction**: Log-Mel Spectrogram (16kHz mono).
   - **Classes**: `spoof / synthetic` vs `bonafide / genuine`.

---

## 📂 Structure

```
audio_model/
├── dataset/
│   ├── best_audio_scam_model.keras
│   ├── audio_scam_model_config.json
│   ├── SCAM_CALLS/
│   └── NORMAL_CALLS/
├── LA/
│   ├── best_model.keras
│   ├── preprocessing_config.json
│   └── training_log.csv
└── README.md
```
