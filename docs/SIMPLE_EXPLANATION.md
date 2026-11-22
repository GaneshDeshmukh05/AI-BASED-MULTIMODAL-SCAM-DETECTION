# Multimodal Scam Detection System — Simple Plain-English Guide

> **Who is this guide for?**  
> Students, evaluators, teachers, or anyone who wants to understand this project **in plain, everyday English** without getting lost in complicated mathematical jargon.

---

## 1. What Does This Project Do? (The 1-Minute Pitch)

Imagine a security guard who protects you from fraudsters across every channel:
* When a scammer **calls you on the phone**,
* When they send an **urgent text message or WhatsApp**,
* When they give you a **link to click**, and
* When they show you a **fake bank website screen**.

Normally, people use separate apps: an antivirus checks a file, Truecaller checks a phone number, and a browser checks a link. 

**The problem?** Attackers use all four channels together! A scammer might use a polite voice call, but pair it with a deadly fake bank link. If your security app only looks at the voice call, it thinks everything is safe. 

**Our System's Solution:** It checks **all four channels at the exact same time**. If even **one** part is a confirmed scam, it sounds the alarm and explains to you in plain English exactly what to do to protect your money.

---

## 2. A Real-Life Example of How Scams Work

```
[Phone Call] ─────────► "Hello sir, I am calling from your bank branch."
                           (Sounds friendly and polite)
                                 │
[SMS Message] ────────► "ALERT: Your account will be locked in 24 hours!"
                           (Creates artificial panic and rush)
                                 │
[Website Link] ───────► "Click: http://paypal.com-account-verify.online"
                           (Look closely: it ends in .online, not paypal.com!)
                                 │
[Login Screen] ───────► Fake page asking for your Password & OTP
```

If an AI only checks the phone call, it might say: *"Sounds normal!"*  
If an AI checks only the website, you might already have shared your OTP before checking.  

**Our Decision Fusion Brain puts all the puzzle pieces together.** It notices that the SMS is creating artificial panic and the link is fake, so it declares:  
**🚨 VERDICT: SCAM (100% Risk) — Do NOT enter your credentials!**

---

## 3. How Each Part Works (Simple Words)

### 📞 1. The Audio Model (Phone Calls)
* **What it looks for**: Voice deepfakes (computer-generated fake voices) and conversational scam tactics.
* **How it works**: It turns the sound waves of the call into a visual picture called a **spectrogram** (like a voice fingerprint). A Convolutional Neural Network (CNN) scans this picture to see if it matches patterns of known scam calls.

### 💬 2. The Text Model (SMS / Chat Messages)
* **What it looks for**: Panic triggers, fake rewards, and urgency.
* **How it works**: It uses Natural Language Processing (NLP) to detect:
  * *"Your account is blocked"* (Fear/Panic)
  * *"You won $50,000 lottery, pay $100 fee to release it"* (Greed/Prize lure)
  * *"Send your OTP to verify"* (Credential theft)
* **Performance**: Tested on 5,000 real messages with **98.42% accuracy**.

### 🔗 3. The URL Model (Web Links)
* **What it looks for**: Fake websites pretending to be real companies (like PayPal, Chase, SBI, or Amazon).
* **How it works**: It extracts **26 clues** from the link:
  * Does it use an IP address instead of a real name? (`http://192.168.1.1/login`)
  * Does it have sneaky subdomains? (`paypal.com-verify.online`)
  * Was the domain bought just 2 days ago?
  * Does it lack proper HTTPS security certificates?
* **Performance**: Tested on over 50,000 brand-new links with **93.94% accuracy**.

### 🖼️ 4. The Screenshot Model (Images & Webpages)
* **What it looks for**: Fake login boxes, password fields, and counterfeit bank logos.
* **How it works**: It uses **Optical Character Recognition (OCR)** to read every word visible on the screen and searches for password-stealing forms or fake security alerts.

### 🧠 5. The Decision Fusion Engine (The "Brain")
This is the core innovation of our project. It follows three strict security rules:
1. **Safety First (Defense-in-Depth)**: If one channel is a proven dangerous scam (like a phishing link with 100% certainty), the overall result is immediately marked as **SCAM**. A safe phone call cannot average down or hide a malicious link!
2. **Conflict Warning**: If the phone call seems normal but the website is fake, it warns you: *"Attention: Conflicting signals detected! Treat with caution."*
3. **No Made-Up Stories**: The AI explanation engine only states facts that the detectors actually found. It will never invent fake reasons.

---

## 4. How to Run the Project (Just 2 Commands!)

You do not need to configure complex settings. Everything is pre-configured.

### 🖥️ Step 1: Open the Visual Web Dashboard
In your PowerShell or terminal, type:
```powershell
npm run dev
```
*(Or `python -m streamlit run app.py`)*  
Now open your web browser to: **`http://localhost:8501`**

You will see:
* An audio player to test call recordings.
* An input box for URLs.
* An input box for SMS messages.
* An image uploader for screenshots.
* A colorful **Threat Speedometer Gauge** that shows the risk level (Green = Safe, Orange = Suspicious, Red = Scam).

### ⚙️ Step 2: Run the API for Mobile Apps (Optional)
If you want to connect a mobile app or React website to this AI:
```powershell
npm run api
```
Now the API is running at: **`http://localhost:8000/docs`**

---

## 5. Quick Test Cases You Can Try Right Now

| What to test | What to paste | Expected Result |
|---|---|---|
| **Phishing Link** | `http://paypal.com-account-verify.online/login` | 🚨 **SCAM (100% Risk)** |
| **Urgent Fraud Text** | `URGENT: Your Chase bank account will be suspended in 24 hours. Verify your credentials immediately.` | 🚨 **SCAM (88% Risk)** |
| **Safe Website** | `https://www.google.com` | ✅ **SAFE (0% Risk)** |
| **Normal Message** | `Hi Mom, I will be home by 6 PM for dinner. Let me know if you need any groceries.` | ✅ **SAFE (10% Risk)** |

---

## 6. Common Viva / Examiner Questions (With Simple Answers)

### Q1: "Why not just use an LLM like ChatGPT for everything?"
> **Answer**: Large Language Models (LLMs) are expensive, slow, cannot listen to raw audio waveforms directly in real-time, and can hallucinate (make up facts). Instead, our system uses specialized, fast ML models (XGBoost, CNN, Naive Bayes) to compute mathematical probabilities, and uses the LLM only to generate a plain-English explanation of those facts.

### Q2: "What is Decision Fusion?"
> **Answer**: Decision fusion is the process of combining predictions from different models into one final verdict. Instead of calculating a simple average (which would let a scam link pass if the call sounded polite), our fusion uses **defense-in-depth security** where high-risk evidence always takes priority.

### Q3: "What is SHAP?"
> **Answer**: SHAP stands for *SHapley Additive exPlanations*. It is a mathematical technique that tells us exactly **why** the model made a decision. For example, it shows that a link was flagged because it had the word "login", lacked HTTPS, and had an unusually long web address.

### Q4: "Where is the frontend?"
> **Answer**: We built an interactive Streamlit web dashboard in `app.py` with Plotly risk gauges and SHAP charts, and a FastAPI REST API in `api.py` that allows any external mobile app or web frontend to connect using standard JSON.

---

## 7. Summary of Files

* [`app.py`](file:///c:/Users/Abhishek/OneDrive/Desktop/AI-BASED-MULTIMODAL-SCAM-DETECTION-main/app.py) — The interactive web browser dashboard.
* [`api.py`](file:///c:/Users/Abhishek/OneDrive/Desktop/AI-BASED-MULTIMODAL-SCAM-DETECTION-main/api.py) — The REST API service for mobile and web apps.
* [`main.py`](file:///c:/Users/Abhishek/OneDrive/Desktop/AI-BASED-MULTIMODAL-SCAM-DETECTION-main/main.py) — The command-line scanner tool.
* [`fusion/`](file:///c:/Users/Abhishek/OneDrive/Desktop/AI-BASED-MULTIMODAL-SCAM-DETECTION-main/fusion/) — The decision fusion brain and explanation generator.
* [`domain_intelligence/`](file:///c:/Users/Abhishek/OneDrive/Desktop/AI-BASED-MULTIMODAL-SCAM-DETECTION-main/domain_intelligence/) — The URL phishing detector and domain checker.
* [`text_and_screenshot_module/`](file:///c:/Users/Abhishek/OneDrive/Desktop/AI-BASED-MULTIMODAL-SCAM-DETECTION-main/text_and_screenshot_module/) — The SMS text and screenshot analyzers.
* [`audio_model/`](file:///c:/Users/Abhishek/OneDrive/Desktop/AI-BASED-MULTIMODAL-SCAM-DETECTION-main/audio_model/) — The voice scam and voice deepfake models.
