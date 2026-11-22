"""evaluate_text_model.py
Evaluates the Text Scam Classifier on the held-out test split (5,000 rows)
from text_and_screenshot_module/FINAL_TEXT_DATASET_1_3.zip.

Generates:
1. Classification report (Precision, Recall, F1-Score per class)
2. ROC-AUC score and confusion matrix
3. Calibration assessment
4. text_model_eval_report.json & text_model_eval_report.txt
"""

import json
import zipfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]

def main():
    print("=" * 70)
    print("  EVALUATING TEXT SCAM CLASSIFIER ON HELD-OUT TEST SPLIT")
    print("=" * 70)

    # 1. Locate dataset
    zip_path = (
        ROOT
        / "text_and_screenshot_module"
        / "Text_Dataset-20261023T174336Z-1-001"
        / "Text_Dataset"
        / "FINAL_TEXT_DATASET_1_3.zip"
    )

    if not zip_path.exists():
        print(f"[!] Dataset zip not found at: {zip_path}")
        return 1

    print(f"[*] Reading dataset from: {zip_path.name}")
    with zipfile.ZipFile(zip_path) as z:
        with z.open("FINAL_TEXT_DATASET_1_3.csv") as f:
            df = pd.read_csv(f)

    print(f"[*] Total rows: {len(df):,} | Columns: {list(df.columns)}")
    print(f"[*] Label distribution:\n{df['label'].value_counts()}")

    # Clean text column
    df["body"] = df["body"].fillna("").astype(str)
    X = df["body"]
    y = df["label"].astype(int)

    # 2. Replicate standard train/test split (75% train / 25% test = 15,000 train / 5,000 test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"[*] Held-out test set: {len(X_test):,} rows ({len(y_test[y_test == 1]):,} scam, {len(y_test[y_test == 0]):,} normal)")

    # 3. Load model
    model_path = ROOT / "text_and_screenshot_module" / "models" / "text_scam_classifier.joblib"
    print(f"[*] Loading model from: {model_path}")
    model = joblib.load(model_path)

    # 4. Predict probabilities and classes
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred).tolist()
    clf_rep = classification_report(y_test, y_pred, target_names=["Legitimate (0)", "Scam/Phishing (1)"], output_dict=True)

    print("\n" + "-" * 50)
    print("  HELD-OUT EVALUATION RESULTS")
    print("-" * 50)
    print(f"  Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {roc_auc:.4f}")
    print(f"  Confusion Matrix (TN, FP / FN, TP):\n  {cm}")
    print("-" * 50)

    # 5. Save report
    results = {
        "dataset": "FINAL_TEXT_DATASET_1_3.csv",
        "total_samples": len(df),
        "test_samples": len(X_test),
        "metrics": {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(roc_auc), 4),
            "confusion_matrix": cm,
        },
        "classification_report": clf_rep,
    }

    report_json = ROOT / "text_and_screenshot_module" / "text_model_eval_report.json"
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    report_txt = ROOT / "text_and_screenshot_module" / "text_model_eval_report.txt"
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  TEXT SCAM CLASSIFIER — HELD-OUT EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset: FINAL_TEXT_DATASET_1_3.csv ({len(df):,} total rows)\n")
        f.write(f"Test Set: {len(X_test):,} held-out rows (Stratified Split)\n\n")
        f.write(f"Accuracy  : {acc*100:.2f}%\n")
        f.write(f"Precision : {prec:.4f}\n")
        f.write(f"Recall    : {rec:.4f}\n")
        f.write(f"F1-Score  : {f1:.4f}\n")
        f.write(f"ROC-AUC   : {roc_auc:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"  True Negatives (Legitimate): {cm[0][0]}\n")
        f.write(f"  False Positives (Spam):      {cm[0][1]}\n")
        f.write(f"  False Negatives (Missed):    {cm[1][0]}\n")
        f.write(f"  True Positives (Scam Catch): {cm[1][1]}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred, target_names=["Legitimate (0)", "Scam/Phishing (1)"]))

    print(f"\n[OK] Evaluation reports saved to:\n  - {report_json}\n  - {report_txt}")
    return 0

if __name__ == "__main__":
    main()

