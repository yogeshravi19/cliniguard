import sys
import json
from pathlib import Path
import joblib
import pandas as pd
import math

# Import feature functions from the main pipeline (relative import)
from cliniguard_pipeline import tokenize, med_isp, c_aas, med_eem, cdt, classify_task, risk_label, FEATURES

MODEL_PATH = Path(__file__).parent / "cliniguard_model.joblib"
SCALER_PATH = Path(__file__).parent / "cliniguard_scaler.joblib"

def load_artifacts():
    clf = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return clf, scaler

def compute_features(question: str, answer: str) -> pd.DataFrame:
    # Build a one‑row DataFrame with the same columns used in training
    df = pd.DataFrame({
        "_question": [question],
        "_answer":   [answer]
    })
    df["med_isp"] = df["_answer"].apply(med_isp)
    df["c_aas"]   = df["_answer"].apply(c_aas)
    df["med_eem"] = df["_answer"].apply(med_eem)
    df["cdt"]     = df.apply(lambda r: cdt(r["_answer"], r["_question"]), axis=1)
    return df[FEATURES]

def predict(question: str, answer: str):
    clf, scaler = load_artifacts()
    X = compute_features(question, answer)
    X_scaled = scaler.transform(X)
    prob = clf.predict_proba(X_scaled)[:, 1][0]
    label = risk_label(prob)
    return {"risk_score": prob, "label": label}

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python cliniguard_inference.py \"<question>\" \"<answer>\"")
        sys.exit(1)
    q = sys.argv[1]
    a = sys.argv[2]
    result = predict(q, a)
    print(json.dumps(result, indent=2))
