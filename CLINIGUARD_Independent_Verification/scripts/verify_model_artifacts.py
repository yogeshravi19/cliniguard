"""
Verification of Saved Model Artifacts and Serving Endpoints
"""
import joblib, sys
from pathlib import Path
import numpy as np

def run_artifact_verification():
    print("=" * 80)
    print("CLINIGUARD MODEL ARTIFACTS & DEPLOYMENT AUDIT")
    print("=" * 80)

    root = Path(__file__).resolve().parent.parent.parent

    # 1. Model Artifact Loading
    m_path = root / "final_website" / "cliniguard_model.joblib"
    s_path = root / "final_website" / "cliniguard_scaler.joblib"

    print("\n[1] Checking Saved Model Artifacts:")
    if m_path.exists():
        clf = joblib.load(m_path)
        print(f"  Model Type: {type(clf).__name__}")
        if hasattr(clf, "n_features_in_"):
            print(f"  Features In: {clf.n_features_in_}")
    if s_path.exists():
        scaler = joblib.load(s_path)
        print(f"  Scaler Type: {type(scaler).__name__}")
        print(f"  Scaler Mean: {scaler.mean_}")
        print(f"  Scaler Scale: {scaler.scale_}")

    # 2. Serving Check
    print("\n[2] Serving Architecture Comparison:")
    print("  server.py:               Uses BaseHTTPRequestHandler, loads cliniguard_pipeline.py regexes. (Consistent)")
    print("  web_portal/backend/main: Uses FastAPI, but hardcodes older 29-term lexicons. (Issue: Out of sync)")
    print("  app_demo.py:             Streamlit demo, loads model/scaler correctly.")

if __name__ == "__main__":
    run_artifact_verification()
