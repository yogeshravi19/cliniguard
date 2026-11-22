# Phase 11: Application & Deployment Verification

## 1. Deployment Inventory
* **HTTP Server (`server.py`):** Serves `final_website/index.html`. Loads `final_website/cliniguard_model.joblib` and `final_website/cliniguard_scaler.joblib`. Imports signals directly from `cliniguard_pipeline.py`. **VERIFIED CONSISTENT**.
* **Streamlit Demo (`app_demo.py`):** Interactive UI. Loads model and scaler, computes 4 signals, displays color-coded triage banner. **VERIFIED CONSISTENT**.
* **FastAPI Backend (`web_portal/backend/main.py`):** Uses FastAPI framework with `/predict` endpoint.
  - *Issue Identified:* Hardcodes duplicated, legacy 29-term and 19-term lexicons with uncompiled substring matching (`any(t in x for t in DRUG_TERMS)`). Does NOT import from `cliniguard_pipeline.py`.
  - *Severity:* Major (Out of sync with updated pipeline).

## 2. Production Model Artifact Details
* `final_website/cliniguard_model.joblib`: Trained `LGBMClassifier`, 4 features input.
* `final_website/cliniguard_scaler.joblib`: `StandardScaler`, mean = `[0.6594, 0.5896, 0.1370, 0.6861]`, scale = `[0.3665, 0.4171, 0.2772, 0.1998]`.
