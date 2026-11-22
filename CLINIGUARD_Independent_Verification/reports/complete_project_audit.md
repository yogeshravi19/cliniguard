# Phase 1: Complete Project Inventory and System Architecture Audit

## 1. Project Overview & GitHub Repository Comparison
* **Project Name:** CLINIGUARD — Medical AI Hallucination Detector
* **GitHub Repository:** https://github.com/yogeshravi19/cliniguard
* **Research Problem:** Detecting hallucinations in Large Language Model (LLM) medical answers without accessing internal token logits or closed model weights.
* **Core Proposal:** A lightweight screening system based on four domain features (Med-ISP, C-AAS, Med-EEM, CDT) combined using LightGBM.

## 2. Directory & Script Inventory

| File / Directory | Purpose in Repository | Actual Execution Role | Audit Status |
| :--- | :--- | :--- | :---: |
| `cliniguard_pipeline.py` | Core feature extraction, lexicons, and training logic | Active authoritative pipeline | **VERIFIED** |
| `load_datasets.py` | Data ingestion & filtering from Hugging Face | Ingests Med-HALT & MedHallu; filters synthetic sources | **VERIFIED** |
| `train_model.py` | Legacy training script on `cliniguard_all_datasets.csv` | Legacy training artifact creator | **ARCHIVED/LEGACY** |
| `dl_model_comparison.py` | Script comparing LightGBM vs MLPs | Generated `dl_model_comparison.csv` (2,161 sample mixture) | **VERIFIED LEGACY** |
| `get_lr_metrics.py` | Standalone script evaluating Logistic Regression | Evaluated LR on same 2,161 sample split | **VERIFIED LEGACY** |
| `server.py` | Local HTTP server serving `final_website/` | Serving endpoint for UI demo | **VERIFIED** |
| `app_demo.py` | Interactive Streamlit demo | Front-end interactive evaluator | **VERIFIED** |
| `web_portal/backend/main.py` | FastAPI backend implementation | Alternative API server | **ISSUE IDENTIFIED (Out of sync)** |
| `cliniguard_results.csv` | Output predictions on 2,161 sample legacy mixture | Legacy results record | **VERIFIED** |
| `dl_model_comparison.csv` | Saved 3-model comparison metrics | Legacy comparison table | **VERIFIED** |
| `cliniguard_all_datasets.csv` | 55,518 row legacy CSV mixture | Legacy dataset mixture (contains synthetic labels) | **ISSUE IDENTIFIED (Contaminated)** |
| `final_website/cliniguard_model.joblib` | Trained LightGBM model artifact | Production model loaded by server.py | **VERIFIED** |
| `final_website/cliniguard_scaler.joblib` | Trained StandardScaler artifact | Production scaler loaded by server.py | **VERIFIED** |
