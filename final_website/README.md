# CLINIGUARD — Final Website

## Files in this folder

| File | Description |
|---|---|
| `index.html` | Complete SPA frontend — 6 tabs, dark‑blue theme, all project details |
| `main.py` | FastAPI backend — `/predict`, `/health`, `/model/download` endpoints |
| `cliniguard_model.joblib` | Trained LightGBM fusion model (AUROC 0.73) |
| `PROJECT_OVERVIEW.md` | Full project documentation |

## How to Run

```bash
# From this folder:
pip install fastapi uvicorn lightgbm scikit-learn joblib

python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Then open **http://127.0.0.1:8000/** in your browser.

## Tabs

| Tab | Content |
|---|---|
| Home | Project overview, stats, problem statement, gap analysis |
| Signals | All 4 signals with exact formulas, learned weights, fusion equation |
| Datasets | All 6 datasets with descriptions, AUROC analysis, row counts |
| Model | LightGBM architecture, hyper-parameters, baseline comparison |
| Analytics | Animated AUROC chart, ablation study, 5-fold CV results |
| Assess | Live hallucination detection with signal breakdown |

## Model Performance

- **AUROC**: 0.7299
- **Avg Precision**: 0.7354
- **F1-Score**: 0.6138
- **5-Fold CV AUROC**: 0.72 ± 0.03

---
*CLINIGUARD · Multi-Signal Medical Hallucination Detector · 2026*
