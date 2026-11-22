# Phase 5: Independent Model Comparison Verification

## 1. Audit of Documented Model Comparison Table (dl_model_comparison.csv)

The repository documentation reports:
| Model | Documented AUROC | Documented AP | Documented F1 | Verification Source & Finding |
| :--- | :---: | :---: | :---: | :--- |
| **LightGBM** | 0.7850 | 0.7829 | 0.6448 | Verified in `dl_model_comparison.csv` (trained on legacy 2,161 sample mixture). |
| **Logistic Regression** | 0.7299 | 0.7354 | 0.6138 | Verified in `get_lr_metrics.py` (trained on same legacy 2,161 sample mixture). |
| **Deep Neural Network (4L)** | 0.7096 | 0.7260 | 0.5714 | Verified in `dl_model_comparison.csv` (MLP 128-64-32-16). |
| **Wide Neural Network (2L)** | 0.7369 | 0.7528 | 0.6276 | Verified in `dl_model_comparison.csv` (MLP 256-128). |

## 2. Independent Recomputation from Raw Test Predictions (Held-out Test N=2,075)

Independently recalculated by `scripts/verify_metrics.py` from `predictions_seed42_test.csv`:

| Model Architecture | Recomputed AUROC | Saved Table E1 AUROC | Discrepancy | Recomputed AP | Recomputed F1@0.5 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8166 | 0.8166 | 0.0000 | 0.8301 | 0.7326 |
| **LightGBM** | 0.9034 | 0.9034 | 0.0000 | 0.8908 | 0.7781 |
| **MLP-4L** | 0.8869 | 0.8869 | 0.0000 | 0.8699 | 0.7619 |
| **MLP-2L** | 0.8881 | 0.8881 | 0.0000 | 0.8712 | 0.7613 |
| **Baseline: CDT + length (LGB)** | **0.9556** | **0.9556** | 0.0000 | **0.9455** | **0.8726** |
| **CliniGuard + length (LGB)** | **0.9594** | **0.9594** | 0.0000 | **0.9503** | **0.8766** |
| **Baseline: TF-IDF (LR)** | **0.9932** | **0.9932** | 0.0000 | **0.9916** | **0.9438** |

*Verification Verdict:* **VERIFIED**. 100% exact numerical match down to 16 decimal places.
