# CliniGuard Reproducibility and Audit Report

**Date:** November 2026  
**Auditor:** Automated Project Audit System  
**Repository:** `F:\cliniguard`  
**Target Environment:** Local Python 3.8+ / Python 3.9 / Google Colab (Python 3.10+)

---

## 1. System and Execution Environment

```json
{
  "system": "Windows 10 / Linux / Google Colab",
  "python_version": "3.8.10 / 3.9.13 / 3.13.0",
  "scikit_learn_version": "1.3.2",
  "lightgbm_version": "4.6.0",
  "pandas_version": ">=1.3.0",
  "primary_dataset_rows": 6914,
  "headline_seed": 42,
  "cross_validation_seeds": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
}
```

---

## 2. Status of Research Findings: Verified vs. Unresolved

### A. Independently Verified Findings (100% Deterministic)

1. **Pooled Benchmark AUROC:**
   * Stratified 10-seed average: **$0.922 \pm 0.005$**
   * Question-grouped 5-fold CV average: **$0.923 \pm 0.009$**
   * *Status:* Fully verified against `research_paper/results/E1_pooled_all_runs.csv` and `E3_grouped_cv_all_folds.csv`.

2. **Per-Dataset Asymmetry:**
   * Med-HALT alone: **$0.976 \pm 0.003$** AUROC
   * MedHallu alone: **$0.683 \pm 0.012$** AUROC
   * *Status:* Fully verified against `research_paper/results/E2_per_dataset_heldout.csv`.

3. **Baseline Comparison:**
   * Answer Length baseline on MedHallu: AUROC = **0.745** (outperforms CliniGuard's 0.683).
   * Bag-of-Words / TF-IDF Logistic Regression: 0.993 pooled AUROC in-distribution, but collapses to **0.397** on MedHallu $\rightarrow$ Med-HALT cross-dataset transfer.
   * *Status:* Fully verified against `research_paper/results/E1_headline_seed42.csv` and `E4_cross_dataset_transfer.csv`.

4. **Ablation Importance:**
   * Dropping CDT: AUROC drops from 0.922 to **0.807** ($\Delta = -0.115$).
   * CDT alone: AUROC = **0.817**.
   * *Status:* Fully verified against `research_paper/results/E5_ablation_summary.csv`.

5. **Inference Latency:**
   * Median execution time per sample: **15.1 ms** on an Intel CPU.
   * *Status:* Fully verified against `research_paper/results/E9_latency.json`.

6. **Pillar 2 Synthetic Topic Mismatch Check:**
   * CDT on 2,000 paired NIH MedQuAD samples: **0.796** AUROC.
   * Model pre-trained on Med-HALT/MedHallu: **0.479** AUROC (fails due to feature direction flip).
   * *Status:* Fully verified against `research_paper/results/Pillar2_mismatch_10seed_cv.csv`.

---

### B. Unresolved / Boundary Limitations (Explicitly Disclosed in Manuscript)

1. **Substring Collisions in Drug Lexicons:**
   * The current substring matcher in `Med-ISP` matches short chemical strings inside ordinary English words (e.g., `"iv"` inside `"given"`, `"table"` inside `"comfortable"`).
   * *Academic Disclosure:* Explicitly discussed as a known limitation in Section V.C of the manuscript.

2. **Closed-Vocabulary Coverage:**
   * The lexicons contain 34 drug terms, 24 clinical terms, and 21 hedging terms. Novel medications or complex medical phrasing outside this vocabulary rely entirely on CDT.
   * *Academic Disclosure:* Framed as a lightweight preliminary screen, not a definitive clinical diagnostic system.

---

## 3. Step-by-Step Reproduction Guide

To replicate the entire audit and verification pipeline locally:

```bash
# 1. Activate virtual environment
cd F:\cliniguard
.venv\Scripts\activate   # on Windows

# 2. Run local dataset verification
python Dataset_Source_and_Reproducibility_Audit\scripts\verify_existing_datasets.py

# 3. Verify external benchmark connections & cache samples
python Dataset_Source_and_Reproducibility_Audit\scripts\download_missing_datasets.py

# 4. Optional: Run the complete experimental suite from scratch (approx. 2 minutes on CPU)
python research_paper\scripts\run_experiments.py
```

All generated tables, CSVs, and figures are deterministically reproducible.
