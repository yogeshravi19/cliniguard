# TABLES_REQUIRED.md: Verified Table Inventory

All tables in `RESEARCH_PAPER_CLINIGUARD.md` are 100% generated from verified genuine experimental results (6,914 samples from Med-HALT and MedHallu).

| Table # | Table Title / Scope | Source File in `research_paper/results/` | Experimental Protocol | Status |
|:---:|---|---|---|:---:|
| **Table 1** | Qualitative comparison of post-hoc hallucination detection paradigms | Conceptual analysis of literature ([21]–[29]) | Architectural comparison (no fabricated latency) | **COMPLETED** |
| **Table 2** | *(Reserved; omitted to align table numbers directly with sections)* | — | — | — |
| **Table 3** | Verified genuine benchmark dataset characteristics (Audit E0) | `E0_dataset_audit.csv` | Dataset Audit across genuine sources | **COMPLETED & VERIFIED** |
| **Table 4** | Headline and multi-seed pooled evaluation on genuine data | `E1_headline_seed42.csv`, `E1_pooled_10seed_summary.csv`, `E1_significance_tests.csv` | 10-Seed 70/30 stratified splits + 2,000 bootstrap | **COMPLETED & VERIFIED** |
| **Table 5** | Question-grouped 5-fold cross-validation results (Zero Question Leakage) | `E3_grouped_cv_summary.csv` | StratifiedGroupKFold on question groups | **COMPLETED & VERIFIED** |
| **Table 6** | Per-dataset evaluation on held-out test partitions | `E2_per_dataset_heldout.csv` | Disaggregated held-out test evaluation | **COMPLETED & VERIFIED** |
| **Table 7** | Zero-shot cross-dataset domain transfer | `E4_cross_dataset_transfer.csv` | Train on Source A -> Test on Source B | **COMPLETED & VERIFIED** |
| **Table 8** | Feature ablation study across 10 random splits | `E5_ablation_summary.csv` | 9 configurations across LightGBM & LR | **COMPLETED & VERIFIED** |
| **Table 9** | Raw univariate feature AUROC across genuine benchmarks | `E6_univariate_signal_auroc.csv` | Raw feature AUROC (Med-ISP, C-AAS, Med-EEM, CDT) | **COMPLETED & VERIFIED** |
| **Table 10** | Risk band stratification and probability calibration | `E8_band_analysis.csv`, `E8_calibration_metrics.json` | Observed hallucination rates per tier + ECE/Brier | **COMPLETED & VERIFIED** |
| **Table 11** | Cross-Specialty Extrinsic Clinical Mismatch & Clinical Specificity (Pillars 2 & 3) | `Pillar2_mismatch_metrics.json`, `Pillar2_mismatch_10seed_cv.csv` | 2,000 NIH pairs: 10-seed CV + Univariate CDT + Specificity audit | **COMPLETED & VERIFIED** |

---

## Future Experimental Tables (Marked for Expansion)
The following tables are reserved for future external validation (see `EXPERIMENTS_REQUIRED.md`):
* **Table 12 (Future X1)**: Live clinician-annotated EHR generation benchmark.
* **Table 13 (Future X5)**: Direct computational comparison with GPU-based SelfCheckGPT and GPT-4 judge.
* **Table 14 (Future X7)**: Automated UMLS/RxNorm ontological concept extraction comparison.
