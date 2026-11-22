# Phase 1: Complete Project Inventory and System Architecture Audit

## 1. Project Identification and Research Problem
* **Project Name:** CliniGuard
* **Core Research Problem:** Detection and triage of clinical hallucinations in Large Language Model (LLM) generated medical texts.
* **Original Premise:** Proposed a lightweight, CPU-efficient, privacy-preserving closed-API hallucination detector using four surface lexical features (Med-ISP, C-AAS, Med-EEM, CDT).
* **Audited Reality:** Rigorous empirical evaluation demonstrated that the four features do not form a robust generalizable clinical detector. Instead, performance is dominated by prompt unigram cosine distance (CDT) and answer length, while the benchmark Med-HALT exhibits severe shortcut learning (TF-IDF AUROC = 1.0000) and detectors collapse under cross-dataset domain shift.

## 2. Directory and File Inventory

| Path | Purpose | Audit Status | Key Contents / Functionality |
| :--- | :--- | :---: | :--- |
| `cliniguard_pipeline.py` | Core feature extraction and preprocessing | **VERIFIED** | Implements tokenization, lookbehind unit regex, 4 signals, length, and task classifier. |
| `load_datasets.py` | Dataset ingestion and filtering | **VERIFIED** | Ingests Med-HALT and MedHallu; excludes synthetic sources (PubMedQA, MedQuAD). |
| `research_paper/scripts/run_experiments.py` | Comprehensive benchmark experiment suite | **VERIFIED** | Runs 10 seeds, grouped CV, cross-dataset transfer, ablation, band analysis, TF-IDF. |
| `research_paper/scripts/make_figures.py` | Publication-ready visual plotting | **VERIFIED** | Generates all 8 high-resolution figures in `research_paper/figures/`. |
| `research_paper/scripts/generate_paper_tables.py` | Automatic LaTeX/Markdown table formatting | **VERIFIED** | Extracts exact deterministic metrics from CSVs into tables. |
| `data_extraction/Med-HALT/medhalt.parquet` | Genuine benchmark dataset 1 | **VERIFIED** | $N=4,914$ records (1,989 fake / 2,925 real PubMed abstracts). |
| `data_extraction/MedHallu/medhallu.parquet` | Genuine benchmark dataset 2 | **VERIFIED** | $N=2,000$ records (1,000 ground truth / 1,000 hallucinated QA pairs). |
| `research_paper/results/predictions_seed42_test.csv` | Raw test predictions | **VERIFIED** | $N=2,075$ held-out test predictions across all 12 evaluated model architectures. |
| `research_paper/results/E1_pooled_all_runs.csv` | Multi-seed execution logs | **VERIFIED** | 120 individual runs across 10 stratified random seeds (0 through 9). |
| `research_paper/results/E1_significance_tests.csv` | Statistical significance | **VERIFIED** | Paired t-tests, standard errors, and 95% confidence intervals vs LightGBM. |
| `research_paper/results/E4_cross_dataset_transfer.csv` | Domain shift transfer results | **VERIFIED** | Med-HALT $\leftrightarrow$ MedHallu zero-shot transfer evaluations. |
| `research_paper/results/E8_band_analysis.csv` | Clinical safety triage simulation | **VERIFIED** | GREEN/AMBER/RED coverage and False-Reassurance Rates. |

## 3. Project Objectives & Claimed Contributions
1. **Lightweight Triage:** Evaluating whether surface-level features can triage clinical outputs before clinician review.
2. **Benchmark Artifact Exposure:** Demonstrating that benchmark datasets like Med-HALT contain superficial synthetic markers that allow linear models to achieve near-perfect discrimination without clinical reasoning.
3. **Domain Transfer Collapse:** Documenting that surface-level models fail to generalize across clinical task formats (abstract completion vs clinical question answering).
