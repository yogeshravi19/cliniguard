# Comprehensive Project Audit Report: CliniGuard

**Audit Date:** November 2026  
**Project Workspace:** `F:\cliniguard`  
**Lead Investigator:** Yogesh Ravi M, School of Computer Science and Engineering, Vellore Institute of Technology, Chennai  
**Audit Purpose:** Full-scope verification of research objectives, software implementation, local datasets, experimental results, and external benchmark sources.

---

## Executive Summary

This comprehensive audit was conducted across all files, codebases, local parquet storage, experimental result CSVs, and manuscript drafts in `F:\cliniguard`. The investigation confirms that **CliniGuard** has completed all core research milestones, with deterministic empirical verification across 6,914 authentic medical benchmark records and a 2,000-sample clinical topic-mismatch evaluation.

---

## Section A: Existing Project Status

### 1. Research Objective
CliniGuard investigates a fundamental question in clinical artificial intelligence safety:
> *Can four lightweight, post-generation lexical features (drug term sparsity, patient context sparsity, hedging entropy, and question-answer vocabulary drift) reliably detect hallucinations in medical LLM answers on a standard CPU, and where do surface-level lexical heuristics physically fail?*

The project deliberately contrasts with heavy, computationally expensive alternatives:
* Does **not** require repeated stochastic sampling (unlike SelfCheckGPT).
* Does **not** require external knowledge retrieval or live web lookups (unlike FActScore/FacTool).
* Does **not** require a secondary judge LLM (avoiding evaluator bias and hospital data transmission risks).
* Does **not** require access to internal neural network weights or hidden layer activations (allowing it to monitor closed commercial APIs such as GPT-4, Claude, and Gemini).

### 2. Implementation Architecture
* **Language & Runtime:** Python 3.8+ / 3.9 / 3.13 on standard x86 CPU.
* **Latency:** Median inference time of **15.1 ms** per QA pair.
* **Feature Extraction Pipeline (`cliniguard_pipeline.py` & `run_experiments.py`):**
  1. **Med-ISP (Medication Information Sparsity):** Evaluates answer density against a 34-item pharmacology lexicon ($L_D$).
  2. **C-AAS (Clinical Context-Awareness Sparsity):** Evaluates answer density against a 24-item clinical context lexicon ($L_C$).
  3. **Med-EEM (Medical Uncertainty Entropy Measure):** Evaluates binary Shannon entropy over a 21-item hedging word list ($L_U$).
  4. **CDT (Context Drift):** Computes cosine distance between Bag-of-Words vectors of prompt question $\mathbf{q}$ and answer $\mathbf{a}$.
* **Classification Engine:** **LightGBM** (Gradient-boosted decision trees, 100 estimators, max depth 4, learning rate 0.05) mapping the 4-dimensional score vector into **GREEN** (low risk), **AMBER** (manual clinician review required), and **RED** (high risk of error) triage tiers.

### 3. Empirical Results Verified Against Raw Data
* **Pooled Benchmark AUROC (Med-HALT + MedHallu, 6,914 items):**
  * 10-Seed Random Split: **$0.922 \pm 0.005$**
  * 5-Fold Question-Grouped Cross-Validation: **$0.923 \pm 0.009$**
* **The Benchmark Disparity (Key Finding):**
  * **Med-HALT (4,914 items):** AUROC = **0.976** (synthetic paper fabrication is easily captured by lexical drift).
  * **MedHallu (2,000 items):** AUROC = **0.683** (realistic clinical falsehoods evade surface word filters).
* **Cross-Benchmark Transfer:**
  * Train Med-HALT $\rightarrow$ Test MedHallu: LightGBM AUROC = **0.713** (TF-IDF Baseline = 0.518).
  * Train MedHallu $\rightarrow$ Test Med-HALT: LightGBM AUROC = **0.549** (TF-IDF Baseline = 0.397).
* **Feature Importance:** CDT is the dominant predictor; removing CDT drops AUROC by 0.115.

---

## Section B: Existing Dataset Inventory

A code-level audit was executed across all raw parquet files in `data_extraction/`. All files remain intact with zero modifications:

| Dataset Name | Local Path | Format | Size (Bytes) | SHA-256 (Prefix) | Total Rows | Label 0 (Real) | Label 1 (Halluc.) | Unique Prompts | Verification Role |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Med-HALT** | `data_extraction/Med-HALT/medhalt.parquet` | Parquet | 5,222,639 | `f58b8e77f644` | 4,914 | 2,925 (59.5%) | 1,989 (40.5%) | 4,817 | **Verified Primary Benchmark 1** |
| **MedHallu** | `data_extraction/MedHallu/medhallu.parquet` | Parquet | 1,094,799 | `8b68acbbae38` | 2,000 | 1,000 (50.0%) | 1,000 (50.0%) | 1,000 | **Verified Primary Benchmark 2** |
| **PubMedQA** | `data_extraction/pubmedqa/pubmedqa.parquet` | Parquet | 928,645 | `b58e4f4e5278` | 1,000 | 666 (66.6%) | 334 (33.4%) | 1,000 | **Excluded** (No native hallucination labels; 100% prompt overlap with MedHallu) |
| **MedQuAD** | `data_extraction/medquad/medquad.parquet` | Parquet | 9,342,818 | `640c5df42ba9` | 47,441 | 31,627 (66.7%) | 15,814 (33.3%) | 44,603 | **Verified Task-Restricted** (Pillar 2 Synthetic Topic Mismatch Check) |
| **MedQuAD XML** | `data_extraction/github/github.parquet` | Parquet | 78,866 | `663abd53a89f` | 107 | 83 (77.6%) | 24 (22.4%) | 75 | **Excluded** (100% duplicate subset of MedQuAD) |
| **MedHall-Bench** | `data_extraction/medhall_bench/medhall_bench.parquet` | Parquet | 17,674 | `3efa4ff9afbc` | 54 | 27 (50.0%) | 27 (50.0%) | 27 | **Excluded** (Evaluation rubric test harness without paired LLM outputs) |

---

## Section C: Research Verification Status

1. **Experimental Traceability (E0 – E9):**
   * All metrics reported in the final paper (`RESEARCH_PAPER_CLINIGUARD.md` and compiled docx) match with 100% numerical fidelity to the CSV outputs in `research_paper/results/`.
   * P-values for paired DeLong and Wilcoxon significance tests are verified in `E1_significance_tests.csv`.
2. **Leakage Elimination:**
   * Group-preserved 5-fold cross-validation (`E3_grouped_cv_all_folds.csv`) explicitly grouped by `question` hashes ensures zero shared prompts between train and test splits.
3. **Pillar 2 Synthetic Mismatch Traceability:**
   * 2,000 paired questions and answers sampled from NIH MedQuAD (`research_paper/results/Pillar2_mismatch_10seed_cv.csv`) verified that CDT reaches 0.796 AUROC on pure topic drift, while pre-trained models fail (0.479) due to inverted feature directionality.

---

## Section D: Gaps Identified and Resolved

1. **Legacy Loader Artifact Identified:**
   * An earlier draft script (`cliniguard_pipeline.py`) synthesized pseudo-hallucination labels on PubMedQA and MedQuAD using fixed string injections and modulo row indexing (`row_index % 3 == 0`).
   * **Resolution:** Fully documented and isolated. The primary research paper now strictly uses authentic benchmarks (Med-HALT and MedHallu) and repurposes MedQuAD exclusively for topic-mismatch validation.
2. **Medical Benchmark Context Completeness:**
   * To establish CliniGuard's position within the wider medical AI landscape, the evaluation ecosystem required linking to the 4 foundational medical benchmarks cited in literature (MedQA-USMLE, MedMCQA, MMLU Medical, TruthfulQA Health).
   * **Resolution:** Successfully audited, fetched, and cached sample validation suites in `Dataset_Source_and_Reproducibility_Audit/datasets/`.

---

## Section E: Recommended Actions

1. **Retain All Existing Research:** Preserve `research_paper/results/` and the compiled manuscripts (`RESEARCH_PAPER_CLINIGUARD_IEEE_FINAL.docx`).
2. **Deploy Standalone Colab Notebook:** Maintain `Dataset_Verification_Colab.ipynb` for independent third-party and journal reviewer replication.
3. **Submit to Top Venue:** CliniGuard is fully verified and ready for journal submission (e.g., *Journal of Biomedical Informatics*, *IEEE Journal of Biomedical and Health Informatics*).
