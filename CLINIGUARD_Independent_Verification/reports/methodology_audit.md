# Phase 2 & 6: Research Methodology & Scientific Reliability Audit

## 1. Experimental Design & Split Rigor
* **Multi-Seed Evaluation:** 10 stratified random splits (seeds 0 through 9) evaluated in `research_paper/results/E1_pooled_all_runs.csv`.
* **Question-Grouped Cross-Validation:** 5-fold grouped CV evaluated in `E3_grouped_cv_summary.csv` to ensure paired human-truth and LLM-hallucination answers from the same question are never split across train and test sets.
* **Factorial Design:** Evaluated both Logistic Regression and LightGBM across `CDT + Length`, `Answer Length`, and the 4 lexical signals to decouple model architecture effects from feature representation.

## 2. Scientific Reliability & Benchmark Validity Findings
* **The "One-Feature" Finding:** Ablation studies show that `CDT + Length (LGB)` achieves **0.9572 AUROC**. Adding the other three domain signals (`Med-ISP`, `C-AAS`, `Med-EEM`) yields **0.9593 AUROC** (+0.0021 gain). Prompt distance and answer length account for $99.8\%$ of the signal.
* **Shortcut Learning on Med-HALT:** TF-IDF achieves **1.0000 AUROC** on Med-HALT due to synthetic distractors containing words like `fictional`, `quantum`, and `whimsical` compared to structured PubMed headers (`Background:`, `Methods:`).
* **Cross-Dataset Generalization Failure:** Models trained on Med-HALT fail when evaluated on MedHallu (**0.5518 AUROC**, near chance). TF-IDF rank correlation inverts (**0.3973 AUROC**) because human doctors hedge while LLMs assert certainty.
