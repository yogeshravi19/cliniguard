# CliniGuard Research Paper Repository

This directory contains the complete, self-contained research paper deliverables for **CliniGuard**. It is isolated from the application codebase to preserve system stability.

---

## Deliverables Summary

| File Name | Purpose / Contents |
|---|---|
| **[RESEARCH_PAPER_CLINIGUARD.docx](RESEARCH_PAPER_CLINIGUARD.docx)** | Publication-grade Microsoft Word document (772 KB) formatted according to academic submission standards with embedded 200 DPI figures and styled tables. |
| **[RESEARCH_PAPER_CLINIGUARD.md](RESEARCH_PAPER_CLINIGUARD.md)** | The complete academic research manuscript following the IMRAD format and academic template guidelines. Verified strictly on genuine datasets across all 4 pillars. |
| **[SAMPLE_COUNT_AUDIT.md](SAMPLE_COUNT_AUDIT.md)** | Code-level forensic sample-count audit resolving dataset conflicts, rejecting the legacy 9,236 count, and resolving cross-dataset question overlap. |
| **[FULL_6_DATASET_IMPLEMENTATION_PLAN.md](FULL_6_DATASET_IMPLEMENTATION_PLAN.md)** | Technical architecture for utilizing all six candidate datasets across the 4-pillar evaluation framework. |
| **[DATASET_AUDIT_FINAL.md](DATASET_AUDIT_FINAL.md)** | Full forensic audit of all six candidate datasets, documenting why four synthetic sources were excluded from supervised training and verifying genuine sources. |
| **[RESULTS_REPRODUCIBILITY_CHECK.md](RESULTS_REPRODUCIBILITY_CHECK.md)** | Direct line-by-line verification table mapping every number, table (Tables 1–11), and figure to the raw experimental results. |
| **[TABLES_REQUIRED.md](TABLES_REQUIRED.md)** | Complete inventory of all verified tables (Tables 1–11) and experimental sources. |
| **[FIGURES_REQUIRED.md](FIGURES_REQUIRED.md)** | Complete inventory of all verified figures (Figures 1–8). |
| **[EXPERIMENTS_REQUIRED.md](EXPERIMENTS_REQUIRED.md)** | Experimental roadmap detailing completed core experiments (E0–E9, Pillars 1–4) and prospective future extensions. |
| **`figures/`** | Publication-quality PNG figures (Figures 1–8, 200 DPI). |
| **`results/`** | Raw experimental result files (CSV, JSON) produced by `run_experiments.py` and `run_cross_specialty_mismatch.py`. |
| **`references/`** | Verified bibliography (`REFERENCES_VERIFIED.md`) and raw PDF first-page dumps. |
| **`scripts/`** | Python scripts for data extraction, experiment execution, cross-specialty benchmarking, figure rendering, docx compilation, and citation auditing. |

---

## How to Reproduce All Results From Scratch

The complete experimental workflow is automated and can be executed via command-line:

```bash
# Step 1: Extract verified genuine datasets (Med-HALT & MedHallu)
python load_datasets.py

# Step 2: Run complete primary evaluation suite (~6.5 minutes on CPU)
python research_paper/scripts/run_experiments.py

# Step 3: Run Cross-Specialty Extrinsic Mismatch & Specificity Benchmark (Pillars 2 & 3)
python research_paper/scripts/run_cross_specialty_mismatch.py

# Step 4: Render publication-quality figures
python research_paper/scripts/make_figures.py

# Step 5: Compile professional Word manuscript with embedded figures and tables
python research_paper/scripts/compile_docx.py
```

All generated metrics are saved to `results/` and figures to `figures/`.

---

## Key Experimental Findings

1. **Benchmark Integrity**: Pervasive synthetic sentence injections were uncovered in PubMedQA, MedQuAD, MedQuAD XML, and MedHall-Bench. Evaluating models on these contaminated sets yields artificially inflated scores. They have been formally excluded.
2. **Verified Genuine Accuracy**: On the 6,914-sample genuine benchmark pool (Med-HALT and MedHallu across 5,817 unique questions), CliniGuard (LightGBM) achieves **0.922 ± 0.005 AUROC** and **0.910 ± 0.006 AP**, significantly outperforming linear models ($p < 0.001$) and neural MLPs ($p = 0.003$).
3. **Question-Grouped 5-Fold Cross-Validation**: With zero question leakage across folds, CliniGuard maintains **0.923 ± 0.009 AUROC**.
4. **The TF-IDF Paradox**: While a surface TF-IDF n-gram baseline achieves 0.993 AUROC within-distribution by exploiting dataset-specific phrasing, it completely collapses to **0.397 AUROC** under zero-shot cross-dataset transfer. In contrast, CliniGuard's linguistic features achieve **0.936 AUROC** across domains.
5. **Actionable Clinical Stratification**: The calibrated **GREEN band (<0.35)** reduces error rate to **9.9%** (relative to a 43.2% base rate), while the **RED band (>0.65)** concentrates hallucinated answers at **88.2%**.
6. **Inference Latency**: Single-core CPU latency averages **15.1 ms** end-to-end.
