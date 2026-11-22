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

## Key Experimental Findings (corrected November 2026)

1. **Pooled in-distribution result:** On 6,914 items (Med-HALT + MedHallu) LightGBM reaches 0.922 +/- 0.005 AUROC (10 splits) and 0.923 +/- 0.009 (question-grouped CV).
2. **The pooled number hides a large gap:** 0.976 AUROC on Med-HALT (positives are overtly fabricated papers) versus 0.683 on MedHallu, where answer length alone scores 0.745.
3. **Transfer is inconsistent:** LightGBM 0.713 / 0.549 across the two directions; TF-IDF drops from 0.993 to 0.397. No claim of robust generalization is made.
4. **One signal dominates:** dropping CDT costs 0.115 AUROC; CDT alone reaches 0.817. Its sign differs between the hallucination benchmarks and the NIH topic-mismatch check (zero-shot LightGBM AUROC 0.479).
5. **Med-ISP is mostly substring noise:** 93% of its hits are not whole-word lexicon matches; whole-word matching lowers its single-signal AUROC from 0.649 to 0.549.
6. **Tiers depend on prevalence:** 88% RED precision holds at the benchmark base rate (43%); at an assumed 10% prevalence it would be about 52% (illustrative).
7. **Not done:** live LLM outputs, clinician labels, comparison with published detectors.
