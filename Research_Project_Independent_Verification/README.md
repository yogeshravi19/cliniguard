# Research Project Independent Verification Workspace

This directory contains the complete independent verification suite, execution scripts, evidence logs, and audit reports for the **CliniGuard** research project.

## Directory Structure

```text
Research_Project_Independent_Verification/
│
├── reports/
│   ├── complete_project_audit.md           <- Phase 1: Project inventory & architecture audit
│   ├── methodology_verification.md        <- Phase 2: Mathematical formulations & CV audit
│   ├── dataset_verification.md            <- Phase 3 & 7: Corpus integrity & dataset audit
│   ├── code_verification.md               <- Phase 4: Code, tokenizer, & regex unit audit
│   ├── experimental_results_audit.md      <- Phase 5: Independent metrics recalculation
│   ├── research_claims_audit.md           <- Phase 6: Claim-by-claim scientific audit
│   └── final_verification_report.md       <- Phase 10: Final comprehensive audit report
│
├── datasets/                              <- Verified benchmark dataset pointers & schemas
│
├── scripts/
│   ├── verify_datasets.py                 <- Automated dataset integrity verification script
│   ├── verify_metrics.py                  <- Independent metric & statistical recalculation script
│   └── reproducibility_checks.py          <- Pipeline, tokenizer, and smoke test check script
│
├── notebooks/
│   └── Project_Independent_Verification_Colab.ipynb <- Standalone Google Colab verification notebook
│
├── evidence/
│   ├── dataset_verification.log           <- Execution output of dataset audit
│   ├── metric_verification.log            <- Execution output of independent metric recomputation
│   ├── reproducibility_verification.log   <- Execution output of reproducibility checks
│   └── verified_metrics_recalculated.csv  <- Recalculated metrics from raw test predictions
│
└── README.md
```

## How to Re-Run Independent Verification

```bash
# 1. Verify datasets
python scripts/verify_datasets.py

# 2. Independently recalculate all metrics from raw test predictions
python scripts/verify_metrics.py

# 3. Verify tokenizer, lexicons, and pipeline determinism
python scripts/reproducibility_checks.py
```
