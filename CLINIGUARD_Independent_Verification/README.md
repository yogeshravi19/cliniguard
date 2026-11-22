# CLINIGUARD Independent Verification Workspace

This workspace provides independent verification scripts, evidence logs, dataset catalogs, audit reports, and a self-contained Google Colab notebook for the **CLINIGUARD** medical hallucination detection project.

## Workspace Layout

```text
CLINIGUARD_Independent_Verification/
├── reports/
│   ├── complete_project_audit.md       <- Phase 1: Directory, script & artifact inventory
│   ├── methodology_audit.md            <- Phase 2 & 6: Research methodology & CV rigor
│   ├── signal_verification.md          <- Phase 2: Mathematical audit of the 4 signals
│   ├── dataset_source_audit.md         <- Phase 3 & 8: Benchmark & 10 dataset analysis
│   ├── label_mapping_audit.md          <- Phase 4: Medical safety categories & FRR audit
│   ├── model_comparison_audit.md       <- Phase 5: Independent recomputation of metrics
│   ├── deployment_verification.md      <- Phase 11: server.py, FastAPI & model artifacts
│   └── final_verification_report.md    <- Phase 12: Final comprehensive audit report
├── scripts/
│   ├── verify_datasets.py              <- Verifies extraction parquet files & catalog
│   ├── verify_features.py              <- Verifies 4 signal implementations & fallbacks
│   ├── verify_label_mapping.py         <- Audits GREEN/AMBER/RED triage logic
│   ├── verify_metrics.py               <- Recomputes test set metrics from raw predictions
│   └── verify_model_artifacts.py       <- Audits saved model & scaler joblib artifacts
├── evidence/
│   ├── execution_logs/                 <- Console outputs from verification script executions
│   └── metric_comparisons/             <- Recalculated CSV metrics files
├── notebooks/
│   └── CLINIGUARD_Verification_Colab.ipynb <- Executable Google Colab verification notebook
├── dataset_sources.csv                 <- Curated 10 medical datasets catalog
├── requirements.txt                    <- Python dependency specification
└── README.md
```

## Running the Verification Suite

```bash
# Execute all verification scripts
python scripts/verify_datasets.py
python scripts/verify_features.py
python scripts/verify_label_mapping.py
python scripts/verify_metrics.py
python scripts/verify_model_artifacts.py
```
