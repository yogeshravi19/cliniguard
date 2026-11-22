# CliniGuard: Dataset Source & Reproducibility Audit Suite

This directory contains the independent audit, source verification, and reproducibility infrastructure for the **CliniGuard** research project.

---

## Directory Overview

```text
Dataset_Source_and_Reproducibility_Audit/
│
├── datasets/                                 # Cached validation samples for external medical benchmarks
│   ├── medqa_sample.parquet                  # MedQA-USMLE (100 exam questions)
│   ├── medmcqa_sample.parquet                # MedMCQA (100 multi-specialty questions)
│   ├── mmlu_clinical_sample.parquet          # MMLU Clinical Knowledge (100 questions)
│   └── truthfulqa_health_sample.parquet      # TruthfulQA Health/Medical (55 questions)
│
├── notebooks/
│   └── Dataset_Verification_Colab.ipynb      # Standalone Google Colab replication notebook
│
├── scripts/
│   ├── verify_existing_datasets.py           # Audits local datasets in data_extraction/
│   └── download_missing_datasets.py          # Verifies & caches external benchmark suites
│
├── reports/
│   ├── full_project_audit.md                 # Detailed forensic audit of project codebase
│   ├── existing_dataset_inventory.csv        # Local dataset row counts, schemas, and hashes
│   ├── verified_dataset_sources.csv          # Authoritative sources for all 10 medical datasets
│   ├── dataset_metadata.json                 # Machine-readable dataset specifications
│   └── reproducibility_report.md             # Traceability of empirical metrics & findings
│
├── requirements.txt                          # Pip dependency specifications
└── README.md                                 # This file
```

---

## Quick Start: Running the Audit

### 1. Local Python Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Step 1: Verify all 6 local datasets in data_extraction/
python scripts/verify_existing_datasets.py

# Step 2: Verify and fetch external medical benchmarks (10-dataset landscape)
python scripts/download_missing_datasets.py
```

### 2. Google Colab Replication
Open `notebooks/Dataset_Verification_Colab.ipynb` directly in Google Colab. It automatically handles environment setup, detects data paths, verifies the 10 datasets, and replicates CliniGuard's four-signal scoring pipeline interactively.
