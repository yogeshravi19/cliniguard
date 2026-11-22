# Phase 3 & 8: Full Dataset Analysis & 10 Curated Medical Datasets

## 1. Verified Genuine Benchmark Datasets in Repository

| Dataset | Official Source / Publication | Local File | Usable Rows | Target Labels | Authentic Annotation |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Med-HALT** | `openlifescienceai/Med-HALT` (CoNLL 2023) | `data_extraction/Med-HALT/medhalt.parquet` | 4,914 | 0: Real PubMed (2,925)<br>1: Fake Abstract (1,989) | **Yes** (Reasoning & Hallucination) |
| **MedHallu** | `UTAustin-AIHealth/MedHallu` (2025) | `data_extraction/MedHallu/medhallu.parquet` | 2,000 | 0: Ground Truth (1,000)<br>1: LLM Hallucinated (1,000) | **Yes** (Expert Physician Annotation) |

## 2. Excluded / Unsuitable Sources Audit
* **PubMedQA:** $N=1,000$. Authentic dataset contains only factual PubMed answers. Legacy scripts injected a synthetic sentence every 3rd row.
* **MedQuAD:** $N=47,441$. Authentic dataset has no hallucination pairs. Legacy scripts assigned arbitrary labels via `index % 3 == 0`.
* **cliniguard_all_datasets.csv:** $N=55,518$. Contaminated with the above synthetic injection. Properly excluded from rigorous evaluation.

## 3. Curated 10 Medical Datasets Catalog (dataset_sources.csv)
1. **Med-HALT:** Primary genuine benchmark (Local, Verified).
2. **MedHallu:** Primary clinical QA benchmark (Local, Verified).
3. **MedQuAD:** NIH consumer QA (Local, Verified Task-Restricted).
4. **PubMedQA:** Biomedical QA (Local, Excluded from primary pool).
5. **MedQuAD GitHub XML:** Duplicate NIH QA (Local, Excluded).
6. **MedHall-Bench:** Evaluation rubrics ($N=54$, Local, Excluded).
7. **MedQA-USMLE:** USMLE 4-option board exam questions (Sample cached).
8. **MedMCQA:** Indian medical entrance exam questions (Sample cached).
9. **MMLU Clinical Knowledge & Professional Medicine:** Medical capability probing (Sample cached).
10. **TruthfulQA (Health/Medicine):** Health misconception & falsehood evaluation (Sample cached).
