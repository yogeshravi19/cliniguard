# Phase 3 & 7: Dataset Verification & Corpus Integrity Audit

## 1. Verified Genuine Datasets

### A. Med-HALT (IR_abstract2pubmedlink)
* **Source:** HuggingFace `openlifescienceai/Med-HALT` (CoNLL 2023)
* **License:** CC BY-SA 4.0
* **Storage Location:** `data_extraction/Med-HALT/medhalt.parquet`
* **Total Records:** 4,914
* **Target Labels:** Binary ($0 = \text{real PubMed abstract}, 1 = \text{fake LLM abstract}$)
* **Class Distribution:** 2,925 Factual (59.52%), 1,989 Hallucinated (40.48%)
* **Missing Values:** 0 nulls across question, context, answer, label.
* **Text Characteristics:** Real abstracts mean $208.0 \pm 77.4$ words; Fake abstracts mean $127.3 \pm 17.7$ words.

### B. MedHallu (pqa_labeled)
* **Source:** HuggingFace `UTAustin-AIHealth/MedHallu` (2025)
* **License:** Apache 2.0
* **Storage Location:** `data_extraction/MedHallu/medhallu.parquet`
* **Total Records:** 2,000 (1,000 paired clinical questions)
* **Target Labels:** Binary ($0 = \text{Human Doctor Ground Truth}, 1 = \text{LLM Hallucinated Answer}$)
* **Class Distribution:** 1,000 Factual (50.00%), 1,000 Hallucinated (50.00%)
* **Missing Values:** 0 nulls across question, context, answer, label.
* **Text Characteristics:** Ground truth answers mean $39.7 \pm 17.2$ words; Hallucinated answers mean $27.8 \pm 15.1$ words.

## 2. Excluded / Unsuitable Datasets Audit
* **PubMedQA (`qiaojin/PubMedQA`):** EXCLUDED. Authentic dataset contains only factual PubMed answers. Legacy scripts corrupted labels by injecting a hardcoded hedging sentence into every third answer.
* **MedQuAD (`lavita/MedQuAD`):** EXCLUDED. Authentic dataset contains verified NIH consumer health Q&A without negative hallucination samples. Legacy scripts assigned arbitrary labels based on `index % 3 == 0`.
* **MedQuAD XML & MedHall-Bench:** EXCLUDED. Same synthetic tampering or sample size ($N=54$) too small for statistical power.

## 3. Evaluation of the 10-Dataset Requirement
* **Finding:** Expanding to 10 datasets by including general medical QA benchmarks (e.g. MedQA, MedMCQA, MMLU Medical) without authentic negative hallucination labels requires artificial corruptions (sentence replacement or arbitrary label flipping).
* **Scientific Recommendation:** Restricting evaluation to the two authentic, peer-reviewed medical hallucination benchmarks (Med-HALT and MedHallu, $N=6,914$) preserves scientific validity and reproducibility.
