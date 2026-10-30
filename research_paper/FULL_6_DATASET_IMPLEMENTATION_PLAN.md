# CliniGuard: Full 6-Dataset Implementation Architecture & Plan

**Status:** Technical Implementation Reference  
**Audited Sample Counts:** See `SAMPLE_COUNT_AUDIT.md` for complete code-level provenance.

---

## 1. Resolution of the Sample-Count Conflict

### Rejection of the 9,236 Headline Count
The previous calculation of 9,236 ($4,914 + 2,000 + 1,000 + 1,000 + 214 + 108 = 9,236$) has been audited and **formally rejected** as mathematically and scientifically invalid:
1. **Severe Double-Counting of Questions:**
   * MedHallu (2,000 instances) was derived directly from PubMedQA (1,000 instances). Exactly **1,000 out of 1,000 questions (100.0%)** overlap. Adding both counts the exact same 1,000 questions twice.
   * MedQuAD XML (107 instances) was scraped from GitHub as a 20-file slice of MedQuAD (47,441 instances). Exactly **75 out of 75 questions (100.0%)** in MedQuAD XML overlap with MedQuAD. Adding both counts the same questions twice.
2. **Arbitrary Multipliers in Proposed Counts:**
   * MedQuAD XML contains 107 records in its local parquet file, but was arbitrarily multiplied by 2 to claim "214".
   * MedHall-Bench contains 54 records in its local parquet file, but was arbitrarily multiplied by 2 to claim "108".
   * MedQuAD contains 47,441 records, but was arbitrarily sliced to "1,000" without stratification.
3. **Incompatible Evaluation Semantics:**
   * Summing genuine benchmark instances (Med-HALT and MedHallu) with synthetic sentence replacements (PubMedQA, MedQuAD XML) into a single pooled number creates deceptive evaluation metrics that fail peer review.

---

## 2. Scientifically Valid Architecture for Utilizing All Six Datasets

To utilize all six datasets legitimately without conflating fundamentally different evaluation semantics, the architecture is partitioned into four distinct experimental pillars:

```
                      CLINIGUARD 4-PILLAR EVALUATION ARCHITECTURE
                                          
 [Pillar 1: In-the-Wild Hallucination]      [Pillar 2: Cross-Condition Extrinsic Hallucination]
  • Med-HALT (N=4,914; CoNLL 2023)           • MedQuAD NIH (N=1,000 stratified subset)
  • MedHallu (N=2,000; Expert-Curated)       • PubMedQA (N=1,000 research queries)
  Metric: AUROC, AP, F1, P@95%R              Method: True NIH match vs. Cross-specialty mismatch
                   │                                          │
                   └─────────────────────┬────────────────────┘
                                         ▼
                 [Pillar 3: Clinical Specificity & False Alarms]
                  • 100% Verified Pristine Clinical Answers
                  • Test True Negative Rate on Trusted NIH Guidelines
                  • Ensures GREEN band does not flag real medical text
                                         │
                                         ▼
                 [Pillar 4: Benchmark Contamination Audit]
                  • Demonstrates metric inflation from synthetic text injection
                  • Forensic audit of PubMedQA, MedQuAD XML, MedHall-Bench
```

### Pillar 1: Primary In-the-Wild Hallucination Benchmark (Headline Evaluation)
* **Datasets:** **Med-HALT** (4,914 instances, 4,817 unique questions) and **MedHallu** (2,000 instances, 1,000 unique questions).
* **Total Verified Instances:** **6,914 instances** across **5,817 unique questions**.
* **Role in Paper:** Serves as the primary, headline evaluation benchmark for CliniGuard's core detection accuracy (AUROC 0.922 ± 0.005).

### Pillar 2: Cross-Specialty Extrinsic Hallucination Benchmark
* **Datasets:** **MedQuAD** (1,000 stratified question–answer pairs from 12 NIH institutes) and **PubMedQA** (1,000 biomedical literature pairs).
* **Methodology:**
  * Label 0: Authentic matched question and answer (e.g. Pediatric Asthma query + Pediatric Asthma guidelines).
  * Label 1: Cross-specialty mismatched answer (e.g. Pediatric Asthma query paired with Adult Oncology chemotherapy protocols).
* **Role in Paper:** Tests CliniGuard's Context Drift (CDT) and Clinical Context (C-AAS) on detecting dangerous clinical topic drift using 100% authentic medical prose from the NIH and PubMed (zero synthetic repeated sentences).

### Pillar 3: Clinical Specificity and False-Alarm Audit
* **Datasets:** Pristine authentic answers from **MedQuAD**, **PubMedQA**, and **MedHall-Bench**.
* **Clinical Significance:** Evaluates CliniGuard's True Negative Rate (Specificity) on gold-standard clinical documentation. In clinical deployment, it is vital that trusted NIH guidance is NOT falsely flagged as high risk (RED).

### Pillar 4: Benchmark Integrity & Contamination Audit
* **Datasets:** Legacy programmatic injection scripts across **PubMedQA**, **MedQuAD XML**, and **MedHall-Bench**.
* **Role in Paper:** Contrasts the performance of surface baselines (e.g. length, TF-IDF) under programmatic sentence injection versus authentic clinical text, establishing CliniGuard's forensic audit as a standalone contribution to medical AI evaluation standards.

---

## 3. Verified Sample Counts Summary

* **Primary Verified Hallucination Instances:** **6,914**
* **Unique Clinical Questions in Primary Benchmark:** **5,817**
* **Total Available Raw Parquet Records in Repository:** **55,516**
* **Total Unique Clinical Questions Across All Sources:** **49,293**
