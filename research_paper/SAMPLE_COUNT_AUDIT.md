# CliniGuard Forensic Sample-Count & Overlap Audit

**Date:** October 2026  
**Status:** Complete Code-Level Audit  
**Execution Context:** Audited directly from `data_extraction/` local parquet files and upstream Hugging Face / GitHub sources.

---

## 1. Executive Summary Table

| Dataset | Raw Records | Parsed Records | Unique Questions | Question Duplicates | Excluded Records | Final Verified Used | Evaluation / Label Type |
|---|---:|---:|---:|---:|---:|---:|---|
| **Med-HALT** | 4,916 | 4,916 | 4,817 | 97 | 2 (empty text) | **4,914** | Authentic binary labels (`real_data` vs. `fake_data`) |
| **MedHallu** | 2,000 | 2,000 | 1,000 | 1,000 | 0 | **2,000** | Authentic binary labels (Ground Truth vs. LLM Hallucination pairs) |
| **PubMedQA** | 1,000 | 1,000 | 1,000 | 0 | 1,000 (redundant & synthetic) | **0** (Primary) | Excluded: 100% question overlap with MedHallu + synthetic injection |
| **MedQuAD** | 47,441 | 47,441 | 43,449 | 2,838 | 47,441 (no hallucination labels) | **0** (Primary) | Excluded: Authentic QA with no annotated hallucinations |
| **MedQuAD XML** | 107 | 107 | 75 | 32 | 107 (redundant & synthetic) | **0** (Primary) | Excluded: 100% duplicate subset of MedQuAD + synthetic injection |
| **MedHall-Bench** | 54 | 54 | 27 | 27 | 54 (no model outputs) | **0** (Primary) | Excluded: Rubric test harness without pre-packaged LLM text |
| **Verified Genuine Benchmark** | **6,916** | **6,916** | **5,817** | **1,097** | **2** | **6,914** | **100% Authentic Peer-Reviewed Clinical Ground Truth** |

---

## 2. In-Depth Per-Dataset Code Audit

### 2.1 Med-HALT
1. **Official / Source Dataset Name:** `openlifescienceai/Med-HALT` (Config: `IR_abstract2pubmedlink`).
2. **Local File / Path Used:** `data_extraction/Med-HALT/medhalt.parquet`.
3. **Number of Raw Records:** 4,916 rows in Hugging Face split.
4. **Number of Records after Parsing:** 4,916 rows.
5. **Number of Records after Preprocessing:** 4,914 rows.
6. **Number of Unique Questions / Prompts:** 4,817 unique paper titles (`Title`).
7. **Number of Duplicate Records:** 1 exact duplicate row; 97 titles appear more than once with differing abstracts.
8. **Number of Records Excluded and Why:** 2 records excluded due to empty or malformed abstract text (<10 characters).
9. **Number Actually Used in Experiment:** **4,914** (2,925 label 0 [`real_data`] + 1,989 label 1 [`fake_data`]).
10. **Unit of Count:** Individual question–answer instances (PubMed paper title + abstract).
11. **Label / Evaluation Type:** Authentic binary labels from a peer-reviewed medical LLM hallucination benchmark (CoNLL 2023).

---

### 2.2 MedHallu
1. **Official / Source Dataset Name:** `UTAustin-AIHealth/MedHallu` (Config: `pqa_labeled`).
2. **Local File / Path Used:** `data_extraction/MedHallu/medhallu.parquet`.
3. **Number of Raw Records:** 1,000 raw rows in Hugging Face split (each row contains 1 Ground Truth and 1 Hallucinated Answer).
4. **Number of Records after Parsing:** 2,000 paired question–answer instances.
5. **Number of Records after Preprocessing:** 2,000 instances.
6. **Number of Unique Questions / Prompts:** 1,000 unique clinical examination questions.
7. **Number of Duplicate Records:** 0 exact duplicate rows; 1,000 question duplicates because each question appears twice (once as Ground Truth [label 0], once as Hallucinated Answer [label 1]).
8. **Number of Records Excluded and Why:** 0 records excluded.
9. **Number Actually Used in Experiment:** **2,000** instances (1,000 label 0 + 1,000 label 1).
10. **Unit of Count:** Individual question–answer instances (representing 1,000 question pairs).
11. **Label / Evaluation Type:** Authentic binary labels curated and verified by medical domain experts.

---

### 2.3 PubMedQA
1. **Official / Source Dataset Name:** `qiaojin/PubMedQA` (Config: `pqa_labeled`).
2. **Local File / Path Used:** `data_extraction/pubmedqa/pubmedqa.parquet`.
3. **Number of Raw Records:** 1,000 rows in Hugging Face split.
4. **Number of Records after Parsing:** 1,000 rows.
5. **Number of Records after Preprocessing:** 1,000 rows.
6. **Number of Unique Questions / Prompts:** 1,000 unique questions.
7. **Number of Duplicate Records:** 0 duplicate questions within PubMedQA.
8. **Number of Records Excluded and Why:**
   * **Reason 1 (Overlap):** **100% of the 1,000 questions in PubMedQA are identical to the 1,000 questions in MedHallu.** MedHallu was built directly upon PubMedQA's `pqa_labeled` split. Counting both PubMedQA and MedHallu constitutes 100% question double-counting.
   * **Reason 2 (Synthetic Contamination):** PubMedQA contains zero hallucination annotations. The legacy script manufactured "hallucinations" by replacing every 3rd answer with a single hardcoded sentence (`"We suggest that it could perhaps be possible..."`).
9. **Number Actually Used in Primary Experiment:** **0** (Excluded to prevent data leakage and artificial inflation).
10. **Unit of Count:** Question–answer pairs.
11. **Label / Evaluation Type:** Authentic biomedical QA (EMNLP 2019), but synthetic/invalid as a hallucination detector when using programmatic text injection.

---

### 2.4 MedQuAD
1. **Official / Source Dataset Name:** `lavita/MedQuAD` (NIH 12 Institute Collection by Ben Abacha & Demner-Fushman).
2. **Local File / Path Used:** `data_extraction/medquad/medquad.parquet`.
3. **Number of Raw Records:** 47,441 rows in Hugging Face dataset.
4. **Number of Records after Parsing:** 47,441 rows.
5. **Number of Records after Preprocessing:** 47,441 rows.
6. **Number of Unique Questions / Prompts:** 43,449 unique consumer health questions.
7. **Number of Duplicate Records:** 237 exact duplicate rows; 2,838 questions repeated across multiple NIH guideline documents.
8. **Number of Records Excluded and Why:**
   * MedQuAD contains 100% authentic, correct clinical guidance from the NIH. It has **zero annotated hallucinations**.
   * The legacy script arbitrarily assigned labels via `row_index % 3 == 0` without altering the authentic text, creating arbitrary/unlearnable labels.
   * The proposed "1,000" count was an arbitrary, unstratified cutoff of the 47,441 total records.
9. **Number Actually Used in Primary Experiment:** **0** (Excluded from primary hallucination evaluation).
10. **Unit of Count:** Question–answer pairs.
11. **Label / Evaluation Type:** Authentic clinical reference QA (BMC Bioinformatics 2019), not a hallucination benchmark.

---

### 2.5 MedQuAD XML (GitHub Fallback)
1. **Official / Source Dataset Name:** `abachaa/MedQuAD` (raw XML files from `5_NIDDK_QA/` directory on GitHub).
2. **Local File / Path Used:** `data_extraction/github/github.parquet`.
3. **Number of Raw Records:** 107 rows parsed from 20 XML files.
4. **Number of Records after Parsing:** 107 rows.
5. **Number of Records after Preprocessing:** 107 rows.
6. **Number of Unique Questions / Prompts:** 75 unique questions.
7. **Number of Duplicate Records:** 5 exact duplicate rows; 32 question repeats across XML sections.
8. **Number of Records Excluded and Why:**
   * **100% Overlap:** All 75 unique questions in MedQuAD XML are **already present** in `data_extraction/medquad/medquad.parquet`. Adding MedQuAD XML and MedQuAD represents pure double-counting.
   * Contaminated by the same synthetic sentence injection as PubMedQA.
   * The proposed "214" count was a completely fabricated doubling of the actual 107 rows ($107 \times 2 = 214$).
9. **Number Actually Used in Primary Experiment:** **0** (Excluded).
10. **Unit of Count:** Question–answer pairs.
11. **Label / Evaluation Type:** 100% duplicate subset of MedQuAD.

---

### 2.6 MedHall-Bench
1. **Official / Source Dataset Name:** `healthmemoryarena/MedHall-Bench` (MiroBody / HolyEval).
2. **Local File / Path Used:** `data_extraction/medhall_bench/medhall_bench.parquet`.
3. **Number of Raw Records:** 54 rows (27 factual + 27 contextual test cases).
4. **Number of Records after Parsing:** 54 rows.
5. **Number of Records after Preprocessing:** 54 rows.
6. **Number of Unique Questions / Prompts:** 27 unique patient clinical prompts (`strict_inputs`).
7. **Number of Duplicate Records:** 0 exact duplicate rows; 27 duplicate questions (each prompt appears once for label 0 and once for label 1).
8. **Number of Records Excluded and Why:**
   * MedHall-Bench is an evaluation rubric harness designed for evaluating external LLM APIs via HolyEval. It contains **no pre-packaged LLM response texts**.
   * The legacy script took `known_facts` as answer with label 0, and hardcoded an identical Chinese sentence as label 1 across all 27 cases.
   * The proposed "108" count was a completely fabricated doubling of the actual 54 rows ($54 \times 2 = 108$).
9. **Number Actually Used in Primary Experiment:** **0** (Excluded).
10. **Unit of Count:** Benchmark test cases / prompts.
11. **Label / Evaluation Type:** Agent evaluation rubric, contaminated in legacy code by hardcoded text injection.

---

## 3. Cross-Dataset Overlap Matrix

The code-level cross-dataset question overlap matrix reveals complete redundancies across candidate datasets:

```
               Med-HALT   MedHallu   PubMedQA    MedQuAD   MedQuAD_XML   MedHall-Bench
Med-HALT          4,817          0          0          0             0               0
MedHallu              0      1,000      1,000          0             0               0  <-- 100% overlap with PubMedQA
PubMedQA              0      1,000      1,000          0             0               0  <-- 100% overlap with MedHallu
MedQuAD               0          0          0     43,449            75               0  <-- 100% of XML is inside MedQuAD
MedQuAD_XML           0          0          0         75            75               0  <-- 100% overlap with MedQuAD
MedHall-Bench         0          0          0          0             0              27
```

### Key Overlap Findings:
1. **MedHallu vs. PubMedQA:** Exactly **1,000 out of 1,000 questions (100.0%)** overlap. MedHallu took the 1,000 questions from PubMedQA `pqa_labeled`, kept the PubMedQA answer as Ground Truth, and added LLM-generated hallucinations. Summing MedHallu + PubMedQA counts the same 1,000 questions twice.
2. **MedQuAD vs. MedQuAD XML:** Exactly **75 out of 75 questions (100.0%)** in MedQuAD XML overlap with MedQuAD. MedQuAD XML is simply a tiny 20-file slice of MedQuAD. Summing them counts the same NIH questions twice.
3. **Med-HALT vs. All Others:** Zero overlap (0 questions). Med-HALT is completely independent.
4. **MedHall-Bench vs. All Others:** Zero overlap (0 questions). MedHall-Bench is completely independent.

---

## 4. Synthesis of Counts

### A. Total Genuine Benchmark Instances (Primary Ground Truth)
* **Med-HALT:** 4,914 instances
* **MedHallu:** 2,000 instances
* **Verified Genuine Benchmark Total:** **6,914 instances** (100% non-overlapping, authentic medical hallucination test instances).

### B. Total Unique Questions / Prompts
* Med-HALT unique titles: 4,817
* MedHallu unique questions: 1,000
* MedQuAD unique questions: 43,449
* MedHall-Bench unique prompts: 27
* **Total Unique Questions Across Entire Project Repository:** **49,293 unique clinical questions**.
* **Total Unique Questions Across Verified Genuine Benchmark:** **5,817 unique clinical questions** ($4,817 + 1,000$).

### C. Total Raw Records in Local Parquet Files
* Med-HALT: 4,914
* MedHallu: 2,000
* PubMedQA: 1,000
* MedQuAD: 47,441
* MedQuAD XML: 107
* MedHall-Bench: 54
* **Total Local Raw Parquet Rows:** **55,516 rows**.

---

## 5. Final Scientific Decision

### Is 9,236 valid? **NO.**

### Exactly Why 9,236 Is Scientifically Invalid:
1. **The Math Was Constructed From Arbitrary Doublings and Subsets:**
   * Claimed count: $4,914 + 2,000 + 1,000 + 1,000 + 214 + 108 = 9,236$.
   * **MedQuAD XML is 107 rows, NOT 214.** Someone arbitrarily multiplied 107 by 2.
   * **MedHall-Bench is 54 rows, NOT 108.** Someone arbitrarily multiplied 54 by 2.
   * **MedQuAD is 47,441 rows, NOT 1,000.** Claiming 1,000 was an arbitrary, undocumented slice.
2. **Severe Double-Counting of Identical Questions:**
   * Adding MedHallu (2,000) and PubMedQA (1,000) double-counts the **exact same 1,000 questions** because MedHallu is directly derived from PubMedQA.
   * Adding MedQuAD (1,000) and MedQuAD XML (214) double-counts the **exact same NIH questions** because MedQuAD XML is a 20-file subset of MedQuAD.
3. **Incompatible Evaluation Semantics:**
   * Med-HALT and MedHallu contain authentic hallucination labels.
   * PubMedQA, MedQuAD, MedQuAD XML, and MedHall-Bench contained synthetic sentence injections or arbitrary index labels in the legacy code. Adding them together conflates genuine hallucination testing with synthetic artifacts.

### The Scientifically Correct Headline Number:
* **The verified, scientifically defensible sample count for the primary hallucination detection evaluation is exactly 6,914 instances** (representing **5,817 unique clinical questions** across **Med-HALT [4,914]** and **MedHallu [2,000]**).
* All project documentation, tables, and manuscript sections must report **6,914 verified genuine instances** and formally document the audit in this document.
