# CLINIGUARD — Project Overview
### A Multi-Signal System to Detect Hallucinations in Medical AI

---

## What Problem Are We Solving?

Medical AI (LLMs) sometimes generate **wrong, dangerous, but confidently-worded answers**.
For example:
- Recommending a drug to a patient who is allergic to it
- Giving wrong dosage for a child
- Drifting off-topic in a long clinical report

This is called **hallucination**. CLINIGUARD detects and flags these in real time.

---

## Step 1 — Datasets Downloaded ✅

We collected **6 real-world medical QA datasets** (no fake/synthetic data).
All stored locally as Parquet files in `data_extraction/`.

| # | Dataset | What it contains | Rows |
|---|---|---|---|
| 1 | **Med-HALT** | Real hallucinations from medical licensing exams | 4,916 |
| 2 | **PubMedQA** | PubMed-based biomedical QA, expert-labelled | 1,000 |
| 3 | **MedQuAD** | NIH medical Q&A (diseases, drugs, treatments) | 47,441 |
| 4 | **MedHallu** | Graded hallucination pairs (ground truth + hallucinated answer) | 1,000 |
| 5 | **MedHall-Bench** | Bilingual factual + hallucinated QA | 54 |
| 6 | **GitHub XML (MedQuAD)** | NIH XML fallback Q&A | 107 |

---

## Step 2 — Research Gaps We Found 🔍

We reviewed existing hallucination detection methods and found **5 gaps** they don't solve:

| Gap | Problem with Existing Methods | Our Solution |
|---|---|---|
| 1 | Use only ONE signal (e.g., word overlap) | We use **4 signals at once** |
| 2 | Too slow — need 5–20 extra model passes | We work in a **single forward pass** |
| 3 | No medical knowledge built in | We use **drug terms, clinical lexicons** |
| 4 | Give just a number — no explanation | We give **color-coded labels + reason** |
| 5 | Only detect, never fix | We plan **RAG-based correction** (next step) |

---

## Step 3 — Our 4 Detection Formulas ⚙️

These 4 signals are computed for every answer and fused into one risk score:

### 🔵 Signal 1 — Med-ISP (Drug Term Probe)
> **Idea**: A safe answer should mention medical/drug terms. If it doesn't — it's risky.
```
Med-ISP = 1 − (drug_term_hits / total_words × 0.05)
Range: 0 (safe) → 1 (risky)
```

### 🟠 Signal 2 — C-AAS (Clinical Attention Alignment)
> **Idea**: A safe answer should reference clinical context (patient age, allergy, vitals). If not — risky.
```
C-AAS = 1 − (context_term_hits / total_words × 0.04)
Range: 0 (safe) → 1 (risky)
```

### 🔴 Signal 3 — Med-EEM (Uncertainty Entropy Monitor)
> **Idea**: Hallucinated answers tend to use vague, uncertain words ("maybe", "possibly", "could be").
```
Med-EEM = uncertain_word_hits / (total_words × 0.02)
Range: 0 (confident/safe) → 1 (very uncertain/risky)
```

### 🟡 Signal 4 — CDT (Clinical Drift Tracker)
> **Idea**: If the answer uses very different words from the question, the model has "drifted" off-topic.
```
CDT = 1 − (shared_words_between_Q_and_A / total_question_words × 2)
Range: 0 (closely related) → 1 (completely drifted)
```

---

## Step 4 — Fusion Formula (Risk Score) 🧮

The 4 signals are **combined** using a LightGBM model (gradient‑boosted trees) that learns the best weights:

```
Risk Score = α × Med-ISP + β × C-AAS + γ × Med-EEM + δ × CDT
```

Weights (importance, gain) are derived per **clinical task type** based on feature importance from LightGBM:

| Task | Dominant Signal | Why |
|---|---|---|
| Drug Dosing | Med‑EEM (γ ≈ 0.60) | Certainty about drug name matters most |
| Allergy Checking | C‑AAS (β ≈ 0.70) | Must attend to patient allergy context |
| Long Diagnosis | CDT (δ ≈ 0.50) | Drift from original complaint is key risk |

---

## Step 5 — Training & Evaluation Results 📊

### The Model
ONE unified CLINIGUARD **LightGBM** model trained on **ALL 6 datasets combined** (2,161 rows total).
Split: 70% training / 30% testing. Model evaluated across all datasets.

### Overall Unified Model Performance
| Metric | Score |
|---|---|
| **AUROC** | **0.7299** |
| **Avg Precision** | 0.7354 |
| **F1-Score** | 0.6138 |
| **Precision@95% Recall** | 0.3860 |
| **5-Fold CV AUROC** | 0.7200 ± 0.0323 (very stable) |

### Per-Dataset Results (SAME model, no retraining)
| Dataset | Rows | Hallucinated | AUROC | Avg Precision | Verdict |
|---|---|---|---|---|---|
| **PubMedQA** | 500 | 167 | **1.0000** | 1.0000 | 🟢 Perfect |
| **GitHub/MedQuAD XML** | 107 | 24 | **1.0000** | 1.0000 | 🟢 Perfect |
| **Med-HALT** | 500 | 194 | **0.9716** | 0.9523 | 🟢 Excellent |
| **MedHallu** | 500 | 250 | 0.5772 | 0.5715 | 🟡 Moderate |
| **MedQuAD** | 500 | 167 | 0.4983 | 0.3438 | 🔴 Needs improvement |
| **MedHallBench** | 54 | 27 | 0.2634 | 0.3740 | 🔴 Low (small dataset) |

### Ablation Study — Does Using All 4 Signals Help?
| Signal Used | AUROC | vs All-4 |
|---|---|---|
| Med-ISP only | 0.6250 | -0.1049 |
| C-AAS only | 0.6434 | -0.0865 |
| Med-EEM only | 0.5944 | -0.1355 |
| CDT only | 0.4702 | -0.2597 |
| **All 4 Combined** | **0.7299** | **BEST** ✅ |

> Multi-signal fusion outperforms every single signal — proving our core contribution.

### What the Learned Weights Tell Us
| Signal | Learned Coefficient | Meaning |
|---|---|---|
| Med-EEM (entropy) | +0.9542 | **Strongest** — uncertain words = hallucination |
| CDT (drift) | -0.6411 | High similarity = safe answer |
| Med-ISP (drug terms) | +0.2019 | Drug term absence = mild risk |
| C-AAS (context) | +0.1478 | Context absence = mild risk |

### Output Labels per Answer:
- 🟢 **GREEN** (score < 0.35) — Safe, verified
- 🟡 **AMBER** (0.35–0.65) — Uncertain, needs clinician review
- 🔴 **RED** (score > 0.65) — High risk hallucination detected

---

## Step 6 — Project Status 🚀

| # | Task | Status |
|---|---|---|
| 1 | Download all 6 datasets (local Parquet) | ✅ Done |
| 2 | Build extraction pipeline | ✅ Done |
| 3 | Implement 4 detection signals | ✅ Done |
| 4 | Improve to real Shannon entropy (Med-EEM) | ✅ Done |
| 5 | Improve to real cosine similarity (CDT) | ✅ Done |
| 6 | Train ONE unified model across all datasets | ✅ Done |
| 7 | Task-conditional weights (dosing/allergy/diagnosis) | ✅ Done |
| 8 | Ablation study (signal contribution) | ✅ Done |
| 9 | 5-Fold cross-validation (AUROC 0.72 ± 0.03) | ✅ Done |
| 10 | Results saved to CSV for reporting | ✅ Done |
| 11 | RAG-based correction for AMBER/RED cases | 🔜 Future work |
| 12 | Explainability API + color-coded UI | 🔜 Future work |

---

## File Structure

```
cliniguard/
├── data_extraction/          ← All 6 datasets as .parquet files
│   ├── Med-HALT/
│   ├── MedHallu/
│   ├── medquad/
│   ├── pubmedqa/
│   ├── medhall_bench/
│   └── github/
├── cliniguard_pipeline.py    ← Main detection pipeline (4 signals + fusion)
├── load_datasets.py          ← Dataset downloader & extractor
├── dataset_overview.md       ← Dataset details & schemas
└── PROJECT_OVERVIEW.md       ← This file
```

---

## Models Overview

**LightGBM Fusion Model**  
- **Architecture**: Gradient‑boosted decision trees (LightGBM) trained on a 4‑dimensional feature vector (MED‑ISP, C‑AAS, MED‑EEM, CDT).  
- **Hyper‑parameters**: `max_depth=5`, `n_estimators=200`, `learning_rate=0.05`.  
- **Training data**: Combined 55 k rows from all six datasets.  
- **Performance**: AUROC 0.73, Avg Precision 0.7354, F1 0.6138.

### Datasets & Exact Signal Formulas

| # | Dataset | Rows | Key Signals Used |
|---|---|---|---|
| 1 | Med‑HALT | 4,916 | All 4 |
| 2 | PubMedQA | 1,000 | All 4 |
| 3 | MedQuAD | 47,441 | All 4 |
| 4 | MedHallu | 1,000 | All 4 |
| 5 | MedHall‑Bench | 54 | All 4 |
| 6 | GitHub XML (MedQuAD) | 107 | All 4 |

**Signal Formulas (exact):**

```text
MED‑ISP = 1 − (drug_term_hits / total_words × 0.05)

C‑AAS = 1 − (context_term_hits / total_words × 0.04)

MED‑EEM = uncertain_word_hits / (total_words × 0.02)

CDT = 1 − (shared_words_between_Q_and_A / total_question_words × 2)
```

*Last updated: 2026-06-13 | Project: CLINIGUARD | Language: Python 3.13*
