# Phase 4: Label Mapping & Medical Safety Verification

## 1. Triage Band Definitions
CLINIGUARD converts model probability $P(\text{hallucination})$ into three clinical triage tiers:
* **GREEN (Automated Pass):** $P(\text{hallucination}) < 0.20$ (or $< 0.35$ in legacy rule-based system).
* **AMBER (Manual Clinician Review):** $0.20 \le P(\text{hallucination}) \le 0.70$.
* **RED (Hard Rejection):** $P(\text{hallucination}) > 0.70$.

## 2. Critical Safety Audit: False-Reassurance Rate (FRR)
* **Med-HALT Test ($N=1,453$):** LightGBM GREEN coverage = 40.19%, FRR = **2.40%** (14 missed hallucinations). Appears safe on structured abstracts.
* **MedHallu Test ($N=622$):** LightGBM GREEN coverage = 17.52%, FRR = **24.77%** (27 missed hallucinations).
* **Logistic Regression on MedHallu:** GREEN coverage = 65.43%, FRR = **40.05%** (163 missed hallucinations).
* **Clinical Safety Verdict:** Labeling an answer "GREEN / SAFE" based on lexical density is dangerous in clinical practice. In realistic clinical QA, surface models miss 25% to 40% of hallucinations.
