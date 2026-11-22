# Phase 6: Research Claims & Scientific Validity Audit

## 1. Claim-by-Claim Evidence Check

| Claim in Documentation | Initial Framing | Audited Empirical Evidence | Scientific Verdict |
| :--- | :--- | :--- | :---: |
| "Four distinct domain features provide robust hallucination detection." | Core proposed model | Ablation and factorial baselines show that `CDT + Length` alone achieves **0.9572 AUROC**, while adding the other 3 features yields only **0.9593 AUROC** (+0.0021 lift). | **REFUTED AS NOVEL MULTI-SIGNAL DETECTOR; CONFIRMED AS CDT+LENGTH PROBE** |
| "Med-EEM provides an entropy-based monitor of clinical uncertainty." | Information-theoretic signal | Hedging word fractions ($< 0.05$) reside entirely on the monotonic branch of Shannon entropy. LightGBM treats it identically to linear hedging rate. | **RE-FRAMED AS HEDGING FREQUENCY PROBE** |
| "TF-IDF achieves 0.9932 AUROC across medical benchmarks." | Surface-level baseline | Med-HALT contains deterministic synthetic generation markers (`fictional`, `quantum`, `Background:` tags) that allow TF-IDF to reach **1.0000 AUROC**. | **CONFIRMED; EXPOSES BENCHMARK SHORTCUT LEARNING** |
| "Models generalize reliably across clinical hallucination datasets." | Generalizability claim | Cross-dataset transfer from Med-HALT to MedHallu collapses to **0.5518 AUROC** (near random chance). MedHallu to Med-HALT TF-IDF drops to **0.3973 AUROC** due to polarity inversion. | **REFUTED; SUPPORTS NEGATIVE TRANSFER AUDIT** |
| "The GREEN risk band safely automates clinician review." | Clinical safety triage | While safe on Med-HALT (FRR = 2.40%), on MedHallu LightGBM allows **24.77% to 31.20%** of the green band to be hallucinated. Logistic Regression allows **40.05% to 52.75%** missed hallucinations. | **CONFIRMED AS CRITICAL SAFETY WARNING** |

## 2. Scientific Contribution Realignment
The project is not viable as a new state-of-the-art closed-API screening detector because TF-IDF and simple length heuristics outperform it. However, it represents an outstanding **benchmark-validity and shortcut-learning audit paper** in clinical NLP, directly paralleling hypothesis-only audits in natural language inference (e.g. Gururangan et al. 2018, Poliak et al. 2018).
