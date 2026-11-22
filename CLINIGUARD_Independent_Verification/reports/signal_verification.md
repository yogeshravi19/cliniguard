# Phase 2: Signal Implementation & Mathematical Verification

## 1. Mathematical Analysis of the Four Signals

### A. Med-ISP (Medical Information Safety Proxy)
* **Implementation:** `1.0 - min(hits / max(len(words)*0.05, 1), 1.0)`
* **Lexicon:** 83 RxNorm pharmaceutical agents and attached dosage units (`500mg`, `10mcg`).
* **Behavior:** Operates as a *penalty score*. Dense medication text yields 0.0 (low risk / factual); sparse text yields 1.0 (high risk).
* **Empty Input:** Yields 1.0 safely.

### B. C-AAS (Clinical Assertion Accuracy Score)
* **Implementation:** `1.0 - min(hits / max(len(words)*0.04, 1), 1.0)`
* **Lexicon:** 64 clinical context terms (e.g. `contraindication`, `symptoms`, `diagnosis`).
* **Scientific Note:** C-AAS measures *clinical vocabulary density*, not true factual accuracy. A completely fabricated clinical statement full of medical jargon scores 0.0 (falsely labeled low risk).

### C. Med-EEM (Medical Epistemic Entropy Measure)
* **Implementation:** $H(p) = -p \log_2 p - (1-p) \log_2 (1-p)$ where $p = \text{hits} / n$.
* **Lexicon:** 50 epistemic uncertainty words (e.g. `suggest`, `possibly`, `uncertain`).
* **Scientific Note:** Because clinical hedging density $p < 0.05$, $H(p)$ lies entirely on the monotonically increasing branch ($p < 0.5$). Tree models like LightGBM are invariant to monotonic scaling; Med-EEM operates as a scaled hedging frequency counter.

### D. CDT (Contextual Drift Threshold)
* **Implementation:** $1.0 - \text{cosine\_similarity}(\mathbf{u}_{\text{prompt}}, \mathbf{v}_{\text{answer}})$
* **Behavior:** Computes prompt-to-response lexical unigram drift. Dominates model performance.
