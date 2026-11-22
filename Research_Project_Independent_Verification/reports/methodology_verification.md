# Phase 2: Methodology and Theoretical Framework Verification

## 1. Mathematical Formulations & Correctness
* **Med-ISP (Information Specificity):**
  $$\text{Med-ISP} = \min\left(1.0, \frac{|\text{Tokens} \cap \mathcal{L}_D|}{n \cdot 0.05}\right)$$
  *Status:* **VERIFIED**. Formulated as an inverted risk signal ($1.0 - \min(\text{density}, 1.0)$) where dense pharmacological text yields lower risk scores.
* **C-AAS (Contextual Anchor Alignment Score):**
  $$\text{C-AAS} = \min\left(1.0, \frac{|\text{Tokens} \cap \mathcal{L}_C|}{n \cdot 0.04}\right)$$
  *Status:* **VERIFIED**. Captures clinical terms with lemmatization patterns.
* **Med-EEM (Epistemic Entropy Monitor):**
  *Status:* **ISSUE IDENTIFIED & CLARIFIED**. Originally framed as Shannon entropy $H(p) = -p \log_2 p - (1-p) \log_2 (1-p)$. However, hedging word density $p < 0.05$ lies strictly on the monotone-increasing branch ($p < 0.5$). Tree-based classifiers like LightGBM are invariant to monotonic transformations. The signal operates mathematically as a scaled hedging frequency probe.
* **CDT (Cosine Distance of Text):**
  $$\text{CDT} = 1 - \frac{\mathbf{u}_{\text{prompt}} \cdot \mathbf{v}_{\text{answer}}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
  *Status:* **VERIFIED**. Captures lexical divergence between prompt and answer.

## 2. Leakage Prevention and Cross-Validation
* **Stratified 10-Seed Random Splits:** 70% train ($N=4,839$), 30% test ($N=2,075$), stratified by both dataset origin and target label.
* **5-Fold Question-Grouped Cross-Validation:** Verified in `E3_grouped_cv_summary.csv`. All paired responses (human truth + LLM hallucination) originating from the same clinical prompt are constrained to the same fold, guaranteeing zero question-level leakage.

## 3. Fair Baseline Comparison (Factorial Design)
* Earlier iterations confounded feature sets with model architectures (comparing Logistic Regression on lexical features against LightGBM on CliniGuard).
* *Correction Verified:* The experimental suite includes both `Baseline: CDT + length (LR)` and `Baseline: CDT + length (LGB)`, alongside `CliniGuard + length (LGB)` and `Baseline: TF-IDF answer text (LR)`.
