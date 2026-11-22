# Phase 5: Experimental Results and Statistical Audit

## 1. Independent Headline Metric Recalculation (Seed 42, $N=2,075$)
Independently recomputed from raw probabilities in `predictions_seed42_test.csv`:

| Model Architecture | Features | Recalculated AUROC | Saved AUROC (E1) | Delta | Recalculated AP | Saved AP (E1) | Recalculated F1@0.5 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Class** | None | 0.5000 | 0.5000 | 0.0000 | 0.4323 | 0.4323 | 0.0000 |
| **All-Positive** | None | 0.5000 | 0.5000 | 0.0000 | 0.4323 | 0.4323 | 0.6036 |
| **Answer Length (LR)** | Length | 0.7246 | 0.7246 | 0.0000 | 0.6063 | 0.6063 | 0.4097 |
| **Answer Length (LGB)**| Length | 0.8854 | 0.8854 | 0.0000 | 0.8087 | 0.8087 | 0.8010 |
| **Logistic Regression** | 4 Signals | 0.8166 | 0.8166 | 0.0000 | 0.8301 | 0.8301 | 0.7326 |
| **MLP-4L** | 4 Signals | 0.8869 | 0.8869 | 0.0000 | 0.8699 | 0.8699 | 0.7619 |
| **MLP-2L** | 4 Signals | 0.8881 | 0.8881 | 0.0000 | 0.8712 | 0.8712 | 0.7613 |
| **LightGBM** | 4 Signals | **0.9034** | **0.9034** | 0.0000 | **0.8908** | **0.8908** | **0.7781** |
| **CDT + Length (LR)** | CDT + Len | 0.9149 | 0.9149 | 0.0000 | 0.8418 | 0.8418 | 0.8394 |
| **CDT + Length (LGB)** | CDT + Len | **0.9556** | **0.9556** | 0.0000 | **0.9455** | **0.9455** | **0.8726** |
| **CliniGuard + Len (LGB)**| 4 Sig + Len | **0.9594** | **0.9594** | 0.0000 | **0.9503** | **0.9503** | **0.8766** |
| **TF-IDF (LR)** | Text | **0.9932** | **0.9932** | 0.0000 | **0.9916** | **0.9916** | **0.9438** |

*Audit Verification:* **100% Exact Numerical Match** down to 16 decimal places.

## 2. 10-Seed Significance Testing (E1_significance_tests.csv)
Paired differences computed across identical seeds ($\Delta_s = 	ext{AUROC}_{	ext{LGB}} - 	ext{AUROC}_{	ext{Model}}$):
* `Baseline: CDT + length (LGB)` vs `LightGBM`:
  - Mean Paired Difference: **-0.0515** (CDT+Length outperforms 4 signals)
  - 95% Confidence Interval: **[-0.0554, -0.0475]**
  - Paired t-test: $t = -30.73$, $p = 3.01 	imes 10^{-10}$ (Statistically significant)
* `Baseline: TF-IDF answer text (LR)` vs `LightGBM`:
  - Mean Paired Difference: **-0.0875** (TF-IDF outperforms LightGBM)
  - 95% Confidence Interval: **[-0.0919, -0.0831]**
  - Paired t-test: $t = -46.85$, $p = 6.82 	imes 10^{-12}$ (Statistically significant)

## 3. Clinical Risk Triage Verification (E8)
* **Med-HALT ($N=1,453$ held-out test):**
  - GREEN Coverage: **40.19%** (584 / 1,453)
  - False-Reassurance Rate (FRR): **2.40%** (14 missed hallucinations)
  - Safety Lift vs Random Router: **+94.1%**
* **MedHallu ($N=622$ held-out test):**
  - GREEN Coverage: **17.52%** (109 / 622)
  - False-Reassurance Rate (FRR): **24.77%** (27 missed hallucinations)
  - Safety Lift vs Random Router: **+50.1%**
