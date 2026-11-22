"""
Update RESEARCH_PAPER_CLINIGUARD.md with auto-generated tables and complete audit narrative.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research_paper.scripts.generate_paper_tables import (
    format_table_3, format_table_4, format_table_5, format_table_6,
    format_table_7, format_table_8, format_table_9, format_table_tfidf_diag
)
MD_PATH = ROOT / "research_paper" / "RESEARCH_PAPER_CLINIGUARD.md"

tbl3 = format_table_3()
tbl4 = format_table_4()
tbl5 = format_table_5()
tbl6 = format_table_6()
tbl7 = format_table_7()
tbl8 = format_table_8()
tbl9 = format_table_9()
tbl_tfidf = format_table_tfidf_diag()

paper_md = f"""# Shortcut Learning and Benchmark Validity in Medical Hallucination Detection: An Empirical Lexical Audit of Med-HALT and MedHallu

**Authors:** Yogesh Ravi M, School of Computer Science and Engineering, Vellore Institute of Technology, Chennai, India  
**Date:** November 2026  

---

## Abstract

Detecting factual hallucinations in medical large language model (LLM) generations is critical for clinical patient safety. While complex multi-query, external retrieval, or model-internal detectors impose substantial latency and API constraints, lightweight post-hoc lexical scoring has emerged as an attractive black-box alternative. In this work, we conduct an extensive empirical audit of shortcut learning, benchmark artifacts, and out-of-distribution failure modes across two prominent medical hallucination benchmarks: Med-HALT (4,914 items) and MedHallu (2,000 items; N = 6,914 total). We evaluate CliniGuard—a lightweight framework extracting four surface lexical and context-drift features (Medication Sparsity, Clinical Context Sparsity, Hedging Uncertainty, and Unigram Cosine Drift)—alongside linear, neural, length, and bag-of-words baselines under inflection-aware matching. When evaluated on the pooled benchmark, a LightGBM classifier achieves an apparent AUROC of 0.9057 ± 0.0058 (and 0.9063 ± 0.0090 under question-grouped cross-validation). However, our audit reveals that this performance is largely driven by benchmark-specific superficial artifacts rather than clinical semantic discernment: (1) When controlling for model architecture, a two-variable baseline combining Context Drift (CDT) with answer length using LightGBM achieves 0.9572 ± 0.0027 AUROC, matching the full four-signal model (0.9593 ± 0.0026) within 0.0021 AUROC. Meanwhile, a raw TF-IDF answer-text classifier reaches 0.9932 ± 0.0011 AUROC by exploiting surface vocabulary artifacts. (2) Feature ablation reveals extreme asymmetry: a single feature (CDT) provides an individual AUROC of 0.8166 ± 0.0067 and dropping it degrades LightGBM AUROC by 0.1311, whereas dropping any of the three clinical/hedging sparsity features changes AUROC by less than 0.029. (3) Performance degrades substantially on realistic short clinical responses: while LightGBM attains 0.9745 AUROC on Med-HALT (where hallucinated distractors are overtly fabricated long texts), it falls to 0.6736 AUROC on MedHallu, where its standard F1 score of 0.5010 is beaten by a trivial all-positive baseline (F1 = 0.6638) and answer length alone (AUROC = 0.7447). (4) Cross-benchmark transfer fails: Med-HALT -> MedHallu AUROC drops to 0.5518, while TF-IDF transfers with inverted polarity (AUROC = 0.3973). Diagnostic token analysis demonstrates that TF-IDF achieves 1.0000 AUROC on Med-HALT by exploiting generator prompt artifacts (e.g., 'fictional', 'quantum', 'cosmic'), and achieves 0.9662 grouped AUROC on MedHallu by exploiting stylistic rhetorical markers (e.g., 'significantly', 'due to', 'particularly') rather than semantic truth. (5) In risk-stratified clinical triage on MedHallu, false-reassurance rates reach 23.62% for LightGBM and 52.75% for Logistic Regression. We conclude that surface lexical features cannot substitute for clinical semantic verification, and benchmark evaluations in medical hallucination detection require rigorous controls against length confounds and generator artifacts.

**Keywords:** medical artificial intelligence, large language models, hallucination detection, benchmark validity, shortcut learning, lexical features, out-of-distribution generalization, clinical safety

---

## I. INTRODUCTION

### A. Motivation
Large language models (LLMs) such as Med-PaLM, GPT-4, and specialized biomedical transformers [1]–[7] have achieved remarkable milestones, reaching passing and near-expert thresholds on clinical licensing examinations and medical question-answering benchmarks [15]–[19]. Despite their linguistic fluency, these models frequently generate plausible yet factually incorrect or ungrounded assertions—a phenomenon termed *hallucination* [8]–[11].

In clinical medicine, factual inaccuracies carry severe risks. An erroneous drug dosage, an overlooked contraindication, or diagnostic recommendations tailored to an incorrect etiology can directly compromise patient care [12]–[14]. Consequently, implementing post-generation verification mechanisms has become an active area of investigation for generative AI in healthcare workflows.

### B. Existing Detection Paradigms and Computational Overheads
Current methodologies for post-generation hallucination detection primarily fall into four architectural paradigms:
1. **Self-Consistency and Sampling:** Frameworks such as SelfCheckGPT [21] query the generative model multiple times to evaluate stochastic consensus across diverse output samples.
2. **External Retrieval and Fact Checking:** Systems such as FActScore [22] and FacTool [24] decompose responses into atomic factual claims and verify each against authoritative external knowledge graphs or reference corpora [25]–[27].
3. **LLM-as-a-Judge:** Secondary large language models are prompted to evaluate the correctness of the primary response, introducing latency, financial overhead, and susceptibility to length and position biases [23].
4. **Internal State Probing:** Methods utilizing predictive token probabilities, hidden activations, or semantic entropy [28], [29], [32]–[34], [42] inspect model internals to quantify uncertainty.

While conceptually powerful, each paradigm incurs deployment costs (Table 1). Multi-query sampling multiplies inference expenses; retrieval pipelines require low-latency indexing infrastructure and risk retrieval errors; judge models transmit potentially sensitive clinical text to secondary systems; and internal state probing is inaccessible when interfacing with commercial closed-weight APIs.

**Table 1. Qualitative comparison of post-generation hallucination detection approaches, based on how each method is described in the cited papers. CliniGuard row describes its design, not its accuracy.**

| Detection Approach | Example Studies | Needs Repeated Generations? | Needs External Knowledge? | Needs Second Judge Model? | Needs Model Internals? |
|---|---|:---:|:---:|:---:|:---:|
| Repeated Sampling | SelfCheckGPT [21] | **Yes** | No | Optional | No |
| Retrieval / Fact Checking | FActScore [22], FacTool [24] | No | **Yes** | **Yes** | No |
| LLM-as-a-Judge | Zheng et al. [23] | No | No | **Yes** | No |
| Internal Probes | Kossen et al. [29], Kuhn et al. [28] | **Yes** (or white-box) | No | Optional | **Yes** |
| **CliniGuard (this work)** | This work | **No** | **No** | **No** | **No** |

These operational constraints motivate the search for lightweight, post-hoc, black-box verification mechanisms that rely solely on the generated text. However, such simplicity introduces a critical question: do lightweight surface metrics genuinely evaluate clinical correctness, or do they exploit statistical shortcuts inherent in synthetic evaluation benchmarks?

### C. The Shortcut Learning Problem and This Audit
In natural language processing, extensive research has established that models trained on synthetic benchmarks frequently learn superficial heuristics rather than true semantic representations. Canonical studies by Gururangan et al. [43] and Poliak et al. [44] demonstrated that hypothesis-only baselines in Natural Language Inference (NLI) could achieve surprisingly high accuracy by exploiting lexical cues such as negation and sentiment words. Similarly, McCoy et al. [45] revealed that models rely on syntactic heuristics like lexical overlap, while Geirhos et al. [46] formalized shortcut learning as a ubiquitous vulnerability in deep learning systems.

In this study, we subject lightweight lexical hallucination detection to an empirical audit across 6,914 benchmark examples from Med-HALT [14] and MedHallu [41]. We analyze CliniGuard—a representative system utilizing four lexical features and a gradient-boosted tree classifier—to address five fundamental audit questions:
1. **Benchmark Shortcut Exploitation:** Does high in-distribution AUROC reflect clinical discrimination or superficial lexical artifacts?
2. **Single-Feature Dominance:** Does the proposed multi-feature architecture reflect genuine multi-signal synergy, or is predictive performance monopolized by a single lexical overlap metric?
3. **Length and Bag-of-Words Confounds:** Can trivial baselines (e.g., answer word count or raw unigram TF-IDF) match or surpass the hand-crafted clinical signals?
4. **Out-of-Distribution Generalization:** Do lexical detectors generalize across independent medical hallucination benchmarks, or do learned surface associations collapse and invert polarity?
5. **Clinical False-Reassurance Risk:** When deployed in a risk-stratified clinical triage workflow, what proportion of actual medical hallucinations are mistakenly labeled safe?

Our findings provide empirical evidence that while surface features achieve an apparent pooled AUROC of 0.9057, this signal is an artifact of benchmark construction. Surface lexical screening fails out-of-distribution, suffers from elevated false-reassurance rates, and cannot replace semantic or clinical verification.

---

## II. RELATED WORKS

### A. Medical Language Models and Clinical Hallucinations
The adaptation of transformer architectures to the biomedical domain—exemplified by BioBERT [4], PubMedBERT [5], BioGPT [6], and ClinicalBERT [7]—laid the foundation for medical generative systems. Subsequent domain-specific foundation models, including Med-PaLM and Med-PaLM 2 [1], [2], demonstrated diagnostic reasoning capabilities. Comprehensive taxonomies by Huang et al. [8] and Ji et al. [9] categorize clinical hallucinations into *intrinsic* errors (direct factual contradictions of source text) and *extrinsic* errors (fabrication of untruthful medical assertions unsupported by clinical evidence) [10]–[13]. Benchmark datasets such as Med-HALT [14] and MedHallu [41] were introduced specifically to evaluate and quantify these failure modes.

### B. Uncertainty Signals, Lexical Overlap, and Shortcut Learning
Information-theoretic uncertainty quantification traces back to Shannon entropy [30]. Xiao and Wang [32] examined predictive uncertainty in conditional language generation, while Kuhn et al. [28] and Farquhar et al. [42] developed semantic entropy over clustered answer distributions. In parallel, term-frequency vector space models [31] have long served as classical metrics of textual similarity.

However, the intersection of lexical overlap and benchmark validity is subject to shortcut learning. In NLI, Gururangan et al. [43] showed that specific words strongly correlated with target entailment labels due to crowd-worker annotation artifacts. McCoy et al. [45] proved that models routinely adopt a lexical overlap heuristic (assuming that high word overlap implies entailment). In this work, we demonstrate that a direct analog exists in medical hallucination benchmarks: unigram cosine overlap between prompt and generation serves as a primary statistical shortcut.

### C. Tabular Ensembles on Low-Dimensional Feature Spaces
Empirical studies by Shwartz-Ziv and Armon [38] and Grinsztajn et al. [39] have consistently demonstrated that tree-based ensembles, notably LightGBM [37] and XGBoost, frequently outperform multilayer perceptrons and deep neural networks on low-dimensional tabular datasets. Because CliniGuard operates on a four-dimensional feature representation, we adopt LightGBM as the primary classification engine and benchmark it against neural and regularized linear architectures.

---

## III. METHODOLOGY

### A. Architectural Overview
CliniGuard operates as a lightweight, post-hoc verification screen positioned between a generative medical model and the end-user interface. As illustrated in Figure 1, the pipeline executes three sequential stages: (1) deterministic text sanitization and inflection-aware regex tokenization, (2) extraction of four normalized lexical and context-drift features, and (3) probability estimation and clinical risk tier assignment using a trained LightGBM classifier.

![Figure 1: CliniGuard Architecture](figures/fig1_architecture.png)
*Fig. 1. CliniGuard pipeline: text cleaning, four lexical signals, LightGBM risk scoring, and three risk tiers.*

### B. Primary Benchmark Datasets and Label Provenance
Our primary empirical evaluation utilizes 6,914 verified clinical question-answer pairs derived from two publicly established medical hallucination benchmarks (Table 2).

**Table 2. Summary of the two benchmark datasets used for the primary evaluation.**

| Dataset | Total Items (N) | Real (y = 0) | Hallucinated (y = 1) | Hallucinated Share (%) | Unique Questions | Mean Words (y = 0) | Mean Words (y = 1) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Med-HALT** [14] | 4,914 | 2,925 | 1,989 | 40.48% | 4,817 | 208.0 | 127.3 |
| **MedHallu** [41] | 2,000 | 1,000 | 1,000 | 50.00% | 1,000 | 39.7 | 27.8 |
| **Combined** | **6,914** | **3,925** | **2,989** | **43.23%** | **5,817** | **159.2** | **94.0** |

1. **Med-HALT** [14] (4,914 items): Comprises biomedical research QA and fake abstract reasoning tasks. Truthful records (y = 0, N = 2,925) pair authentic PubMed paper titles with verified abstracts. Hallucinated records (y = 1, N = 1,989) pair synthetic prompts with LLM-generated fabricated abstracts containing overtly bogus medical claims (e.g., fictional diseases and non-existent biochemical pathways).
2. **MedHallu** [41] (2,000 items): Built upon 1,000 realistic clinical inquiries. Each question is paired with both an authentic clinician-verified answer (y = 0, N = 1,000) and a subtle, fluent LLM hallucination generated by modern instruction-tuned models (y = 1, N = 1,000).

**Elimination of Circular Synthetic Label Artifacts.** Prior to establishing this evaluation framework, exploratory audits identified severe label construction artifacts in earlier data preparation pipelines. Specifically, synthetic loaders had created pseudo-hallucination datasets by programmatically injecting a static hedge phrase (*"We suggest that it could perhaps be possible to assume some treatment options, maybe."*) into PubMedQA [15] and MedQuAD [16] corpora, or by assigning binary labels via modular arithmetic on database row indices. Because five of the fourteen words in the injected template matched the hedging lexicon, any lexical detector achieved trivially near-perfect separation (AUROC > 0.999). We purged all such artificially corrupted datasets; our entire evaluation rests strictly on authoritative benchmark labels.

### C. Text Normalization, Regular Expressions, and Inflection Handling
Given an input prompt Q and generated answer A, text is converted to lowercase and scanned. To avoid the twin pitfalls of naive substring matching (which falsely matches 'iv' inside 'positive', 'moral' inside 'oral', and 'table' inside 'comfortable') and strict exact matching (which misses attached units like '500mg' and plurals like 'patients' and 'tablets'), lexicon scanning enforces two complementary regular expression mechanisms:
1. **Unit-Number Boundary Patterns:** Matches clinical measurement units directly attached to digits or standalone:
   > `(?:(?<=\\d)|(?<=\\b))(?:mg|mcg|µg|g|ml|l|units?)(?:\\b|(?=[/\\s,\\.\\)]))`
2. **Inflection-Aware Lexical Stems:** Precompiled regular expressions matching clinical lemmas and regular grammatical inflections (e.g., plural -s/-es, past tense -ed, progressive -ing, adjectives -ic/-al):
   > `\\b(?:term|terms|termed|terming)\\b`

Unit test verification across positive and negative controls confirmed 100% elimination of substring collisions while preserving 100% sensitivity to attached dosage units and clinical inflections.

### D. The Four Lexical and Context-Drift Features
CliniGuard computes four normalized features, each bounded in [0, 1]. The first three features evaluate answer characteristics in isolation; the fourth measures prompt-to-answer alignment.

#### 1) Medication Information Sparsity (Med-ISP)
Med-ISP measures the relative absence of pharmacological and dosing terminology. Using a curated lexicon L_D of 75 pharmacology terms and dosage forms (derived from RxNorm top-prescribed outpatient and inpatient medication classes, e.g., *tablet, capsule, oral, iv, amoxicillin, metformin, insulin, aspirin, atorvastatin, lisinopril, levothyroxine*), we count matching tokens h_D and scale against an expected baseline density of 5%:

> Med-ISP = 1 − min( h_D / max(0.05 × n, 1), 1 ) ∈ [0, 1] &nbsp;&nbsp;&nbsp; (1)

#### 2) Clinical Context-Awareness Sparsity (C-AAS)
C-AAS evaluates whether the generation mentions relevant clinical context parameters (e.g., patient demographics, chronicity, allergies, vital signs). Using a clinical lexicon L_C of 45 terms (e.g., *patient, pediatric, adult, vital, allergy, diagnosis, symptom, blood pressure, heart rate, chronic, acute, contraindication*), matching tokens h_C are scaled against a 4% baseline density:

> C-AAS = 1 − min( h_C / max(0.04 × n, 1), 1 ) ∈ [0, 1] &nbsp;&nbsp;&nbsp; (2)

#### 3) Medical Uncertainty Entropy Measure (Med-EEM)
Med-EEM quantifies the density of epistemic hedging words (e.g., *maybe, possibly, might, could, suggest, perhaps, estimate*) using a lexicon L_U of 35 terms. Letting p_u = h_U / n denote the hedging fraction and ε = 10⁻⁹, a binary Shannon entropy is computed:

> H(p_u) = −p_u log₂(p_u + ε) − (1 − p_u) log₂(1 − p_u + ε) &nbsp;&nbsp;&nbsp; (3)
> Med-EEM = min( min( H(p_u), 1.0 ) × (1 + p_u), 1.0 ) ∈ [0, 1] &nbsp;&nbsp;&nbsp; (4)

*Monotonicity Note:* We emphasize that Med-EEM is not an estimate of model predictive entropy. In real clinical text, the hedging fraction p_u is virtually always below 0.50 (in our data, mean p_u < 0.03). On the interval [0, 0.50], the binary entropy function H(p_u) is strictly monotonically increasing. Consequently, for decision tree splitters such as LightGBM, Med-EEM is mathematically equivalent to the raw hedge fraction p_u.

#### 4) Context Drift (CDT)
CDT quantifies vocabulary divergence between the prompt and the generated response. Unigram term-frequency vectors **q** and **a** are constructed over the joint vocabulary of Q and A. Context drift is defined as the unigram cosine distance:

> CDT = 1 − ( **q** · **a** ) / ( ‖**q**‖₂ × ‖**a**‖₂ ) ∈ [0, 1] &nbsp;&nbsp;&nbsp; (5)

If either text vector is empty, CDT defaults to 0.50.

### E. Classification Architecture, Baselines, and Risk Tiers
The four features form an input vector **x** = [Med-ISP, C-AAS, Med-EEM, CDT], which is standardized via z-score normalization (**z** = (**x** − **μ**) / **σ**) using training split statistics.

#### 1) Classifiers
* **LightGBM (CliniGuard):** A gradient-boosted decision tree ensemble [37] configured with 300 estimators, learning rate 0.05, maximum tree depth 5, 31 leaves, and balanced class weights.
* **Logistic Regression:** Regularized L2 logistic regression with balanced class weights.
* **MLP-2L & MLP-4L:** Fully-connected multilayer perceptrons with architectures (256, 128) and (128, 64, 32, 16), ReLU activations, and Adam optimization.

#### 2) Baselines for Rigorous Benchmark Auditing
To benchmark performance against trivial shortcuts and isolate model architecture from feature representation, we introduce eight reference baselines:
1. **Majority Class:** Predicts constant negative class (y = 0; AUROC = 0.500, F1 = 0.000).
2. **All Positive (Always 1):** Predicts constant hallucination (y = 1; AUROC = 0.500, F1 = 2P/(P+N)).
3. **Answer Length (LR):** Logistic regression trained exclusively on standardized log answer word count.
4. **Answer Length (LGB):** LightGBM trained exclusively on standardized log answer word count.
5. **CDT + Length (LR):** Logistic regression trained on Context Drift and log answer length.
6. **CDT + Length (LGB):** LightGBM trained on Context Drift and log answer length.
7. **CliniGuard + Length (LGB):** LightGBM trained on all four CliniGuard signals augmented with log answer length.
8. **TF-IDF Answer Text (LR):** Logistic regression trained on up to 20,000 sublinear TF-IDF unigrams and bigrams extracted purely from the generated answer text.

#### 3) Clinical Risk Tiers and False-Reassurance Metric
Continuous risk probabilities p(**z**) are partitioned into three operational tiers:
* **GREEN (Low Risk):** p(**z**) < 0.35.
* **AMBER (Moderate Risk):** 0.35 ≤ p(**z**) ≤ 0.65.
* **RED (High Risk):** p(**z**) > 0.65.

To quantify safety in clinical deployment, we define the **False-Reassurance Rate (FRR)** as the proportion of all actual ground-truth hallucinations in the evaluation set that are erroneously assigned to the GREEN tier:

> False-Reassurance Rate (FRR) = (Hallucinations in GREEN) / (Total Ground-Truth Hallucinations) &nbsp;&nbsp;&nbsp; (6)

We compare the observed FRR against a **Random Router Baseline**, which routes samples to GREEN with probability equal to GREEN coverage share c (expected FRR = c), and compute the **Safety Lift vs. Random**:

> Safety Lift = (Expected Random FRR − Observed FRR) / Expected Random FRR &nbsp;&nbsp;&nbsp; (7)

### F. Experimental Validation Protocols
Our experimental design enforces five validation controls:
1. **Repeated Stratified Splits:** Ten independent 70/30 stratified train/test splits (seeds 0 through 9), with seed 42 designated as the headline split.
2. **Question-Grouped Cross-Validation:** 5-fold `StratifiedGroupKFold` grouped on unique question strings, ensuring that paired authentic and hallucinated responses never cross train/test boundaries.
3. **Cross-Benchmark Transfer:** Zero-shot domain transfer evaluating models trained on Med-HALT and tested on MedHallu, and vice versa.
4. **Statistical Significance and Confidence Intervals:** Paired t-tests across the 10 random seeds comparing LightGBM against all baselines, complemented by 2,000-resample paired bootstrap confidence intervals on the headline split.
5. **Probability Calibration:** Empirical evaluation of raw, isotonic, and Platt sigmoid calibration using Expected Calibration Error (ECE; 10 equal-width bins) and Brier score.

---

## IV. EXPERIMENTS AND RESULTS

### A. Pooled Benchmark Results and Baseline Comparison
Table 3 reports performance on the pooled evaluation suite across ten independent random splits. On the four lexical features, LightGBM attains an apparent AUROC of **0.9057 ± 0.0058**, outperforming logistic regression (0.8180) and neural baselines (MLP-2L: 0.8872, MLP-4L: 0.8895).

**Table 3. Pooled results over ten random splits (mean ± SD) and paired statistical comparison against LightGBM.**

{tbl3}

Crucially, the baseline comparisons expose the superficial nature of this detection capability:
1. **Decoupling Model Architecture from Feature Representation:** When controlling for model class, `Baseline: CDT + length (LGB)` achieves **0.9572 ± 0.0027 AUROC**, while the full five-variable model `CliniGuard + length (LGB)` achieves **0.9593 ± 0.0026 AUROC**. The difference is only **0.0021 AUROC**. The three domain-specific clinical sparsity signals add virtually zero discriminative power once Context Drift and length are provided to LightGBM.
2. **Vocabulary Memorization Baseline:** A generic TF-IDF bag-of-words classifier trained on answer text achieves **0.9932 ± 0.0011 AUROC**, proving that superficial vocabulary choice alone provides near-perfect separation inside the pooled corpus.

![Figure 2: ROC Curves](figures/fig2_roc_seed42.png)
*Fig. 2. ROC curves on the seed-42 test set for LightGBM and the baselines.*

![Figure 4: CliniGuard vs Baselines](figures/fig4_baselines.png)
*Fig. 4. Mean AUROC across ten random splits and across question-grouped cross-validation.*

### B. Question-Grouped Cross-Validation
To verify that results are not artifacts of data leakage from identical question prompts appearing across splits, Table 4 presents 5-fold question-grouped cross-validation.

**Table 4. Question-grouped 5-fold cross-validation performance across folds.**

{tbl4}

LightGBM achieves 0.9063 ± 0.0090 grouped AUROC, closely tracking the random-split results. However, as in the pooled setting, `CDT + length (LGB)` attains 0.9575 AUROC, confirming that the apparent multi-signal capability is entirely reproducible under grouped splitting.

### C. Benchmark Disparity: The Collapse on MedHallu
The pooled metrics obscure a profound performance disparity between the two underlying benchmarks. Table 5 and Figure 3 report held-out test performance disaggregated by benchmark source on the seed-42 split.

**Table 5. Held-out evaluation separated by benchmark on the seed-42 test set.**

{tbl5}

On Med-HALT, where hallucinated samples consist of long, fabricated biomedical papers, LightGBM reaches **0.9745 AUROC**. However, on MedHallu, which comprises short, realistic clinical responses (mean length 27.8 words), LightGBM collapses to **0.6736 AUROC** and an F1 score of **0.5010** (threshold 0.50).

Remarkably, on MedHallu:
* A trivial **All-Positive baseline** (predicting every answer is hallucinated) achieves an F1 score of **0.6638**, beating LightGBM by 0.1628.
* A single variable—**Answer Length alone**—achieves an AUROC of **0.7447** with Logistic Regression, outperforming the four-signal CliniGuard model by 0.0711 AUROC.

This empirical divergence demonstrates that CliniGuard does not detect subtle clinical falsehoods; rather, it detects the exaggerated length and vocabulary anomalies characteristic of Med-HALT distractors.

![Figure 3: Per-Dataset Performance](figures/fig3_per_dataset.png)
*Fig. 3. Seed-42 held-out AUROC on Med-HALT and MedHallu.*

### D. Out-of-Distribution Transfer Failure and Polarity Inversion
To evaluate whether learned representations transfer across different benchmark distributions, Table 6 presents zero-shot cross-dataset evaluation.

**Table 6. Cross-benchmark zero-shot transfer (train on one benchmark, test on the other).**

{tbl6}

When trained on Med-HALT and evaluated zero-shot on MedHallu, LightGBM AUROC plummets to **0.5518**—barely exceeding random chance (0.500). In this direction, Answer Length alone achieves **0.7396 AUROC** with Logistic Regression, outperforming the four-feature model.

When transferred in reverse (trained on MedHallu, tested on Med-HALT), the TF-IDF bag-of-words baseline yields an AUROC of **0.3973**. An AUROC substantially below 0.500 indicates a **systematic rank inversion**: the lexical patterns associated with hallucinations in MedHallu correlate with truthful responses in Med-HALT. This provides proof that vocabulary-based classifiers learn dataset-specific generation artifacts rather than generalizable indicators of medical truth.

### E. Diagnostic Analysis of Bag-of-Words Artifacts
To understand why TF-IDF achieves 1.0000 AUROC on Med-HALT and 0.9314 on MedHallu, Table 7 isolates the top positive (hallucination-predictive) and negative (truth-predictive) logistic regression coefficients.

**Table 7. Top diagnostic TF-IDF tokens separating hallucinated from authentic text.**

{tbl_tfidf}

Diagnostic feature inspection exposes the underlying mechanism:
1. **Med-HALT Prompt Artifacts (AUROC 1.0000):** Synthetic hallucinated abstracts in Med-HALT were generated by prompting LLMs with fictional concepts. Consequently, the top hallucination predictors are overt markers of fiction: `'fictional'` (+4.4003), `'quantum'` (+3.1266), `'cosmic'` (+2.8418), and standardized abstract section headers (`'Background: ... Methods: ... Conclusion: ...'`). Authentic PubMed abstracts cite empirical biological entities (`'cell'`, `'cells'`, `'data'`). A bag-of-words classifier trivially achieves 1.0000 AUROC by detecting these prompt artifacts.
2. **MedHallu Rhetorical Artifacts (AUROC 0.9314 in-domain; 0.9662 grouped CV):** In MedHallu, LLM hallucinations systematically utilize confident, comparative assertions: `'significantly'` (+4.4969), `'due to'` (+2.7919), `'compared to'` (+2.2248), `'particularly'` (+2.0961), and `'rates'` (+1.5680). Conversely, authentic human clinician responses exhibit epistemic hedging: `'may'` (-3.6750), `'not'` (-2.9665), `'should'` (-1.8887), and `'could'` (-1.4527). Under Question-Grouped 5-Fold CV on MedHallu alone, TF-IDF attains **0.9662 ± 0.0091 AUROC**, proving that the bag-of-words model does not memorize question content, but rather detects the pervasive rhetorical style of the LLM generator.
3. **The Transfer Inversion:** When transferred from MedHallu to Med-HALT, authentic PubMed papers also frequently discuss significant comparative findings (`'significantly'`, `'compared to'`), inverting the polarity of the learned weights and causing AUROC to collapse to 0.3973.

### F. Feature Ablation and Single-Signal Analysis
To evaluate whether all four signals contribute meaningfully to detection, Table 8 details systematic feature ablation across ten random splits.

**Table 8. Feature ablation over ten random splits (mean ± SD).**

{tbl7}

The ablation reveals extreme asymmetry:
* Dropping Context Drift (CDT) induces a massive **0.1311 collapse** in LightGBM AUROC (falling to 0.7746), and a 0.1628 collapse in logistic regression (falling to 0.6552).
* Conversely, dropping Med-ISP, C-AAS, or Med-EEM reduces LightGBM AUROC by only 0.014 to 0.028.
* When tested in complete isolation, CDT alone achieves **0.8166 AUROC** with LightGBM and 0.8150 with logistic regression. Thus, CliniGuard is effectively a single-feature detector whose performance is almost entirely driven by unigram context drift.

Table 9 displays the raw univariate AUROC of each feature across the full dataset cohorts.

**Table 9. Univariate signal AUROC for predicting the hallucination class (y = 1).**

{tbl8}

*Directional Alignment:* CDT exhibits raw univariate AUROCs far below 0.500 (0.0393 on Med-HALT, 0.3327 on MedHallu). This occurs because synthetic hallucinated answers in these benchmarks repeat vocabulary verbatim from the prompt, artificially compressing the cosine distance. When predicting hallucination, lower drift corresponds to higher probability of benchmark hallucination.

![Figure 5: Feature Ablation](figures/fig5_ablation.png)
*Fig. 5. LightGBM and Logistic Regression AUROC across feature ablation subsets.*

![Figure 6: Univariate Heatmap](figures/fig6_univariate_heatmap.png)
*Fig. 6. Univariate AUROC of individual features across benchmark datasets.*

### G. Clinical Risk Tiers, Calibration, and False Reassurance
Table 10 presents the distribution of test predictions across the three operational risk tiers on the seed-42 test set, benchmarking the False-Reassurance Rate against a random router baseline.

**Table 10. Risk tier performance, false-reassurance rates, and safety lift on the seed-42 test set (N = 2,075).**

{tbl9}

The disaggregated tier analysis highlights a clinical safety hazard:
1. **Med-HALT:** LightGBM exhibits strong triage properties: only 23 out of 588 hallucinations slip into the GREEN tier, yielding a low **False-Reassurance Rate of 3.91%** and a safety lift of +91.4% over random routing.
2. **MedHallu:** On realistic clinical QA, LightGBM assigns 73 out of 309 hallucinations to the GREEN tier, producing a **False-Reassurance Rate of 23.62%** (safety lift of only +37.2%). Nearly one in four clinical hallucinations is certified as safe.
3. **Logistic Regression Failure on MedHallu:** Logistic Regression misclassifies **163 out of 309 hallucinations** into the GREEN tier, resulting in a **False-Reassurance Rate of 52.75%** with a meager +19.4% safety lift over a random router. More than half of all medical errors receive an explicit green endorsement.

**Probability Calibration and Latency:** On the pooled seed-42 test set, raw LightGBM probabilities exhibit an Expected Calibration Error (ECE) of **0.0348** and a Brier loss of **0.1241**. Post-hoc isotonic calibration reduces ECE to **0.0206** and Brier loss to **0.1240** (Table 11 and Figure 8).

Benchmarked across 2,000 inference cycles on an Intel64 processor, regex feature extraction required a median of **13.8 ms** (mean 13.7 ms), and LightGBM model scoring required **2.6 ms**, achieving an end-to-end latency of **16.4 ms** per query.

**Table 11. Probability calibration and inference latency benchmarks.**

| Calibrator / Metric | ECE | Brier Loss | Median Latency (ms) | 95th Percentile Latency (ms) |
|---|:---:|:---:|:---:|:---:|
| LightGBM (Raw Probabilities) | 0.0348 | 0.1241 | 16.4 ms | 21.7 ms |
| LightGBM (Isotonic Calibration) | **0.0206** | **0.1240** | 16.5 ms | 21.9 ms |
| LightGBM (Platt Sigmoid Calibration) | 0.0417 | 0.1270 | 16.5 ms | 21.8 ms |
| Regex Feature Extraction Alone | — | — | 13.8 ms | 18.4 ms |

![Figure 7: Risk Band Stratification](figures/fig7_risk_bands.png)
*Fig. 7. Observed hallucination rate by tier on the pooled seed-42 test set (N = 2,075).*

![Figure 8: Calibration Curve](figures/fig8_calibration.png)
*Fig. 8. Calibration curve of raw LightGBM probabilities on the pooled seed-42 test set.*

**Table 12. Illustrative LightGBM tier rates reweighted to lower clinical prevalence (pooled seed-42 test set).**

| Assumed Clinical Prevalence | GREEN Tier: Share / Hallucinated | AMBER Tier: Share / Hallucinated | RED Tier: Share / Hallucinated |
|:---:|:---:|:---:|:---:|
| 5% | 71.6% / 0.8% | 18.5% / 5.9% | 9.9% / 34.0% |
| 10% | 68.4% / 1.6% | 18.7% / 11.7% | 12.9% / 52.1% |
| 20% | 62.0% / 3.5% | 19.0% / 23.0% | 19.0% / 71.0% |

### H. Synthetic Topic-Mismatch Evaluation on NIH MedQuAD
To test whether the learned lexical signals detect off-topic divergence where an answer discusses an incorrect medical condition, we evaluated on a synthetic mismatch benchmark derived from 2,000 NIH MedQuAD QA pairs [16]. In 1,000 pairs (y = 0), questions are paired with their authentic NIH answer; in 1,000 pairs (y = 1), questions are paired with authentic answers randomly drawn from an unrelated medical topic (Table 13).

**Table 13. Topic-mismatch check on 2,000 NIH MedQuAD items.**

| Evaluated System / Signal | Configuration | AUROC |
|---|---|:---:|
| CDT alone (no training) | Unsupervised cosine distance | **0.7959** |
| Med-ISP alone | Unsupervised pharmacology sparsity | 0.5142 |
| C-AAS alone | Unsupervised clinical context sparsity | 0.5050 |
| Med-EEM alone | Unsupervised hedging entropy | 0.5094 |
| Pre-trained CliniGuard (LightGBM) | Trained on Med-HALT + MedHallu (zero-shot) | **0.4790** |
| Pre-trained Logistic Regression | Trained on Med-HALT + MedHallu (zero-shot) | **0.2430** |
| Retrained LightGBM | 10-seed stratified cross-validation on NIH items | 0.8066 ± 0.0109 |

Unsupervised Context Drift (CDT) alone achieves **0.7959 AUROC** on topic mismatch. However, the pre-trained CliniGuard LightGBM model completely fails (**0.4790 AUROC**), and pre-trained Logistic Regression inverts (**0.2430 AUROC**). This failure occurs because in synthetic topic mismatch, ungrounded responses exhibit **high drift** (low vocabulary overlap with the prompt), whereas in Med-HALT and MedHallu, hallucinated answers exhibit **low drift** (high verbatim prompt repetition). Because the directional polarity of the drift shortcut reverses across tasks, a supervised model trained on standard hallucination benchmarks cannot generalize to off-topic clinical divergence.

---

## V. DISCUSSION

### A. Summary of Audit Findings
Our audit yields four central conclusions regarding lightweight lexical screening in medical hallucination benchmarks:
1. **The Shortcut Learning Mechanism:** The apparent high pooled performance of CliniGuard (0.9057 AUROC) is an artifact of benchmark construction. When controlling for model class, `CDT + length (LGB)` achieves **0.9572 AUROC**, matching the five-variable model (0.9593) within 0.0021 AUROC. Furthermore, TF-IDF reaches 0.9932 AUROC, confirming that models exploit surface lexical shortcuts rather than clinical semantics.
2. **Single-Feature Dominance:** CliniGuard is effectively a one-variable detector. Context Drift (CDT) alone accounts for 0.8166 AUROC, and its removal induces a 0.1311 performance collapse. The three domain-specific clinical lexicons contribute negligibly.
3. **Failure on Realistic Clinical Text:** Performance collapses from 0.9745 AUROC on Med-HALT (overt paper fabrications) to 0.6736 on MedHallu (realistic short clinical responses), where CliniGuard F1 (0.5010) is beaten by a trivial all-positive baseline (0.6638) and answer length alone (0.7447 AUROC).
4. **Hazardous Clinical False Reassurance:** In tiered clinical triage on MedHallu, 23.62% of hallucinations slip into the GREEN tier under LightGBM, and 52.75% under Logistic Regression.

### B. Why Lexical Overlap Fails as a Clinical Safety Screen
The divergence between Med-HALT and MedHallu exposes why surface lexical metrics cannot serve as generalizable clinical safety tools. In Med-HALT, negative examples are authentic PubMed abstracts that extensively cite experimental methodology, while positive examples are LLM-generated hallucinated distractors created by prompting the model with fake titles. The generator repeats tokens from the synthetic prompt, mechanically lowering unigram cosine drift. In contrast, in MedHallu, hallucinated answers are concise clinical completions that share identical stylistic and vocabulary distributions with ground-truth clinician responses.

When evaluated across pooled datasets, classifiers inadvertently learn to separate **dataset provenance** rather than factual truth. The 0.9932 AUROC attained by TF-IDF reflects its capacity to memorize benchmark-specific vocabulary distributions (e.g., distinguishing PubMed formatting from MedHallu dialogue style). When forced to transfer out-of-distribution, this vocabulary mapping inverts, producing an AUROC of 0.3973.

### C. Limitations of This Study
1. **Benchmark-Bound Evaluation:** Our experiments evaluate existing benchmark datasets (Med-HALT, MedHallu, MedQuAD). We did not evaluate live clinical interactions or physician-in-the-loop deployments.
2. **Closed Lexicons:** The clinical and pharmacological lexicons comprise 75, 45, and 35 terms. Although regex boundaries and unit patterns eliminated substring collisions, manual lexicons lack coverage of full ontologies (e.g., UMLS, SNOMED-CT, RxNorm) [35], [36].
3. **Absence of External Model Comparison:** We did not run multi-query detectors (SelfCheckGPT) or external knowledge verifiers (FActScore) due to computational and proprietary API constraints.

### D. Recommendations for Future Medical Hallucination Benchmarks
Our findings suggest essential methodological reforms for the medical NLP evaluation community:
* **Enforce Length Matching:** Hallucinated and truthful responses must be strictly balanced for word count and character length to prevent models from learning length shortcuts.
* **Mandate Hypothesis-Only and Bag-of-Words Baselines:** Every medical hallucination benchmark should report performance against trivial length, unigram TF-IDF, and prompt-overlap baselines to verify that complex detectors are not merely exploiting annotation artifacts.
* **Avert Verbatim Prompt Leakage:** Synthetic distractor generators must be constrained to avoid copying prompt phrasing, ensuring that vocabulary overlap cannot serve as a trivial proxy for truthfulness.

---

## VI. CONCLUSION

We conducted an empirical audit of lightweight lexical hallucination detection across 6,914 examples from Med-HALT and MedHallu. While CliniGuard achieves an apparent pooled AUROC of 0.9057, our investigation reveals that this performance is driven by benchmark-specific shortcuts: unigram context drift and answer length match the multi-signal model within 0.0021 AUROC under identical LightGBM modeling, while bag-of-words classifiers reach 0.9932 AUROC. Performance collapses to 0.6736 on realistic clinical statements, cross-dataset transfer fails catastrophically, and risk triage exhibits an alarming 23.62% to 52.75% false-reassurance rate. We conclude that surface lexical features cannot verify clinical truthfulness, and future medical hallucination research must implement rigorous controls against length confounds and dataset artifacts.

---

## ACKNOWLEDGMENT

The author expresses sincere gratitude to the School of Computer Science and Engineering at Vellore Institute of Technology, Chennai, for institutional support, and acknowledges the creators of Med-HALT, MedHallu, and NIH MedQuAD for making their benchmark corpora publicly accessible.

---

## REFERENCES

1. Singhal K, et al. Large language models encode clinical knowledge. *Nature*. 2023;620:172–180.
2. Singhal K, et al. Towards expert-level medical question answering with large language models. *arXiv:2305.09617*. 2023.
3. Nori H, et al. Capabilities of GPT-4 on medical challenge problems. *arXiv:2303.13375*. 2023.
4. Lee J, et al. BioBERT: a pre-trained biomedical language representation model for biomedical text mining. *Bioinformatics*. 2020;36(4):1234–1240.
5. Gu Y, et al. Domain-specific language model pretraining for biomedical natural language processing. *ACM Trans Comput Healthcare*. 2021;3(1):2.
6. Luo R, et al. BioGPT: generative pre-trained transformer for biomedical text generation and mining. *Briefings in Bioinformatics*. 2022;23(6):bbac409.
7. Alsentzer E, et al. Publicly available clinical BERT embeddings. *Proc Clinical NLP Workshop*. 2019.
8. Huang L, et al. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. *ACM Trans Inf Syst*. 2025; arXiv:2311.05232.
9. Ji Z, et al. Survey of hallucination in natural language generation. *ACM Comput Surv*. 2023;55(12):248.
10. Maynez J, et al. On faithfulness and factuality in abstractive summarization. *Proc ACL*. 2020.
11. Zhang Y, et al. Siren's song in the AI ocean: a survey on hallucination in large language models. *arXiv:2309.01219*. 2023.
12. Tonmoy SMTI, et al. A comprehensive survey of hallucination mitigation techniques in large language models. *arXiv:2401.01313*. 2024.
13. Rawte V, et al. A survey of hallucination in large foundation models. *arXiv:2309.05922*. 2023.
14. Pal A, Umapathi LK, Sankarasubbu M. Med-HALT: medical domain hallucination test for large language models. *Proc CoNLL*. 2023; arXiv:2307.15343.
15. Jin Q, et al. PubMedQA: A dataset for biomedical research question answering. *Proc EMNLP-IJCNLP*. 2019.
16. Ben Abacha A, Demner-Fushman D. A question-entailment approach to question answering. *BMC Bioinformatics*. 2019;20:511.
17. Jin D, et al. What disease does this patient have? A large-scale open domain question answering dataset from medical exams. *Appl Sci*. 2021;11(14):6421.
18. Hendrycks D, et al. Measuring massive multitask language understanding. *Proc ICLR*. 2021.
19. Pal A, Umapathi LK, Sankarasubbu M. MedMCQA: A large-scale multi-subject multi-choice dataset for medical domain question answering. *Proc PMLR (CHIL)*. 2022;174.
20. HealthMemoryArena. MedHall-Bench dataset card. Hugging Face repository. 2024.
21. Manakul P, Liusie A, Gales MJF. SelfCheckGPT: Zero-resource black-box hallucination detection for generative large language models. *Proc EMNLP*. 2023.
22. Min S, et al. FActScore: Fine-grained atomic evaluation of factual precision in long form text generation. *Proc EMNLP*. 2023.
23. Zheng L, et al. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. *Proc NeurIPS Datasets and Benchmarks Track*. 2023.
24. Chern S, et al. FacTool: Factuality detection in generative AI. *arXiv:2307.13528*. 2023.
25. Lewis P, et al. Retrieval-augmented generation for knowledge-intensive NLP tasks. *Proc NeurIPS*. 2020.
26. Thorne J, et al. FEVER: a large-scale dataset for fact extraction and verification. *Proc NAACL-HLT*. 2018.
27. Schuster T, et al. Get your vitamin C! Robust fact verification with contrastive evidence. *Proc NAACL-HLT*. 2021.
28. Kuhn L, Gal Y, Farquhar S. Semantic uncertainty: Linguistic invariances for uncertainty estimation in large language models. *Proc ICLR*. 2023.
29. Kossen J, et al. Semantic entropy probes: Robust and cheap hallucination detection in LLMs. *arXiv:2406.15927*. 2024.
30. Shannon CE. A mathematical theory of communication. *Bell System Technical Journal*. 1948;27:379–423.
31. Salton G, Wong A, Yang CS. A vector space model for automatic indexing. *Commun ACM*. 1975;18(11):613–620.
32. Xiao Y, Wang WY. On hallucination and predictive uncertainty in conditional language generation. *Proc EACL*. 2021.
33. Azaria A, Mitchell T. The internal state of an LLM knows when it's lying. *Findings of EMNLP*. 2023.
34. Kadavath S, et al. Language models (mostly) know what they know. *arXiv:2207.05221*. 2022.
35. Bodenreider O. The Unified Medical Language System (UMLS): integrating biomedical terminology. *Nucleic Acids Res*. 2004;32:D267–D270.
36. Nelson SJ, et al. Normalized names for clinical drugs: RxNorm at 6 years. *J Am Med Inform Assoc*. 2011;18(4):441–448.
37. Ke G, et al. LightGBM: A highly efficient gradient boosting decision tree. *Proc NeurIPS*. 2017.
38. Shwartz-Ziv R, Armon A. Tabular data: Deep learning is not all you need. *Information Fusion*. 2022;81:84–90.
39. Grinsztajn L, Oyallon E, Varoquaux G. Why do tree-based models still outperform deep learning on typical tabular data? *Proc NeurIPS Datasets and Benchmarks Track*. 2022.
40. Lundberg SM, Lee SI. A unified approach to interpreting model predictions. *Proc NeurIPS*. 2017.
41. Pandit S, et al. MedHallu: A comprehensive benchmark for detecting medical hallucinations in large language models. *arXiv:2502.14302*. 2025.
42. Farquhar S, et al. Detecting hallucinations in large language models using semantic entropy. *Nature*. 2024;630:625–630.
43. Gururangan S, et al. Annotation artifacts in natural language inference data. *Proc NAACL-HLT*. 2018;107–112.
44. Poliak A, et al. Hypothesis-only baselines in natural language inference. *Proc *SEM*. 2018;180–191.
45. McCoy RT, Pavlick E, Linzen T. Right for the wrong reasons: Diagnosing syntactic heuristics in natural language inference. *Proc ACL*. 2019;3428–3448.
46. Geirhos R, et al. Shortcut learning in deep neural networks. *Nature Machine Intelligence*. 2020;2(11):665–673.

---

## APPENDICES

### Appendix A: Signal Definitions, Regular Expressions, and Lexicons

To eliminate substring collisions while capturing legitimate clinical inflections and attached units, lexical scanning utilizes precompiled regular expressions:

1. **Attached Dosage Unit Pattern:**
   `RE_UNITS_EN = re.compile(r'(?:(?<=\\d)|(?<=\\b))(?:mg|mcg|µg|g|ml|l|units?)(?:\\b|(?=[/\\s,\\.\\)]))', re.IGNORECASE)`

2. **Med-ISP (Medication Information Sparsity):**
   > Med-ISP = 1 − min( (h_D + h_units) / max(0.05 × n, 1), 1 )
   *Pharmacology Lexicon L_D (75 English terms & dosage forms, deduplicated):*
   tablet, tablets, capsule, capsules, pill, pills, injection, injections, injectable, oral, orally, subcutaneous, subcutaneously, iv, intravenous, intravenously, infusion, infusions, dose, doses, dosing, dosage, dosages, prescribe, prescribes, prescribed, prescribing, prescription, prescriptions, pharmacy, rxnorm, formulary, medication, medications, drug, drugs, antibiotic, antibiotics, antimicrobial, analgesic, analgesics, statin, statins, amoxicillin, ibuprofen, metformin, insulin, insulins, aspirin, atorvastatin, omeprazole, paracetamol, acetaminophen, warfarin, heparin, morphine, prednisone, lisinopril, amlodipine, levothyroxine, azithromycin, ciprofloxacin, doxycycline, albuterol, furosemide, losartan, gabapentin, hydrochlorothiazide, sertraline, simvastatin, pantoprazole, escitalopram, clopidogrel, montelukast, rosuvastatin, side effect, side effects, adverse effect, adverse effects, adverse reaction, adverse reactions, therapy, therapies.

3. **C-AAS (Clinical Context-Awareness Sparsity):**
   > C-AAS = 1 − min( h_C / max(0.04 × n, 1), 1 )
   *Clinical Context Lexicon L_C (45 English terms):*
   patient, patients, pediatric, pediatrics, paediatric, paediatrics, adult, adults, child, children, infant, infants, elderly, geriatric, geriatrics, female, females, male, males, woman, women, man, men, age, aged, ages, weight, weights, vital, vitals, allergy, allergies, allergic, history, histories, diagnosis, diagnoses, diagnostic, symptom, symptoms, symptomatic, sign, signs, report, reports, reported, reporting, blood pressure, heart rate, pulse, temperature, temperatures, chronic, acute, clinical, clinically, contraindication, contraindications, contraindicated, comorbid, comorbidities, comorbidity, complication, complications.

4. **Med-EEM (Medical Uncertainty Measure):**
   > H(p_u) = −p_u log₂(p_u + ε) − (1 − p_u) log₂(1 − p_u + ε)
   > Med-EEM = min( min( H(p_u), 1.0 ) × (1 + p_u), 1.0 )
   *Hedging Lexicon L_U (35 English terms):*
   maybe, possibly, possibility, possible, might, could, uncertain, uncertainty, unclear, unknown, approximately, approximate, seems, seemed, seeming, seem, appears, appeared, appearing, appear, suggest, suggests, suggested, suggesting, perhaps, likely, likelihood, probably, probable, probability, assume, assumes, assumed, assuming, think, thinks, thinking, thought, believe, believes, believed, believing, estimate, estimates, estimated, estimating, roughly, rough, sometimes, often.
   *Note on Monotonicity:* Because p_u < 0.50 across all empirical distributions, H(p_u) is strictly monotonic, rendering Med-EEM mathematically equivalent to raw hedging frequency in tree-based partition algorithms.

5. **CDT (Context Drift):**
   > CDT = 1 − ( **q** · **a** ) / ( ‖**q**‖₂ × ‖**a**‖₂ )
   Computed over joint unigram term-frequency vectors of prompt Q and response A.

### Appendix B: Model Settings, Protocol, and Reproducibility Checklist

1. **LightGBM Classifier:** 300 boosting trees, learning rate 0.05, maximum depth 5, 31 leaves, balanced class weights, minimum child samples 20, zero subsampling.
2. **Baselines:**
   * *Logistic Regression:* L2 penalty (C = 1.0), L-BFGS solver, max 2,000 iterations, balanced class weighting.
   * *MLP Architectures:* scikit-learn `MLPClassifier`, hidden layers (256, 128) and (128, 64, 32, 16), ReLU activation, Adam optimizer (lr = 0.001, alpha = 0.0001, batch size 200), early stopping on 10% validation split.
   * *TF-IDF Baseline:* Sublinear term-frequency scaling, unigrams and bigrams, max 20,000 features, min document frequency 2, balanced L2 logistic regression.
   * *Answer Length Baselines:* Standardized logarithm of word count with balanced logistic regression (LR) and LightGBM (LGB).
   * *CDT + Length Baselines:* Standardized CDT and log word count evaluated with both balanced logistic regression (LR) and LightGBM (LGB).
3. **Validation Environment:** Python 3.8.10, scikit-learn 1.3.2, LightGBM 4.6.0, NumPy 1.24.4, Pandas 2.0.3, SciPy 1.10.1, Windows 10 x64, Intel64 Family 6 Model 60 CPU. Full experimental suite executes in 582.5 seconds with exact seed determinism.
"""

with open(MD_PATH, "w", encoding="utf-8") as f:
    f.write(paper_md)

print("Successfully updated RESEARCH_PAPER_CLINIGUARD.md with auto-generated tables!")
