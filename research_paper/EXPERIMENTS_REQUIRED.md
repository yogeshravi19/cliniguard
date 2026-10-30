# EXPERIMENTS_REQUIRED.md: Future Empirical Validation Roadmap

This document catalogs the future empirical experiments (marked **X1–X5**) designed to extend CliniGuard beyond the completed retrospective benchmark evaluation (E0–E9).

---

## 1. Status of In-Scope Experiments (E0–E9: COMPLETED)

All core experiments required for the primary research paper have been executed, verified, and saved to `research_paper/results/`:
* [x] **E0: Forensic Dataset Audit** (6 sources audited; 4 synthetic rejected; 2 genuine validated: Med-HALT & MedHallu, 6,914 rows).
* [x] **E1: Headline & Multi-Seed Benchmark** (Seed 42 + 10 seeds; AUROC 0.922 ± 0.005; paired bootstrap significance tests).
* [x] **E2: Held-Out Per-Dataset Performance** (Med-HALT test AUROC 0.976; MedHallu test AUROC 0.683).
* [x] **E3: Question-Grouped 5-Fold Cross-Validation** (AUROC 0.923 ± 0.009; zero question leakage).
* [x] **E4: Zero-Shot Cross-Dataset Domain Transfer** (Demonstrating structural transferability of linguistic signals vs. TF-IDF collapse).
* [x] **E5: Signal Ablation Study** (Quantifying impact of dropping and isolating Med-ISP, C-AAS, Med-EEM, CDT).
* [x] **E6: Univariate Signal Analysis** (Raw AUROC distributions per benchmark).
* [x] **E7: Clinical Task Distribution** (Task-type breakdown across diagnosis, dosing, and allergy).
* [x] **E8: Risk Band Stratification & Probability Calibration** (GREEN 9.9% error rate; RED 88.2% error concentration; Isotonic ECE 1.6%).
* [x] **E9: Latency Profiling** (15.1 ms end-to-end CPU runtime over 2,000 passes).

---

## 2. Recommended Future Experiments (X1–X5)

These experiments require external resources (clinician annotators, commercial LLM API credits, or GPU clusters) and are outlined below for future study expansion.

### Experiment X1: Prospective Real-LLM Benchmark with Clinician Annotation
* **Purpose**: Eliminate dependence on public benchmark test sets by generating real-time responses from 3 frontier LLMs (e.g., GPT-4o, Claude 3.5 Sonnet, Llama-3-70B-Instruct) on 500 patient queries and obtaining dual blinded clinician labels.
* **Command / Script**: *(To write)* `python research_paper/scripts/x1_real_llm_benchmark.py`
* **Required Inputs**: 500 clinical examination queries; API keys for LLM generation; clinician annotation portal.
* **Target Metrics**: Cohen's $\kappa$ / Fleiss' $\kappa$ inter-annotator agreement; AUROC; Brier score; risk-band error rates.
* **Save Location**: `research_paper/results/X1_real_llm_benchmark/`

### Experiment X2: Computational Comparison with GPU-Based External Detectors
* **Purpose**: Directly measure the accuracy-latency tradeoff between CliniGuard and published heavy detectors (SelfCheckGPT [21], FActScore [22], and GPT-4 judge [23]) on identical test prompts.
* **Command / Script**: *(To write)* `python research_paper/scripts/x2_external_detector_benchmark.py`
* **Required Hardware**: 1x NVIDIA A100 / RTX 4090 GPU for SelfCheckGPT multiple generation passes.
* **Target Metrics**: AUROC; Wall-clock latency per query (seconds); Cost per 1,000 queries (USD); GPU memory footprint.
* **Save Location**: `research_paper/results/X2_external_baselines/`

### Experiment X3: UMLS Metathesaurus & RxNorm Ontological Mapping
* **Purpose**: Replace manual string lexicons (`DRUG_TERMS`, `CONTEXT_TERMS`) with comprehensive concept unique identifiers (CUIs) extracted via QuickUMLS or scispaCy from UMLS and RxNorm databases [35], [36].
* **Command / Script**: *(To write)* `python research_paper/scripts/x3_umls_lexicon_enrichment.py`
* **Prerequisites**: UMLS Metathesaurus License from the US National Library of Medicine.
* **Target Metrics**: Comparative AUROC (Manual Lexicon vs. UMLS-Enriched Lexicon); out-of-vocabulary rate.
* **Save Location**: `research_paper/results/X3_umls_mapping/`

### Experiment X4: Nested Cross-Validation Hyperparameter Optimization
* **Purpose**: Perform inner-loop Bayesian or Grid Search hyperparameter optimization for LightGBM (learning rate, tree depth, leave count, min child samples) to remove any potential un-tuned bias.
* **Command / Script**: *(To write)* `python research_paper/scripts/x4_nested_cv_tuning.py`
* **Target Metrics**: Nested CV AUROC vs. default LightGBM AUROC; parameter stability across folds.
* **Save Location**: `research_paper/results/X4_nested_cv/`

### Experiment X5: Multilingual Clinical Dialogue Validation
* **Purpose**: Evaluate CliniGuard's character-level Chinese tokenization module on authentic multilingual clinical dialogue datasets (e.g., CMB-Clin, cMedQA2).
* **Command / Script**: *(To write)* `python research_paper/scripts/x5_multilingual_evaluation.py`
* **Target Metrics**: Chinese-subset AUROC; cross-lingual transferability.
* **Save Location**: `research_paper/results/X5_multilingual/`
