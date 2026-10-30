# Verified Bibliography – CliniGuard manuscript

Numbering is identical to the in-text numbers in `RESEARCH_PAPER_CLINIGUARD.md`.
Items 1–40 keep the numbering of `research_papers/PAPERS_CATALOG.md`; 41–42 are additions.

**Status codes**
- **V-PDF** – first page of the downloaded PDF was inspected and matches (see `pdf_first_page_dump.txt`).
- **V-MEM** – real publication per the author's knowledge; PDF not opened/valid locally. Check DOI/venue before submission.
- **FIXED** – the earlier draft's citation was wrong; corrected here.

| # | Reference | Status | Used to support |
|---|---|---|---|
| 1 | Singhal K. et al. Large language models encode clinical knowledge. *Nature* 620, 172–180 (2023). | V-PDF | LLMs reach strong medical-exam accuracy |
| 2 | Singhal K. et al. Towards expert-level medical question answering with large language models. arXiv:2305.09617 (2023). | V-PDF | Med-PaLM 2 |
| 3 | Nori H. et al. Capabilities of GPT-4 on medical challenge problems. arXiv:2303.13375 (2023). | V-PDF | GPT-4 medical-exam performance |
| 4 | Lee J. et al. BioBERT: a pre-trained biomedical language representation model. *Bioinformatics* 36(4):1234–1240 (2020). | V-PDF | Biomedical LMs |
| 5 | Gu Y. et al. Domain-specific language model pretraining for biomedical NLP. *ACM Trans. Comput. Healthcare* 3(1):2 (2021). | V-PDF | Biomedical LMs |
| 6 | Luo R. et al. BioGPT: generative pre-trained transformer for biomedical text. *Brief. Bioinform.* 23(6):bbac409 (2022). | V-PDF | Biomedical generative LMs |
| 7 | Alsentzer E. et al. Publicly available clinical BERT embeddings. *Clinical NLP Workshop* (2019). | V-PDF | Clinical LMs |
| 8 | Huang L. et al. A survey on hallucination in large language models. arXiv:2311.05232 (2023); ACM TOIS (2025). | V-PDF, FIXED venue | Hallucination taxonomy |
| 9 | Ji Z. et al. Survey of hallucination in natural language generation. *ACM Comput. Surv.* 55(12):248 (2023). | V-PDF | Hallucination definition |
| 10 | Maynez J. et al. On faithfulness and factuality in abstractive summarization. *ACL* (2020). | V-PDF | Intrinsic/extrinsic hallucination |
| 11 | Zhang Y. et al. Siren's song in the AI ocean: a survey on hallucination in LLMs. arXiv:2309.01219 (2023). | V-PDF | Survey |
| 12 | Tonmoy S.M.T.I. et al. A comprehensive survey of hallucination mitigation techniques in LLMs. arXiv:2401.01313 (2024). | V-PDF, FIXED title | Mitigation survey |
| 13 | Rawte V. et al. A survey of hallucination in "large" foundation models. arXiv:2309.05922 (2023). | V-PDF | Survey |
| 14 | Pal A., Umapathi L.K., Sankarasubbu M. Med-HALT: medical domain hallucination test for LLMs. *CoNLL* (2023); arXiv:2307.15343. | V-PDF, FIXED authors/title | Med-HALT dataset (used) |
| 15 | Jin Q. et al. PubMedQA: a dataset for biomedical research question answering. *EMNLP-IJCNLP* (2019). | V-PDF | PubMedQA (used as question/answer source) |
| 16 | Ben Abacha A., Demner-Fushman D. A question-entailment approach to question answering. *BMC Bioinformatics* 20:511 (2019); arXiv:1901.08079. | V-PDF, FIXED title/venue | MedQuAD (used) |
| 17 | Jin D. et al. What disease does this patient have? A large-scale open domain question answering dataset from medical exams. *Appl. Sci.* 11(14):6421 (2021). | V-PDF | Medical QA benchmarks |
| 18 | Hendrycks D. et al. Measuring massive multitask language understanding. *ICLR* (2021). | V-PDF | MMLU (general benchmark context) |
| 19 | Pal A., Umapathi L.K., Sankarasubbu M. MedMCQA. *PMLR* 174 (CHIL 2022). | V-PDF | Medical QA benchmarks |
| 20 | MedHall-Bench dataset card, Hugging Face `healthmemoryarena/MedHall-Bench`. | **FIXED** – earlier "Xiong 2024" PDF was an unrelated paper (moved to `_INVALID`) | MedHall-Bench dataset (used). **No journal paper is cited for it.** |
| 21 | Manakul P., Liusie A., Gales M. SelfCheckGPT. *EMNLP* (2023). | V-PDF | Sampling-consistency detection needs multiple generations |
| 22 | Min S. et al. FActScore. *EMNLP* (2023). | V-PDF | Retrieval-based atomic-fact scoring |
| 23 | Zheng L. et al. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. *NeurIPS D&B* (2023). | V-PDF | LLM-judge approach and its biases |
| 24 | Chern I. et al. FacTool. arXiv:2307.13528 (2023). | V-PDF | Tool-augmented factuality detection |
| 25 | Lewis P. et al. Retrieval-augmented generation for knowledge-intensive NLP tasks. *NeurIPS* (2020). | V-PDF | RAG |
| 26 | Gao Y. et al. Retrieval-augmented generation for LLMs: a survey. arXiv:2312.10997 (2023). | V-PDF | RAG survey |
| 27 | Honovich O. et al. TRUE: re-evaluating factual consistency evaluation. *NAACL* (2022). | V-PDF | Factual-consistency evaluation |
| 28 | Kuhn L., Gal Y., Farquhar S. Semantic uncertainty: linguistic invariances for uncertainty estimation in NLG. *ICLR* (2023). | V-PDF, FIXED title | Semantic entropy (model-side) |
| 29 | Kossen J. et al. Semantic entropy probes: robust and cheap hallucination detection in LLMs. arXiv:2406.15927 (2024). | V-PDF, FIXED (earlier labelled Farquhar) | Needs hidden-state access |
| 30 | Shannon C.E. A mathematical theory of communication. *Bell Syst. Tech. J.* 27:379–423 (1948). | V-PDF | Entropy definition (Med-EEM) |
| 31 | Salton G., Wong A., Yang C.S. A vector space model for automatic indexing. *Commun. ACM* 18(11):613–620 (1975). doi:10.1145/361219.361220 | V-MEM (local PDF was the wrong paper; removed) | Cosine similarity of term vectors (CDT) |
| 32 | Xiao Y., Wang W.Y. On hallucination and predictive uncertainty in conditional language generation. *EACL* (2021). | V-PDF | Uncertainty correlates with hallucination |
| 33 | Azaria A., Mitchell T. The internal state of an LLM knows when it's lying. *Findings EMNLP* (2023). | V-PDF | Hidden-state probes |
| 34 | Kadavath S. et al. Language models (mostly) know what they know. arXiv:2207.05221 (2022). | V-PDF | Model self-knowledge |
| 35 | Bodenreider O. The Unified Medical Language System (UMLS). *Nucleic Acids Res.* 32:D267–D270 (2004). | V-MEM (PDF invalid) | Standard terminology; **CliniGuard lexicons are NOT derived from UMLS** |
| 36 | Nelson S.J. et al. Normalized names for clinical drugs: RxNorm at 6 years. *JAMIA* 18(4):441–448 (2011). | V-MEM (PDF invalid) | Standard drug vocabulary; **not used by CliniGuard** |
| 37 | Ke G. et al. LightGBM: a highly efficient gradient boosting decision tree. *NeurIPS* (2017). | V-PDF | LightGBM |
| 38 | Shwartz-Ziv R., Armon A. Tabular data: deep learning is not all you need. *Information Fusion* 81:84–90 (2022). | V-PDF | Trees vs DNN on tabular data (general) |
| 39 | Grinsztajn L., Oyallon E., Varoquaux G. Why do tree-based models still outperform deep learning on typical tabular data? *NeurIPS D&B* (2022). | V-PDF, FIXED title | Trees vs DNN (general) |
| 40 | Lundberg S.M., Lee S.-I. A unified approach to interpreting model predictions. *NeurIPS* (2017). | V-PDF | SHAP (proposed future analysis, **not run**) |
| 41 | Pandit S. et al. MedHallu: a comprehensive benchmark for detecting medical hallucinations in large language models. arXiv:2502.14302 (2025). | V-MEM – verify author list | MedHallu dataset (used) |
| 42 | Farquhar S. et al. Detecting hallucinations in large language models using semantic entropy. *Nature* 630:625–630 (2024). | V-MEM (not downloaded) | Semantic entropy in *Nature* |

## Claims in the earlier draft that no citation supports (removed or reworded)
- Fixed latency/cost figures for LLM-as-judge or SelfCheckGPT ("5–15 s"): removed; stated qualitatively (multiple model calls required [21],[23]).
- "Statistically significant", HIPAA/cost claims, "55,118 rows" website claim: removed.
- Random-Forest/SVM/Naive-Bayes/BERT AUROC rows and "iteration history" AUROCs: no script produced them; removed.
- "Beats state of the art": removed – no external baseline was run (see EXPERIMENTS_REQUIRED.md, X5).
