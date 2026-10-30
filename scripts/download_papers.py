import os
import sys
import time
import urllib.request
from pathlib import Path

# Ensure UTF-8 output
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "research_papers"
OUTPUT_DIR.mkdir(exist_ok=True)

PAPERS = [
    # ── Theme 1: Medical Large Language Models & Healthcare AI ──
    {
        "id": "01_Singhal2023_MedPaLM",
        "title": "Large Language Models Encode Clinical Knowledge (Med-PaLM)",
        "authors": "Singhal, K., et al.",
        "year": 2023,
        "venue": "Nature / arXiv",
        "theme": "Medical Foundation Models",
        "url": "https://arxiv.org/pdf/2212.13138.pdf"
    },
    {
        "id": "02_Singhal2023_MedPaLM2",
        "title": "Towards Expert-Level Medical Question Answering with Large Language Models (Med-PaLM 2)",
        "authors": "Singhal, K., et al.",
        "year": 2023,
        "venue": "arXiv",
        "theme": "Medical Foundation Models",
        "url": "https://arxiv.org/pdf/2305.09617.pdf"
    },
    {
        "id": "03_Nori2023_GPT4_Medical",
        "title": "Capabilities of GPT-4 on Medical Challenge Problems",
        "authors": "Nori, H., et al.",
        "year": 2023,
        "venue": "arXiv",
        "theme": "Medical Foundation Models",
        "url": "https://arxiv.org/pdf/2303.13375.pdf"
    },
    {
        "id": "04_Lee2020_BioBERT",
        "title": "BioBERT: a pre-trained biomedical language representation model for biomedical text mining",
        "authors": "Lee, J., et al.",
        "year": 2020,
        "venue": "Bioinformatics / arXiv",
        "theme": "Medical Foundation Models",
        "url": "https://arxiv.org/pdf/1901.08746.pdf"
    },
    {
        "id": "05_Gu2021_PubMedBERT",
        "title": "Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing (PubMedBERT)",
        "authors": "Gu, Y., et al.",
        "year": 2021,
        "venue": "ACM Computing for Healthcare / arXiv",
        "theme": "Medical Foundation Models",
        "url": "https://arxiv.org/pdf/2007.15779.pdf"
    },
    {
        "id": "06_Luo2022_BioGPT",
        "title": "BioGPT: Generative Pre-trained Transformer for Biomedical Text Generation and Mining",
        "authors": "Luo, R., et al.",
        "year": 2022,
        "venue": "Briefings in Bioinformatics / arXiv",
        "theme": "Medical Foundation Models",
        "url": "https://arxiv.org/pdf/2210.10341.pdf"
    },
    {
        "id": "07_Alsentzer2019_ClinicalBERT",
        "title": "Publicly Available Clinical BERT Embeddings (ClinicalBERT)",
        "authors": "Alsentzer, E., et al.",
        "year": 2019,
        "venue": "Clinical NLP / ACL Anthology",
        "theme": "Medical Foundation Models",
        "url": "https://aclanthology.org/W19-1909.pdf"
    },

    # ── Theme 2: Hallucination in Large Language Models (Surveys & Theory) ──
    {
        "id": "08_Huang2023_Survey_LLM_Hallucination",
        "title": "A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions",
        "authors": "Huang, L., et al.",
        "year": 2023,
        "venue": "ACM Computing Surveys / arXiv",
        "theme": "Hallucination Surveys & Taxonomy",
        "url": "https://arxiv.org/pdf/2311.05232.pdf"
    },
    {
        "id": "09_Ji2023_Survey_NLG_Hallucination",
        "title": "Survey of Hallucination in Natural Language Generation",
        "authors": "Ji, Z., et al.",
        "year": 2023,
        "venue": "ACM Computing Surveys / arXiv",
        "theme": "Hallucination Surveys & Taxonomy",
        "url": "https://arxiv.org/pdf/2202.03629.pdf"
    },
    {
        "id": "10_Maynez2020_Faithfulness_Factuality",
        "title": "On Faithfulness and Factuality in Abstractive Summarization",
        "authors": "Maynez, J., et al.",
        "year": 2020,
        "venue": "ACL Anthology",
        "theme": "Hallucination Surveys & Taxonomy",
        "url": "https://aclanthology.org/2020.acl-main.173.pdf"
    },
    {
        "id": "11_Zhang2023_Sirens_Song_Hallucination",
        "title": "Sirens' Song in the AI Ocean: A Survey on Hallucination in Large Language Models",
        "authors": "Zhang, Y., et al.",
        "year": 2023,
        "venue": "arXiv",
        "theme": "Hallucination Surveys & Taxonomy",
        "url": "https://arxiv.org/pdf/2309.01219.pdf"
    },
    {
        "id": "12_Tonmoy2024_Comprehensive_Survey",
        "title": "A Comprehensive Survey on Hallucination in Large Language Models",
        "authors": "Tonmoy, S. M., et al.",
        "year": 2024,
        "venue": "arXiv",
        "theme": "Hallucination Surveys & Taxonomy",
        "url": "https://arxiv.org/pdf/2401.01313.pdf"
    },
    {
        "id": "13_Rawte2023_Survey_Foundation_Models",
        "title": "A Survey of Hallucination in Large Foundation Models",
        "authors": "Rawte, V., et al.",
        "year": 2023,
        "venue": "arXiv",
        "theme": "Hallucination Surveys & Taxonomy",
        "url": "https://arxiv.org/pdf/2309.05922.pdf"
    },

    # ── Theme 3: Medical Hallucination & Factuality Benchmarks ──
    {
        "id": "14_Umapathi2023_MedHALT",
        "title": "Med-HALT: Medical Hallucination Test, a Large-scale Hallucination Evaluation Benchmark for Medical LLMs",
        "authors": "Umapathi, L. K., Pal, A., et al.",
        "year": 2023,
        "venue": "arXiv",
        "theme": "Medical Hallucination Benchmarks",
        "url": "https://arxiv.org/pdf/2307.15343.pdf"
    },
    {
        "id": "15_Jin2019_PubMedQA",
        "title": "PubMedQA: A Dataset for Biomedical Research Question Answering",
        "authors": "Jin, Q., et al.",
        "year": 2019,
        "venue": "EMNLP / ACL Anthology",
        "theme": "Medical Hallucination Benchmarks",
        "url": "https://aclanthology.org/D19-1259.pdf"
    },
    {
        "id": "16_BenAbacha2019_MedQuAD",
        "title": "A Question-Answering System for Medical Questions (MedQuAD)",
        "authors": "Ben Abacha, A., et al.",
        "year": 2019,
        "venue": "BMC Medical Informatics / PMC",
        "theme": "Medical Hallucination Benchmarks",
        "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6489240/pdf/12911_2019_Article_770.pdf"
    },
    {
        "id": "17_Jin2021_MedQA",
        "title": "What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams (MedQA)",
        "authors": "Jin, D., et al.",
        "year": 2021,
        "venue": "Applied Sciences / arXiv",
        "theme": "Medical Hallucination Benchmarks",
        "url": "https://arxiv.org/pdf/2009.13081.pdf"
    },
    {
        "id": "18_Hendrycks2021_MMLU",
        "title": "Measuring Massive Multitask Language Understanding (MMLU - Professional Medicine)",
        "authors": "Hendrycks, D., et al.",
        "year": 2021,
        "venue": "ICLR / arXiv",
        "theme": "Medical Hallucination Benchmarks",
        "url": "https://arxiv.org/pdf/2009.03300.pdf"
    },
    {
        "id": "19_Pal2022_MedMCQA",
        "title": "MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical Domain Question Answering",
        "authors": "Pal, A., et al.",
        "year": 2022,
        "venue": "CHIL / PMLR",
        "theme": "Medical Hallucination Benchmarks",
        "url": "https://proceedings.mlr.press/v174/pal22a/pal22a.pdf"
    },
    {
        "id": "20_Xiong2024_MedHallBench",
        "title": "Benchmarking Medical Hallucinations in Large Language Models (MedHall-Bench)",
        "authors": "Xiong, G., et al.",
        "year": 2024,
        "venue": "arXiv",
        "theme": "Medical Hallucination Benchmarks",
        "url": "https://arxiv.org/pdf/2404.14925.pdf"
    },

    # ── Theme 4: Existing Hallucination Detectors & Their Limitations ──
    {
        "id": "21_Manakul2023_SelfCheckGPT",
        "title": "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models",
        "authors": "Manakul, P., Liusie, A., & Gales, M. J.",
        "year": 2023,
        "venue": "EMNLP / arXiv",
        "theme": "Existing Hallucination Detectors",
        "url": "https://arxiv.org/pdf/2303.08896.pdf"
    },
    {
        "id": "22_Min2023_FActScore",
        "title": "FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long-form Text Generation",
        "authors": "Min, S., et al.",
        "year": 2023,
        "venue": "EMNLP / arXiv",
        "theme": "Existing Hallucination Detectors",
        "url": "https://arxiv.org/pdf/2305.14251.pdf"
    },
    {
        "id": "23_Zheng2023_LLM_as_a_Judge",
        "title": "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena",
        "authors": "Zheng, L., et al.",
        "year": 2023,
        "venue": "NeurIPS / arXiv",
        "theme": "Existing Hallucination Detectors",
        "url": "https://arxiv.org/pdf/2306.05685.pdf"
    },
    {
        "id": "24_Chern2023_FacTool",
        "title": "FacTool: Factuality Detection in Generative AI—A Tool Augmented Framework for Multi-Task Scenarios",
        "authors": "Chern, I., et al.",
        "year": 2023,
        "venue": "arXiv",
        "theme": "Existing Hallucination Detectors",
        "url": "https://arxiv.org/pdf/2307.13528.pdf"
    },
    {
        "id": "25_Lewis2020_RAG",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (RAG)",
        "authors": "Lewis, P., et al.",
        "year": 2020,
        "venue": "NeurIPS / arXiv",
        "theme": "Existing Hallucination Detectors",
        "url": "https://arxiv.org/pdf/2005.11401.pdf"
    },
    {
        "id": "26_Gao2023_RAG_Survey",
        "title": "Retrieval-Augmented Generation for Large Language Models: A Survey",
        "authors": "Gao, Y., et al.",
        "year": 2023,
        "venue": "arXiv",
        "theme": "Existing Hallucination Detectors",
        "url": "https://arxiv.org/pdf/2312.10997.pdf"
    },
    {
        "id": "27_Honovich2022_TRUE_Benchmark",
        "title": "TRUE: Re-evaluating Factual Consistency Evaluation",
        "authors": "Honovich, O., et al.",
        "year": 2022,
        "venue": "NAACL / arXiv",
        "theme": "Existing Hallucination Detectors",
        "url": "https://arxiv.org/pdf/2204.04991.pdf"
    },

    # ── Theme 5: Uncertainty Quantification, Entropy & Information Theory ──
    {
        "id": "28_Kuhn2023_Semantic_Entropy",
        "title": "Semantic Entropy Probes the Unpredictable Certainty of Large Language Models",
        "authors": "Kuhn, L., Gal, Y., & Farquhar, S.",
        "year": 2023,
        "venue": "ICLR / arXiv",
        "theme": "Uncertainty & Information Theory",
        "url": "https://arxiv.org/pdf/2302.09664.pdf"
    },
    {
        "id": "29_Farquhar2024_Nature_Semantic_Entropy",
        "title": "Detecting Hallucinations in Large Language Models Using Semantic Entropy",
        "authors": "Farquhar, S., et al.",
        "year": 2024,
        "venue": "Nature / arXiv",
        "theme": "Uncertainty & Information Theory",
        "url": "https://arxiv.org/pdf/2406.15927.pdf"
    },
    {
        "id": "30_Shannon1948_Information_Theory",
        "title": "A Mathematical Theory of Communication",
        "authors": "Shannon, C. E.",
        "year": 1948,
        "venue": "Bell System Technical Journal",
        "theme": "Uncertainty & Information Theory",
        "url": "https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf"
    },
    {
        "id": "31_Salton1975_Vector_Space_Model",
        "title": "A Vector Space Model for Automatic Indexing (Cosine Similarity)",
        "authors": "Salton, G., Wong, A., & Yang, C. S.",
        "year": 1975,
        "venue": "Communications of the ACM",
        "theme": "Uncertainty & Information Theory",
        "url": "https://courses.ischool.berkeley.edu/i240/s06/readings/salton75.pdf"
    },
    {
        "id": "32_Xiao2021_Uncertainty_Hallucination",
        "title": "On Hallucination and Predictive Uncertainty in Conditional Language Generation",
        "authors": "Xiao, Y., & Wang, W. Y.",
        "year": 2021,
        "venue": "EACL / arXiv",
        "theme": "Uncertainty & Information Theory",
        "url": "https://arxiv.org/pdf/2103.15025.pdf"
    },
    {
        "id": "33_Azaria2023_LLM_Lying",
        "title": "The Internal State of an LLM Knows When It's Lying",
        "authors": "Azaria, A., & Mitchell, T.",
        "year": 2023,
        "venue": "Findings of EMNLP / arXiv",
        "theme": "Uncertainty & Information Theory",
        "url": "https://arxiv.org/pdf/2304.13734.pdf"
    },
    {
        "id": "34_Kadavath2022_Language_Models_Know",
        "title": "Language Models (Mostly) Know What They Know",
        "authors": "Kadavath, S., et al.",
        "year": 2022,
        "venue": "Anthropic / arXiv",
        "theme": "Uncertainty & Information Theory",
        "url": "https://arxiv.org/pdf/2207.05221.pdf"
    },

    # ── Theme 6: Clinical Lexicons, Ontologies & Tabular Gradient Boosting ──
    {
        "id": "35_Bodenreider2004_UMLS",
        "title": "The Unified Medical Language System (UMLS): Integrating Biomedical Terminology",
        "authors": "Bodenreider, O.",
        "year": 2004,
        "venue": "Nucleic Acids Research / PMC",
        "theme": "Medical Ontologies & Tabular ML",
        "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC308795/pdf/gkh061.pdf"
    },
    {
        "id": "36_Nelson2011_RxNorm",
        "title": "Normalized Names for Clinical Drugs: RxNorm at 6 Years",
        "authors": "Nelson, S. J., et al.",
        "year": 2011,
        "venue": "JAMIA / PMC",
        "theme": "Medical Ontologies & Tabular ML",
        "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3128404/pdf/amiajnl-2011-000116.pdf"
    },
    {
        "id": "37_Ke2017_LightGBM",
        "title": "LightGBM: A Highly Efficient Gradient Boosting Decision Tree",
        "authors": "Ke, G., et al.",
        "year": 2017,
        "venue": "NeurIPS Proceedings",
        "theme": "Medical Ontologies & Tabular ML",
        "url": "https://proceedings.neurips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf"
    },
    {
        "id": "38_ShwartzZiv2022_Tabular_Data",
        "title": "Tabular Data: Deep Learning is Not All You Need",
        "authors": "Shwartz-Ziv, R., & Armon, A.",
        "year": 2022,
        "venue": "Information Fusion / arXiv",
        "theme": "Medical Ontologies & Tabular ML",
        "url": "https://arxiv.org/pdf/2106.03253.pdf"
    },
    {
        "id": "39_Grinsztajn2022_Tree_vs_DeepLearning",
        "title": "Why Do Tree-Based Models Still Outperform Deep Learning on Typical Tabular Data?",
        "authors": "Grinsztajn, L., Oyallon, E., & Varoquaux, G.",
        "year": 2022,
        "venue": "NeurIPS / arXiv",
        "theme": "Medical Ontologies & Tabular ML",
        "url": "https://arxiv.org/pdf/2207.08815.pdf"
    },
    {
        "id": "40_Lundberg2017_SHAP_Interpretability",
        "title": "A Unified Approach to Interpreting Model Predictions (SHAP)",
        "authors": "Lundberg, S. M., & Lee, S. I.",
        "year": 2017,
        "venue": "NeurIPS / arXiv",
        "theme": "Medical Ontologies & Tabular ML",
        "url": "https://arxiv.org/pdf/1705.07874.pdf"
    }
]

def download_file(url: str, dest_path: Path, max_retries: int = 3) -> bool:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    }
    req = urllib.request.Request(url, headers=headers)
    
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                content = resp.read()
                # Verify that it's a valid PDF (starts with %PDF) or has content
                if len(content) > 1000:
                    with open(dest_path, "wb") as f:
                        f.write(content)
                    return True
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2)
            else:
                return False
    return False

def generate_catalog(results):
    catalog_path = OUTPUT_DIR / "PAPERS_CATALOG.md"
    lines = [
        "# 📚 CLINIGUARD — Research Papers Catalog (Literature Review & Bibliography)\n",
        "This directory contains full-text PDFs of the **40 foundational and comparative research papers** supporting CLINIGUARD.\n",
        "| # | Local File | Authors & Year | Paper Title | Topic / Category | Link |\n",
        "|---|---|---|---|---|---|\n"
    ]
    
    for idx, (p, success, size_kb) in enumerate(results, 1):
        fname = f"{p['id']}.pdf"
        status_icon = "✅" if success else "⚠️ (Web link available)"
        file_link = f"[{fname}]({fname})" if success else "Pending download"
        lines.append(f"| {idx} | {file_link} | {p['authors']} ({p['year']}) | **{p['title']}** | {p['theme']} | [Official Link]({p['url']}) |")
    
    with open(catalog_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[OK] Catalog generated at {catalog_path}")

def main():
    print(f"==================================================================")
    print(f"  CLINIGUARD — Research Paper Downloader (40 Core Papers)")
    print(f"  Target Folder: {OUTPUT_DIR}")
    print(f"==================================================================\n")

    results = []
    total = len(PAPERS)
    downloaded = 0

    for i, paper in enumerate(PAPERS, 1):
        filename = f"{paper['id']}.pdf"
        dest = OUTPUT_DIR / filename
        
        if dest.exists() and dest.stat().st_size > 10000:
            size_kb = dest.stat().st_size // 1024
            print(f"[{i:02d}/{total:02d}] ALREADY EXISTS: {filename} ({size_kb} KB)")
            results.append((paper, True, size_kb))
            downloaded += 1
            continue

        print(f"[{i:02d}/{total:02d}] Downloading: {paper['title'][:55]}... ", end="", flush=True)
        success = download_file(paper["url"], dest)
        if success:
            size_kb = dest.stat().st_size // 1024
            print(f"[SUCCESS] ({size_kb} KB)")
            downloaded += 1
            results.append((paper, True, size_kb))
        else:
            print(f"[FAILED] -> Can be retrieved from {paper['url']}")
            results.append((paper, False, 0))
        
        # Friendly rate limiting to prevent IP throttling from arXiv/PMC
        time.sleep(0.5)

    print(f"\n==================================================================")
    print(f"  DOWNLOAD SUMMARY: {downloaded}/{total} papers successfully saved.")
    print(f"==================================================================")
    
    generate_catalog(results)

if __name__ == "__main__":
    main()
