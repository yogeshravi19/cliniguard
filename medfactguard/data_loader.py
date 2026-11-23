"""
MedFactGuard Harmonized Dataset Loader
Standardizes medical hallucination benchmarks into a unified schema:
    - question: str
    - context: str (clinical evidence or source knowledge)
    - generated_answer: str (the medical claim or LLM generation being evaluated)
    - label: int (0 = Factual / Supported, 1 = Hallucinated / Unsupported)
    - dataset_name: str
    - group_id: str (for leakage-safe question-level or patient-level splitting)
    - annotation_type: str ('human_expert', 'clinical_rag', 'model_hallucination', 'synthetic')
"""

import os
import glob
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import GroupKFold, train_test_split

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data_extraction', 'medfactguard')
os.makedirs(CACHE_DIR, exist_ok=True)

def safe_str(val):
    if pd.isna(val) or val is None:
        return ""
    return str(val).strip()

def load_placebobench():
    """
    PlaceboBench (blue-guardrails/PlaceboBench)
    Pharmaceutical RAG responses (EMA evidence) across 7 LLMs with human-reviewed annotations.
    Total: 482 rows (257 Factual / 225 Hallucinated).
    """
    cache_file = os.path.join(CACHE_DIR, 'placebobench_clean.parquet')
    if os.path.exists(cache_file):
        return pd.read_parquet(cache_file)
    
    print("[MedFactGuard] Loading PlaceboBench from Hugging Face...")
    ds = load_dataset("blue-guardrails/PlaceboBench", split="train")
    raw = ds.to_pandas()
    
    df = pd.DataFrame({
        'question': raw['question'].apply(safe_str),
        'context': raw['retrieved_context'].apply(safe_str),
        'generated_answer': raw['answer'].apply(safe_str),
        'label': raw['hallucinated'].astype(int), # True -> 1 (Hallucinated), False -> 0 (Factual)
        'dataset_name': 'PlaceboBench',
        'group_id': raw['question'].apply(lambda q: safe_str(q)[:60]),
        'annotation_type': 'human_reviewed_rag'
    })
    df.to_parquet(cache_file, index=False)
    print(f"  -> PlaceboBench loaded: {len(df)} rows | Factual: {sum(df['label']==0)}, Hallucinated: {sum(df['label']==1)}")
    return df

def load_medexpert():
    """
    MedExpert (sonal-ssj/MedExpert)
    Expert annotations of medical chatbot responses.
    Total: 572 rows in medexpert-all.
    """
    cache_file = os.path.join(CACHE_DIR, 'medexpert_clean.parquet')
    if os.path.exists(cache_file):
        return pd.read_parquet(cache_file)
    
    print("[MedFactGuard] Loading MedExpert from Hugging Face...")
    ds = load_dataset("sonal-ssj/MedExpert", "medexpert-all", split="train")
    raw = ds.to_pandas()
    
    def extract_fact_label(row):
        sev = str(row.get('highest_fact_severity', 'No Error')).strip().lower()
        if sev not in ('no error', 'none', '', 'nan'):
            return 1
        try:
            n_spans = int(row.get('n_factuality_annotated_spans', 0))
            if n_spans > 0:
                return 1
        except (ValueError, TypeError):
            pass
        return 0

    labels = [extract_fact_label(r) for _, r in raw.iterrows()]
    
    df = pd.DataFrame({
        'question': raw['question'].apply(safe_str),
        'context': raw['references'].apply(safe_str) if 'references' in raw else "",
        'generated_answer': raw['response'].apply(safe_str),
        'label': labels,
        'dataset_name': 'MedExpert',
        'group_id': raw['base_question_id'].fillna(raw['question']).apply(safe_str),
        'annotation_type': 'expert_doctor_chatbot'
    })
    df.to_parquet(cache_file, index=False)
    print(f"  -> MedExpert loaded: {len(df)} rows | Factual: {sum(df['label']==0)}, Hallucinated: {sum(df['label']==1)}")
    return df

def load_medhallu():
    """
    MedHallu (FreedomIntelligence/MedHallu)
    1,000 paired clinical questions (1,000 Doctor truth / 1,000 LLM hallucination).
    Total: 2,000 rows.
    """
    cache_file = os.path.join(CACHE_DIR, 'medhallu_clean.parquet')
    if os.path.exists(cache_file):
        return pd.read_parquet(cache_file)
        
    print("[MedFactGuard] Loading MedHallu...")
    # Check local parquet first
    local_paths = glob.glob('**/medhallu.parquet', recursive=True)
    if local_paths:
        raw = pd.read_parquet(local_paths[0])
        df = pd.DataFrame({
            'question': raw['question'].apply(safe_str),
            'context': raw['context'].apply(safe_str),
            'generated_answer': raw['answer'].apply(safe_str),
            'label': raw['label'].astype(int),
            'dataset_name': 'MedHallu',
            'group_id': raw['question'].apply(safe_str),
            'annotation_type': 'paired_doctor_vs_llm'
        })
    else:
        ds = load_dataset("FreedomIntelligence/MedHallu", "pqa_labeled", split="train")
        raw = ds.to_pandas()
        rows = []
        for _, r in raw.iterrows():
            q = safe_str(r.get('Question', ''))
            ctx = " ".join(r.get('Knowledge', [])) if isinstance(r.get('Knowledge', []), list) else ""
            rows.append({'question': q, 'context': ctx, 'generated_answer': safe_str(r.get('Ground Truth', '')), 'label': 0, 'dataset_name': 'MedHallu', 'group_id': q, 'annotation_type': 'doctor_ground_truth'})
            rows.append({'question': q, 'context': ctx, 'generated_answer': safe_str(r.get('Hallucinated Answer', '')), 'label': 1, 'dataset_name': 'MedHallu', 'group_id': q, 'annotation_type': 'llm_hallucination'})
        df = pd.DataFrame(rows)
        
    df.to_parquet(cache_file, index=False)
    print(f"  -> MedHallu loaded: {len(df)} rows | Factual: {sum(df['label']==0)}, Hallucinated: {sum(df['label']==1)}")
    return df

def load_medhalt():
    """
    Med-HALT (openlifescienceai/Med-HALT)
    4,914 items (2,925 real PubMed abstracts / 1,989 LLM fake abstracts).
    """
    cache_file = os.path.join(CACHE_DIR, 'medhalt_clean.parquet')
    if os.path.exists(cache_file):
        return pd.read_parquet(cache_file)
        
    print("[MedFactGuard] Loading Med-HALT...")
    local_paths = glob.glob('**/medhalt.parquet', recursive=True)
    if local_paths:
        raw = pd.read_parquet(local_paths[0])
        df = pd.DataFrame({
            'question': raw['question'].apply(safe_str),
            'context': raw['context'].apply(safe_str),
            'generated_answer': raw['answer'].apply(safe_str),
            'label': raw['label'].astype(int),
            'dataset_name': 'Med-HALT',
            'group_id': raw['question'].apply(safe_str),
            'annotation_type': 'biomedical_abstract_benchmark'
        })
    else:
        ds = load_dataset('openlifescienceai/Med-HALT', 'IR_abstract2pubmedlink')
        raw = ds[list(ds.keys())[0]].to_pandas()
        df = pd.DataFrame({
            'question': raw['Title'].apply(safe_str),
            'context': raw['source_abstract'].apply(safe_str),
            'generated_answer': raw['Abstract'].apply(safe_str),
            'label': raw['pubmed_data_type'].map({'fake_data': 1, 'real_data': 0}).fillna(0).astype(int),
            'dataset_name': 'Med-HALT',
            'group_id': raw['Title'].apply(safe_str),
            'annotation_type': 'biomedical_abstract_benchmark'
        })
    df.to_parquet(cache_file, index=False)
    print(f"  -> Med-HALT loaded: {len(df)} rows | Factual: {sum(df['label']==0)}, Hallucinated: {sum(df['label']==1)}")
    return df

def load_medhal_sample(n_samples=10000, seed=42):
    """
    MedHal (LAMA-WeST/medhal-lf_qafil_bal)
    Streams and samples n_samples balanced across factual and hallucinated statements.
    """
    cache_file = os.path.join(CACHE_DIR, f'medhal_sample_{n_samples}.parquet')
    if os.path.exists(cache_file):
        return pd.read_parquet(cache_file)
        
    print(f"[MedFactGuard] Sampling {n_samples} balanced records from MedHal (LAMA-WeST/medhal-lf_qafil_bal)...")
    ds = load_dataset('LAMA-WeST/medhal-lf_qafil_bal', split='train', streaming=True)
    
    factual_rows = []
    hallucinated_rows = []
    target_each = n_samples // 2
    
    for item in ds:
        # label == True -> Factual (0); label == False -> Hallucinated (1)
        is_factual = bool(item.get('label', True))
        bin_label = 0 if is_factual else 1
        
        row_dict = {
            'question': safe_str(item.get('explanation', '')[:100]),
            'context': safe_str(item.get('context', '')),
            'generated_answer': safe_str(item.get('statement', '')),
            'label': bin_label,
            'dataset_name': 'MedHal',
            'group_id': safe_str(item.get('id', '')),
            'annotation_type': safe_str(item.get('source', 'medhal'))
        }
        
        if bin_label == 0 and len(factual_rows) < target_each:
            factual_rows.append(row_dict)
        elif bin_label == 1 and len(hallucinated_rows) < target_each:
            hallucinated_rows.append(row_dict)
            
        if len(factual_rows) >= target_each and len(hallucinated_rows) >= target_each:
            break
            
    df = pd.DataFrame(factual_rows + hallucinated_rows)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.to_parquet(cache_file, index=False)
    print(f"  -> MedHal sampled: {len(df)} rows | Factual: {sum(df['label']==0)}, Hallucinated: {sum(df['label']==1)}")
    return df

def load_harmonized_benchmark_suite(include_medhal=True, medhal_n=5000):
    """
    Loads all datasets harmonized into a single multi-source corpus.
    Separates into:
      - Training Pool (Med-HALT, MedHallu, MedHal)
      - External Generalization Test Benchmarks (PlaceboBench, MedExpert)
    """
    print("=== HARMONIZING MULTI-DATASET CORPUS FOR MEDFACTGUARD ===")
    df_placebo = load_placebobench()
    df_expert = load_medexpert()
    df_hallu = load_medhallu()
    df_halt = load_medhalt()
    
    parts = [df_halt, df_hallu, df_placebo, df_expert]
    if include_medhal:
        df_medhal = load_medhal_sample(n_samples=medhal_n)
        parts.append(df_medhal)
        
    df_all = pd.concat(parts, ignore_index=True)
    print(f"\n[PASS] Total Harmonized Corpus: {len(df_all)} records across {len(parts)} benchmarks.")
    print("Dataset Breakdown:")
    for name, group in df_all.groupby('dataset_name'):
        print(f"  - {name:<15}: {len(group):>6} rows (Factual: {sum(group['label']==0):>5}, Hallucinated: {sum(group['label']==1):>5})")
        
    return {
        'full_corpus': df_all,
        'train_pool': df_all[df_all['dataset_name'].isin(['Med-HALT', 'MedHallu', 'MedHal'])].copy(),
        'external_benchmarks': {
            'PlaceboBench': df_placebo,
            'MedExpert': df_expert
        }
    }

if __name__ == '__main__':
    data = load_harmonized_benchmark_suite(include_medhal=True, medhal_n=2000)
    print("\nDataset loader self-test completed successfully!")
