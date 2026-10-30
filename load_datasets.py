"""
CLINIGUARD - Dataset Extraction & Audit Pipeline
-------------------------------------------------
This script manages dataset extraction for CLINIGUARD.
Crucially, it rigorously audits and separates:
  1. GENUINE medical hallucination datasets (Med-HALT, MedHallu)
  2. UNSUITABLE / SYNTHETIC sources that lack genuine hallucination annotations
     (PubMedQA, MedQuAD, MedQuAD XML, MedHall-Bench)

Only genuine hallucination datasets are used for valid scientific evaluation.
"""

import os
import sys
from pathlib import Path
import pandas as pd
from datasets import load_dataset

BASE_EXTRACT_DIR = Path(__file__).parent / "data_extraction"


def section(title: str) -> None:
    safe_title = title.replace("\u2011", "-").encode("utf-8", errors="ignore").decode("utf-8")
    print("\n" + "=" * 78)
    print(f"{safe_title}")
    print("=" * 78 + "\n")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_str(v) -> str:
    return "" if pd.isna(v) else str(v).strip()


def write_parquet(df: pd.DataFrame, out_dir: Path, name: str) -> None:
    ensure_dir(out_dir)
    out_path = out_dir / f"{name}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[OK] {len(df)} rows saved -> {out_path} (labels: {dict(df['label'].value_counts())})\n")


# ==============================================================================
# GENUINE DATASET 1: Med-HALT
# ==============================================================================
def extract_medhalt(base_dir: Path) -> pd.DataFrame:
    """Med-HALT (CoNLL 2023) - Genuine medical domain hallucination benchmark.

    Contains genuine PubMed abstracts (label 0: real_data) vs LLM-hallucinated
    fake biomedical abstracts (label 1: fake_data). Total: 4,916 examples.
    """
    section("GENUINE SOURCE 1: HuggingFace openlifescienceai/Med-HALT (IR_abstract2pubmedlink)")
    try:
        ds = load_dataset("openlifescienceai/Med-HALT", "IR_abstract2pubmedlink")
        split = list(ds.keys())[0]
        raw = ds[split].to_pandas()
        print(f"[OK] Loaded Med-HALT from HuggingFace - raw shape: {raw.shape}")

        rows = []
        for _, r in raw.iterrows():
            pubmed_type = str(r.get("pubmed_data_type", "")).strip()
            if pubmed_type not in ("real_data", "fake_data"):
                continue
            rows.append({
                "question": safe_str(r.get("Title", "")),
                "context": safe_str(r.get("source_abstract", "")),
                "answer": safe_str(r.get("Abstract", "")),
                "label": 1 if pubmed_type == "fake_data" else 0,
            })
        df = pd.DataFrame(rows)
        # Filter empty
        df = df[df["answer"].str.len() > 10].reset_index(drop=True)
        write_parquet(df, base_dir / "Med-HALT", "medhalt")
        return df
    except Exception as e:
        print(f"[WARNING] Could not load Med-HALT: {e}")
        local_p = base_dir / "Med-HALT" / "medhalt.parquet"
        if local_p.exists():
            print(f"[INFO] Using existing local parquet: {local_p}")
            return pd.read_parquet(local_p)
        raise e


# ==============================================================================
# GENUINE DATASET 2: MedHallu
# ==============================================================================
def extract_medhallu(base_dir: Path) -> pd.DataFrame:
    """MedHallu (2025) - Expert-annotated medical hallucination benchmark.

    Contains real medical questions from clinical examinations with paired
    Ground Truth answers (label 0) and LLM-generated Hallucinated answers (label 1).
    Total: 1,000 question pairs = 2,000 samples.
    """
    section("GENUINE SOURCE 2: HuggingFace UTAustin-AIHealth/MedHallu (pqa_labeled)")
    try:
        ds = load_dataset("UTAustin-AIHealth/MedHallu", "pqa_labeled")
        split = list(ds.keys())[0]
        raw = ds[split].to_pandas()
        print(f"[OK] Loaded MedHallu from HuggingFace - raw shape: {raw.shape}")

        rows = []
        for _, r in raw.iterrows():
            q = safe_str(r.get("Question", ""))
            k = r.get("Knowledge", None)
            if isinstance(k, (list, tuple)):
                ctx = " ".join([str(x) for x in k])
            elif hasattr(k, "__iter__") and not isinstance(k, str):
                ctx = " ".join([str(x) for x in k])
            else:
                ctx = safe_str(k)
            gt = safe_str(r.get("Ground Truth", ""))
            halluc = safe_str(r.get("Hallucinated Answer", ""))

            if q and gt:
                rows.append({"question": q, "context": ctx, "answer": gt, "label": 0})
            if q and halluc:
                rows.append({"question": q, "context": ctx, "answer": halluc, "label": 1})

        df = pd.DataFrame(rows)
        df = df[df["answer"].str.len() > 10].reset_index(drop=True)
        write_parquet(df, base_dir / "MedHallu", "medhallu")
        return df
    except Exception as e:
        print(f"[WARNING] Could not load MedHallu: {e}")
        local_p = base_dir / "MedHallu" / "medhallu.parquet"
        if local_p.exists():
            print(f"[INFO] Using existing local parquet: {local_p}")
            return pd.read_parquet(local_p)
        raise e


# ==============================================================================
# AUDITED / EXCLUDED SOURCES: Documented why each cannot be used as genuine data
# ==============================================================================
def audit_excluded_sources() -> dict:
    """Documents the audit findings for the four excluded datasets."""
    return {
        "pubmedqa": {
            "name": "PubMedQA (qiaojin/PubMedQA)",
            "status": "EXCLUDED",
            "reason": "Original dataset is biomedical QA with verified research abstracts. "
                      "It has NO hallucination labels. The legacy script artificially replaced "
                      "every 3rd answer with a fixed hedging sentence ('We suggest that it could perhaps...').",
            "genuine_hallucinations": False,
        },
        "medquad": {
            "name": "MedQuAD (lavita/MedQuAD)",
            "status": "EXCLUDED",
            "reason": "Original dataset consists of verified NIH medical QA pairs. "
                      "It has NO hallucination annotations. The legacy script assigned labels as "
                      "(index % 3 == 0) without modifying the authentic text, creating unlearnable/arbitrary labels.",
            "genuine_hallucinations": False,
        },
        "github_xml": {
            "name": "MedQuAD XML (abachaa/MedQuAD on GitHub)",
            "status": "EXCLUDED",
            "reason": "Identical issue to PubMedQA: every 3rd answer from XML files was replaced "
                      "with the exact same synthetic hedging sentence.",
            "genuine_hallucinations": False,
        },
        "medhallbench": {
            "name": "MedHall-Bench (healthmemoryarena/MedHall-Bench)",
            "status": "EXCLUDED",
            "reason": "MedHall-Bench is a test harness providing user prompts and evaluation criteria "
                      "(known_facts, ground_truth_data) to evaluate external LLMs via HolyEval. "
                      "It does NOT contain pre-generated LLM outputs. The legacy loader artificially "
                      "concatenated known_facts as label 0 and hardcoded a single Chinese sentence as label 1.",
            "genuine_hallucinations": False,
        },
    }


def load_genuine_datasets(base_dir: Path = BASE_EXTRACT_DIR) -> pd.DataFrame:
    """Loads and pools ONLY verified genuine medical hallucination datasets.

    Returns a clean DataFrame with schema:
      question, context, answer, label, source
    """
    df_halt = extract_medhalt(base_dir)
    df_halt["source"] = "Med-HALT"

    df_hallu = extract_medhallu(base_dir)
    df_hallu["source"] = "MedHallu"

    combined = pd.concat([df_halt, df_hallu], ignore_index=True)
    print("\n" + "=" * 78)
    print("VERIFIED GENUINE DATASET POOL:")
    print(f"  Med-HALT : {len(df_halt):>5} samples ({df_halt['label'].sum()} hallucinated, {len(df_halt)-df_halt['label'].sum()} real)")
    print(f"  MedHallu : {len(df_hallu):>5} samples ({df_hallu['label'].sum()} hallucinated, {len(df_hallu)-df_hallu['label'].sum()} real)")
    print(f"  TOTAL    : {len(combined):>5} samples ({combined['label'].sum()} hallucinated, {combined['label'].mean()*100:.1f}%)")
    print("=" * 78 + "\n")
    return combined


if __name__ == "__main__":
    print("Running CLINIGUARD Dataset Extraction & Audit...")
    df = load_genuine_datasets()
    audit = audit_excluded_sources()
    print("Excluded Datasets Audit:")
    for k, v in audit.items():
        print(f"  - {v['name']}: {v['status']} -> {v['reason']}")
