"""
verify_existing_datasets.py
----------------------------
Audits and verifies all existing local datasets in F:\\cliniguard\\data_extraction.
Performs schema inspection, row counts, label distributions, duplicate question counts,
and data integrity checks without modifying or deleting any existing files.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data_extraction"
AUDIT_DIR = PROJECT_ROOT / "Dataset_Source_and_Reproducibility_Audit"
REPORTS_DIR = AUDIT_DIR / "reports"

DATASET_CONFIGS = [
    {
        "name": "Med-HALT",
        "folder": "Med-HALT",
        "file": "medhalt.parquet",
        "repo": "openlifescienceai/Med-HALT",
        "task": "Biomedical Hallucination Benchmark",
        "has_native_labels": True,
        "paper_role": "Primary Benchmark 1 (Included)",
    },
    {
        "name": "MedHallu",
        "folder": "MedHallu",
        "file": "medhallu.parquet",
        "repo": "FreedomIntelligence/MedHallu",
        "task": "Clinical Question Answering Hallucination",
        "has_native_labels": True,
        "paper_role": "Primary Benchmark 2 (Included)",
    },
    {
        "name": "PubMedQA",
        "folder": "pubmedqa",
        "file": "pubmedqa.parquet",
        "repo": "pubmed_qa (pqa_labeled)",
        "task": "Biomedical Literature QA",
        "has_native_labels": False,
        "paper_role": "Excluded (No native hallucination annotations)",
    },
    {
        "name": "MedQuAD",
        "folder": "medquad",
        "file": "medquad.parquet",
        "repo": "lavita/MedQuAD (NIH)",
        "task": "Consumer Health Question Answering",
        "has_native_labels": False,
        "paper_role": "Secondary (Pillar 2 Synthetic Topic Mismatch Check)",
    },
    {
        "name": "MedQuAD XML",
        "folder": "github",
        "file": "github.parquet",
        "repo": "abachaa/MedQuAD (GitHub)",
        "task": "NIH XML-Parsed QA",
        "has_native_labels": False,
        "paper_role": "Excluded (Subset duplicate of MedQuAD)",
    },
    {
        "name": "MedHall-Bench",
        "folder": "medhall_bench",
        "file": "medhall_bench.parquet",
        "repo": "MedHall-Bench",
        "task": "Agent Evaluation Rubric Harness",
        "has_native_labels": False,
        "paper_role": "Excluded (Rubric prompts without paired model outputs)",
    },
]

def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def run_audit():
    print("=" * 80)
    print("CLINIGUARD LOCAL DATASET VERIFICATION AUDIT")
    print(f"Data Directory: {DATA_DIR}")
    print("=" * 80)

    inventory_records = []
    metadata_records = {}

    for cfg in DATASET_CONFIGS:
        file_path = DATA_DIR / cfg["folder"] / cfg["file"]
        print(f"\nAuditing [{cfg['name']}] -> {file_path.name}")
        
        if not file_path.exists():
            print(f"  [ERROR] File missing: {file_path}")
            continue

        size_bytes = file_path.stat().st_size
        sha256_hash = compute_sha256(file_path)

        df = pd.read_parquet(file_path)
        total_rows = len(df)
        cols = list(df.columns)
        
        # Label distribution
        label_counts = df["label"].value_counts().to_dict() if "label" in df.columns else {}
        label_0 = label_counts.get(0, 0)
        label_1 = label_counts.get(1, 0)
        
        # Unique questions
        unique_q = df["question"].nunique() if "question" in df.columns else 0
        dup_q = total_rows - unique_q

        # Missing values
        null_counts = df.isnull().sum().to_dict()
        has_nulls = any(v > 0 for v in null_counts.values())

        print(f"  Rows: {total_rows:,} | Cols: {cols}")
        print(f"  Labels: 0={label_0:,} ({label_0/total_rows*100:.1f}%), 1={label_1:,} ({label_1/total_rows*100:.1f}%)")
        print(f"  Unique Questions: {unique_q:,} | Duplicate Questions: {dup_q:,}")
        print(f"  File Size: {size_bytes:,} bytes | SHA256: {sha256_hash[:12]}...")

        inventory_records.append({
            "Dataset_Name": cfg["name"],
            "Local_Path": str(file_path.relative_to(PROJECT_ROOT)),
            "File_Format": "parquet",
            "File_Size_Bytes": size_bytes,
            "SHA256_Prefix": sha256_hash[:16],
            "Total_Rows": total_rows,
            "Label_0_Rows": label_0,
            "Label_1_Rows": label_1,
            "Unique_Questions": unique_q,
            "Duplicate_Questions": dup_q,
            "Has_Native_Hallucination_Labels": cfg["has_native_labels"],
            "Role_in_Project": cfg["paper_role"],
            "Verification_Status": "VERIFIED_GENUINE" if cfg["has_native_labels"] else "VERIFIED_TASK_RESTRICTED"
        })

        metadata_records[cfg["name"]] = {
            "name": cfg["name"],
            "local_path": str(file_path.relative_to(PROJECT_ROOT)),
            "source_repo": cfg["repo"],
            "task_type": cfg["task"],
            "file_size_bytes": size_bytes,
            "sha256": sha256_hash,
            "total_rows": total_rows,
            "columns": cols,
            "label_distribution": label_counts,
            "unique_questions": unique_q,
            "has_null_values": has_nulls,
            "null_breakdown": null_counts,
            "native_hallucination_annotations": cfg["has_native_labels"],
            "project_role": cfg["paper_role"]
        }

    # Save inventory CSV
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    inv_df = pd.DataFrame(inventory_records)
    inv_csv_path = REPORTS_DIR / "existing_dataset_inventory.csv"
    inv_df.to_csv(inv_csv_path, index=False)
    print(f"\n[OK] Saved existing dataset inventory to: {inv_csv_path}")

    # Save metadata JSON
    meta_json_path = REPORTS_DIR / "dataset_metadata.json"
    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata_records, f, indent=2)
    print(f"[OK] Saved metadata JSON to: {meta_json_path}")

    return inv_df, metadata_records

if __name__ == "__main__":
    run_audit()
