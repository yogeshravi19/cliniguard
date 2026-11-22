"""
Dataset Verification Script for CLINIGUARD
"""
import os, sys
from pathlib import Path
import pandas as pd

def run_dataset_verification():
    print("=" * 80)
    print("CLINIGUARD DATASET VERIFICATION & CORPUS INVENTORY")
    print("=" * 80)
    
    root = Path(__file__).resolve().parent.parent.parent
    
    # Check local extractions
    data_dir = root / "data_extraction"
    sources = ["Med-HALT", "MedHallu", "medquad", "pubmedqa", "github", "medhall_bench"]
    
    print("\n[1] Local Extraction Directory Audit:")
    for s in sources:
        p_dir = data_dir / s
        if p_dir.exists():
            parquets = list(p_dir.glob("*.parquet"))
            for p in parquets:
                df = pd.read_parquet(p)
                labels = df["label"].value_counts().to_dict() if "label" in df.columns else "no label col"
                print(f"  - {s:<15} | File: {p.name:<22} | Rows: {len(df):<6} | Labels: {labels}")
        else:
            print(f"  - {s:<15} | MISSING")
            
    # Check cliniguard_all_datasets.csv
    all_csv = root / "cliniguard_all_datasets.csv"
    if all_csv.exists():
        df_all = pd.read_csv(all_csv)
        print(f"\n[2] Legacy Mixture 'cliniguard_all_datasets.csv':")
        print(f"  Total Rows: {len(df_all)}")
        print(f"  Columns: {list(df_all.columns)}")
        if "source_dataset" in df_all.columns:
            print("  Source Dataset Composition:")
            print(df_all["source_dataset"].value_counts().to_string())
        if "label" in df_all.columns:
            print("  Label Counts:", df_all["label"].value_counts().to_dict())

    # Check 10 Datasets Catalog
    cat_csv = Path(__file__).resolve().parent.parent / "dataset_sources.csv"
    if cat_csv.exists():
        df_cat = pd.read_csv(cat_csv)
        print(f"\n[3] 10 Curated Medical Datasets Catalog ({len(df_cat)} datasets):")
        for _, r in df_cat.iterrows():
            print(f"  {r['id']}: {r['official_name']} | Repo: {r['repo_id_or_url']} | Status: {r['status']}")

if __name__ == "__main__":
    run_dataset_verification()
