# scripts/process_local_parquet_multi.py
# ---------------------------------------------------------------
# Scan the data_extraction folder for any *.parquet files, load each
# using pandas, call the Cliniguard /predict API for every QA row, and
# write one combined CSV with the results.
# ---------------------------------------------------------------

import sys
from pathlib import Path
import pandas as pd
import requests
from tqdm import tqdm
import csv

# ----- USER‑CONFIGURABLE SETTINGS -----
DATA_DIR   = Path(r"C:/Users/ACER/OneDrive/Desktop/cliniguard/data_extraction")
OUTPUT_CSV = Path(r"C:/Users/ACER/OneDrive/Desktop/cliniguard/cliniguard_parquet_results.csv")
SERVER_URL = "http://localhost:5000/predict"
# --------------------------------------

if not DATA_DIR.exists():
    sys.exit(f"Data folder not found: {DATA_DIR}")

# Find all parquet files under the folder (recursively)
parquet_files = list(DATA_DIR.rglob("*.parquet"))
if not parquet_files:
    sys.exit(f"No parquet files found under {DATA_DIR}")

print(f"Found {len(parquet_files)} parquet file(s):")
for pf in parquet_files:
    print(f"  - {pf}")

# Helper to turn a DataFrame row into the payload expected by Cliniguard
def build_payload_from_row(row):
    # Expected columns: 'question', 'context', 'answers' (list) – fallback to empty strings
    question = row.get('question', '')
    context  = row.get('context', '')
    # 'answers' may be a list; take the first element if possible
    answers  = row.get('answers', [])
    answer   = answers[0] if isinstance(answers, list) and answers else ''
    return {
        "question": question,
        "context":  context,
        "answer":   answer,
    }

# Open the output CSV once and write rows from every parquet file
with OUTPUT_CSV.open('w', newline='', encoding='utf-8') as out_f:
    writer = csv.DictWriter(
        out_f,
        fieldnames=["source_file", "idx", "question", "context", "answer", "risk_score", "label"]
    )
    writer.writeheader()

    for parquet_path in parquet_files:
        print(f"\nLoading {parquet_path} …")
        try:
            df = pd.read_parquet(parquet_path)
        except Exception as e:
            print(f"  ❌ Failed to read {parquet_path}: {e}")
            continue

        # Verify required columns exist
        required = {"question", "context", "answers"}
        if not required.issubset(df.columns):
            print(f"  ⚠️ Missing expected columns in {parquet_path}. Expected at least {required}. Skipping.")
            continue

        rows = []
        for _, row in df.iterrows():
            rows.append(build_payload_from_row(row))

        print(f"  📦 {len(rows)} samples – sending to Cliniguard API …")
        for idx, payload in enumerate(tqdm(rows, desc="   →", leave=False)):
            try:
                resp = requests.post(SERVER_URL, json=payload, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                writer.writerow({
                    "source_file": parquet_path.name,
                    "idx": idx,
                    "question": payload["question"],
                    "context":  payload["context"],
                    "answer":   payload["answer"],
                    "risk_score": data.get("risk_score"),
                    "label":      data.get("label")
                })
            except Exception as e:
                writer.writerow({
                    "source_file": parquet_path.name,
                    "idx": idx,
                    "question": payload["question"],
                    "context":  payload["context"],
                    "answer":   payload["answer"],
                    "risk_score": "ERROR",
                    "label": str(e)
                })

print(f"\nDone. Combined results saved to {OUTPUT_CSV.resolve()}")
