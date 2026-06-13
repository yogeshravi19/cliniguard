import json, csv, os, sys
from pathlib import Path
import requests
from tqdm import tqdm

# Configuration
DATA_DIR = Path(r"C:/Users/ACER/OneDrive/Desktop/cliniguard/data_extraction")
OUTPUT_CSV = Path(r"C:/Users/ACER/OneDrive/Desktop/cliniguard/cliniguard_local_results.csv")
SERVER_URL = "http://localhost:5000/predict"

if not DATA_DIR.exists():
    sys.exit(f"Dataset directory not found: {DATA_DIR}")

# Helper to build payload from a row (expects keys: question, context, answer)
def build_payload(row):
    return {
        "question": row.get("question", ""),
        "context": row.get("context", ""),
        "answer": row.get("answer", "")
    }

# Collect all rows from supported files (CSV or JSON lines)
rows = []
for file in DATA_DIR.iterdir():
    if file.suffix.lower() == ".csv":
        with open(file, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
    elif file.suffix.lower() in {".json", ".jsonl"}:
        with open(file, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # skip malformed lines
                    continue

if not rows:
    sys.exit(f"No QA rows found in {DATA_DIR}")

print(f"Found {len(rows)} samples – sending to Cliniguard API…")

with OUTPUT_CSV.open('w', newline='', encoding='utf-8') as out_f:
    writer = csv.DictWriter(out_f, fieldnames=["idx", "question", "context", "answer", "risk_score", "label"])
    writer.writeheader()
    for idx, row in enumerate(tqdm(rows)):
        payload = build_payload(row)
        try:
            resp = requests.post(SERVER_URL, json=payload, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            writer.writerow({
                "idx": idx,
                "question": payload["question"],
                "context": payload["context"],
                "answer": payload["answer"],
                "risk_score": data.get("risk_score"),
                "label": data.get("label")
            })
        except Exception as e:
            writer.writerow({
                "idx": idx,
                "question": payload["question"],
                "context": payload["context"],
                "answer": payload["answer"],
                "risk_score": "ERROR",
                "label": str(e)
            })

print(f"Done. Results saved to {OUTPUT_CSV.resolve()}")
