// scripts/batch_hf_evaluate.py
# ---------------------------------------------------------------
# Download a Hugging Face dataset, call Cliniguard /predict on each sample,
# and save a CSV of the results.
# ---------------------------------------------------------------
import json
import csv
import sys
from pathlib import Path

# Third‑party imports – install if missing:
#   pip install datasets requests tqdm
try:
    from datasets import load_dataset
except ImportError:
    sys.exit("Please install the 'datasets' library: pip install datasets")

import requests
from tqdm import tqdm

# ---------------------------------------------------------------
# Configuration – edit as needed
# ---------------------------------------------------------------
HF_DATASET = "pubmed_qa"          # Hugging Face dataset name (public)
HF_SUBSET  = "abstract"           # Sub‑set within the dataset (if any)
SERVER_URL = "http://localhost:5000/predict"  # Cliniguard API endpoint
OUTPUT_CSV = "cliniguard_hf_results.csv"

# ---------------------------------------------------------------
# Helper: build the payload expected by Cliniguard
# ---------------------------------------------------------------
def build_payload(sample):
    # The PubMed QA dataset fields: "question", "context", "answers" (list)
    # Use the first answer if multiple.
    question = sample.get("question", "")
    context = sample.get("context", "")
    answer = sample.get("answers", [""])[0] if isinstance(sample.get("answers"), list) else ""
    return {
        "question": question,
        "context": context,
        "answer": answer,
    }

# ---------------------------------------------------------------
def main():
    # Load the Hugging Face dataset (downloaded to ~/.cache by default)
    print(f"Loading dataset {HF_DATASET} …")
    ds = load_dataset(HF_DATASET, HF_SUBSET)
    # Most datasets have a "train" split; we will use the first 100 samples for demo.
    split = ds["train"]
    sample_count = min(100, len(split))
    print(f"Evaluating first {sample_count} samples…")

    # Open CSV for writing results
    out_path = Path(OUTPUT_CSV)
    with out_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["idx", "question", "context", "answer", "risk_score", "label"])
        writer.writeheader()

        for idx in tqdm(range(sample_count)):
            sample = split[idx]
            payload = build_payload(sample)
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
                    "label": data.get("label"),
                })
            except Exception as e:
                # On failure, write the error message so the CSV stays aligned
                writer.writerow({
                    "idx": idx,
                    "question": payload["question"],
                    "context": payload["context"],
                    "answer": payload["answer"],
                    "risk_score": "ERROR",
                    "label": str(e),
                })

    print(f"Done. Results saved to {out_path.resolve()}")

if __name__ == "__main__":
    main()
