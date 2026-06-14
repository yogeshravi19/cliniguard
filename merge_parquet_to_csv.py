# Merge six CLINIGUARD Parquet files into a single CSV
# ------------------------------------------------------------
# This script loads the parquet files created by `load_datasets.py`,
# concatenates them, adds a `source_dataset` column, and writes the
# combined result to `cliniguard_all_datasets.csv` at the repository root.
# ------------------------------------------------------------
import pathlib
import pandas as pd

BASE = pathlib.Path(r"F:/cliniguard")
DATA_DIR = BASE / "data_extraction"

# Mapping of folder name → parquet filename (no extension).
DATASETS = {
    "Med-HALT": "medhalt",
    "pubmedqa": "pubmedqa",
    "medquad": "medquad",
    "MedHallu": "medhallu",
    "medhall_bench": "medhall_bench",
    "github": "github",
}

frames = []
for folder, fname in DATASETS.items():
    p = DATA_DIR / folder / f"{fname}.parquet"
    print(f"Loading {p} …")
    df = pd.read_parquet(p)
    df["source_dataset"] = folder
    frames.append(df)

merged = pd.concat(frames, ignore_index=True)
out_path = BASE / "cliniguard_all_datasets.csv"
merged.to_csv(out_path, index=False)
print(f"[OK] Merged CSV written to {out_path}")
