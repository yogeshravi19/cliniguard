# MedHall-Bench Dataset Extraction

This folder contains the scripts and instructions to download and preprocess the **MedHall‑Bench** dataset.

## Overview
- **Source**: HuggingFace `medhall-bench` (raw JSONL).
- **Size**: 54 entries (27 factual, 27 contextual hallucinations).
- **Languages**: Bilingual English‑Chinese.

## Steps
1. **Download**
   ```bash
   python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='medhall-bench', filename='medhall_bench.jsonl', cache_dir='data/raw')"
   ```
2. **Convert to unified schema** (see `process_medhall_bench.py`).
3. **Save** processed CSV to `data/processed/medhall_bench.csv`.

## Files
- `process_medhall_bench.py` – conversion script.
- `README.md` – this file.
