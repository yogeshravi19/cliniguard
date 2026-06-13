# med_halt Dataset Extraction

**Source**: Hugging Face – `medhalt` repository.

**Download & Extraction**
```bash
pip install datasets  # if not installed
python - <<EOF
from datasets import load_dataset
load_dataset("medhalt", split="train").save_to_disk("data/med_halt")
EOF
```

**Pre‑processing**
- Convert each record to the unified schema `{question, context, answer, label}`.
- Tokenise bilingual text using the CLINIGUARD tokenizer (see `src/tokenizer.py`).
- Store the processed CSV as `data_extraction/med_halt/processed.csv`.

**Analysis & Reporting**
- Run `src/evaluate.py --dataset med_halt` to obtain AUROC, precision, recall.
- Generated plots are saved under `analysis/reports/med_halt/`.
- A short HTML report (`report.html`) summarises metrics and error cases.
