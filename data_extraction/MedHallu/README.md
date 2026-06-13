# med_hallu Dataset Extraction

**Source**: Hugging Face – `medhallu` repository.

**Download & Extraction**
```bash
pip install datasets
python - <<EOF
from datasets import load_dataset
load_dataset("medhallu", "pqa_labeled", split="train").save_to_disk("data/med_hallu")
EOF
```

**Pre‑processing**
- Convert each entry to the unified schema `{question, context, answer, label}`.
- Run the CLINIGUARD bilingual tokenizer.
- Save as `data_extraction/med_hallu/processed.csv`.

**Analysis & Reporting**
- Execute `src/evaluate.py --dataset med_hallu`.
- Metrics/plots are stored in `analysis/reports/med_hallu/`.
- An HTML `report.html` summarises AUROC, precision, recall, and example error cases.
