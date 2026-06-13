# pubmedqa Dataset Extraction

**Source**: Hugging Face – `pubmedqa` repository.

**Download & Extraction**
```bash
pip install datasets
python - <<EOF
from datasets import load_dataset
load_dataset("pubmedqa", "pqa_labeled", split="train").save_to_disk("data/pubmedqa")
EOF
```

**Pre‑processing**
- Convert to unified schema `{question, context, answer, label}`.
- Apply CLINIGUARD bilingual tokenizer.
- Save as `data_extraction/pubmedqa/processed.csv`.

**Analysis & Reporting**
- Run `src/evaluate.py --dataset pubmedqa`.
- Figures saved to `analysis/reports/pubmedqa/`.
- HTML report `report.html` summarizes AUROC, precision, recall, error cases.
