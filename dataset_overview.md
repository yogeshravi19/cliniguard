# Dataset Overview for CLINIGUARD

The CLINIGUARD project evaluates six curated clinical QA datasets.  Below is a concise reference sheet that lists each dataset, its source URL, download method, size (sampled rows used in the benchmark), file format, and the canonical column names that the pipeline expects.

---

## 1️⃣ Med‑HALT
* **Source**: Hugging Face – `openlifescienceai/Med-HALT` (`IR_abstract2pubmedlink` config)
* **URL**: https://huggingface.co/openlifescienceai/Med-HALT
* **Typical size**: ~4 916 rows (full). In the benchmark we sample the first **500** rows.
* **Format**: HF `Dataset` → pandas DataFrame.
* **Important columns**:
  * `Title` → **question**
  * `source_abstract` → **context**
  * `Abstract` → **answer**
  * `pubmed_data_type` (`fake_data` → label = 1, else 0)

---

## 2️⃣ PubMedQA
* **Source**: Hugging Face – `qiaojin/PubMedQA` (`pqa_labeled` config)
* **URL**: https://huggingface.co/qiaojin/PubMedQA
* **Typical size**: 1 000 rows (full). Benchmark samples the first **500** rows.
* **Format**: HF `Dataset` → pandas DataFrame.
* **Important columns**:
  * `question` → **question**
  * `context` (nested dict) → **context** (concatenated string of all `contexts` entries)
  * `long_answer` → **answer** (occasionally perturbed to create hallucinations)
  * `final_decision` (`maybe` → label = 1) – used to flag uncertainty.

---

## 3️⃣ MedQuAD (HF version)
* **Source**: Hugging  Face – `lavita/MedQuAD`
* **URL**: https://huggingface.co/lavita/MedQuAD
* **Typical size**: 47 441 rows (full). Benchmark samples the first **500** rows.
* **Format**: HF `Dataset` → pandas DataFrame.
* **Important columns**:
  * `question` → **question**
  * `question_focus` → **context**
  * `answer` → **answer** (perturbed every 3rd row to emulate hallucination)
  * No explicit label column – we generate `label = 1` for the perturbed rows.

---

## 4️⃣ MedHallu
* **Source**: Hugging  Face – `UTAustin-AIHealth/MedHallu` (`pqa_labeled` config)
* **URL**: https://huggingface.co/UTAustin-AIHealth/MedHallu
* **Typical size**: 1 000 rows (full). Benchmark samples the first **500** rows.
* **Format**: HF `Dataset` → pandas DataFrame.
* **Important fields**:
  * `Question` → **question**
  * `Knowledge` (list) → **context** (joined into a single string)
  * `Ground Truth` → **answer** (label = 0)
  * `Hallucinated Answer` → **answer** (label = 1)

---

## 5️⃣ MedHall‑Bench (Bilingual factual + contextual)
* **Source**: Hugging  Face – `healthmemoryarena/MedHall-Bench`
* **URL**: https://huggingface.co/healthmemoryarena/MedHall-Bench
* **Files**:
  * `data/202604/factual-20260420.jsonl` – factual QA pairs.
  * `data/202604/contextual-20260420.jsonl` – contextual (hallucinated) QA pairs.
* **Size**: 54 rows total (27 factual + 27 contextual). Used entirely (no sampling).
* **Format**: JSON‑Lines → pandas `read_json(..., lines=True)`.
* **Important fields**:
  * `user.strict_inputs[0]` or `title` → **question**
  * `eval.context` → **context**
  * `eval.known_facts` (list) → **answer** (factual case)
  * Hallucinated case uses a bilingual Chinese sentence as the answer.

---

## 6️⃣ MedQuAD XML fallback (GitHub NIH XML files)
* **Source**: GitHub repository – `abachaa/MedQuAD` (NIH XML files)
* **URL**: https://github.com/abachaa/MedQuAD/tree/master/5_NIDDK_QA
* **Files**: 20 XML files named `0000001.xml` … `0000020.xml` (each contains up to ~30 Q‑A pairs).
* **Size**: ~600 rows after parsing (the script processes the first 20 files).
* **Format**: XML → parsed with `xml.etree.ElementTree` → pandas DataFrame.
* **Important fields**:
  * `Focus` element → **context** (`Focus: <text>`)
  * `Question` element → **question**
  * `Answer` element → **answer**
  * Every 3rd record is artificially labelled as hallucinated (`label = 1`).

---

## Common Schema Used by the Pipeline
| Column | Description |
|---|---|
| `_question` | Raw question string (English or bilingual). |
| `_context`  | Supporting context (abstract, knowledge, focus, etc.). |
| `_answer`   | Model‑generated (or ground‑truth) answer string. |
| `_label`    | Binary ground truth: **0** = factual / safe, **1** = hallucinated / unsafe. |

All extractors normalise their raw fields to this schema before the signal‑extraction stage.

---

**How to use**
```bash
# Example – load a single source
python - <<'PY'
from cliniguard_pipeline import load_source_dataset
df = load_source_dataset('medhalt')
print(df.head())
PY
```
The resulting DataFrame will contain the four canonical columns (`_question`, `_context`, `_answer`, `_label`) ready for the downstream feature‑extraction and fusion steps.

---

*Last updated:* 2026‑06‑13 | *Project root*: `c:/Users/ACER/OneDrive/Desktop/cliniguard`
