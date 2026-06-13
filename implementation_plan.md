# Implementation Plan – Modular Dataset Extraction for CLINIGUARD

## Goal
Refactor the monolithic `load_source_dataset` function in `cliniguard_pipeline.py` so that each dataset has its own dedicated extraction script/module. This improves readability, makes it straightforward to add new datasets, and isolates dataset‑specific quirks (different field names, formats, fallback logic).

## User Review Required
- **Do you want to keep the existing `--source auto` CLI behaviour?** (i.e., a single command that still loads all datasets in order.)
- **Should the new extractor modules be placed under `data_extraction/`** (one folder per dataset) **or a new top‑level `extractors/` folder?**
- **Do you need unit‑test scaffolding now or later?** (We can add a `tests/` folder with pytest fixtures.)

## Open Questions
> [!IMPORTANT] Are there any additional datasets you foresee adding in the near future (e.g., a new bilingual QA set) that might require special handling beyond what the current loaders support?
> 
> > *Answer:* Provide any upcoming sources so we can design the registry for extensibility.

## Proposed Changes
---
### 1. Directory layout
```
cliniguard/
├─ data_extraction/               # existing raw data folder (unchanged)
├─ extractors/                    # NEW – one sub‑folder per dataset
│   ├─ medhalt/
│   │   └─ extractor.py
│   ├─ pubmedqa/
│   │   └─ extractor.py
│   ├─ medquad_hf/
│   │   └─ extractor.py
│   ├─ medhallu/
│   │   └─ extractor.py
│   ├─ medhallbench/
│   │   └─ extractor.py
│   └─ github_xml/
│       └─ extractor.py
├─ scripts/                      # unchanged – pipeline orchestration lives here
│   └─ cliniguard_pipeline.py
├─ tests/                         # NEW – pytest unit tests for each extractor
│   └─ test_extractors.py
```

### 2. Extractor interface
Each `extractor.py` will expose a single function:
```python
def extract(limit: int | None = 500) -> pd.DataFrame:
    """Return a DataFrame with columns [_question, _context, _answer, _label].
    `limit` truncates the dataset for quick dev runs; `None` returns the full set.
    """
```
*The function hides all HF/HTTP calls, environment‑variable tweaks, and format conversion.*

### 3. Registry
Create a new module `extractors/registry.py`:
```python
from importlib import import_module

EXTRACTOR_MAP = {
    "medhalt": "extractors.medhalt.extractor",
    "pubmedqa": "extractors.pubmedqa.extractor",
    "medquad": "extractors.medquad_hf.extractor",
    "medhallu": "extractors.medhallu.extractor",
    "medhallbench": "extractors.medhallbench.extractor",
    "github": "extractors.github_xml.extractor",
}

def get_extractor(name: str):
    module_path = EXTRACTOR_MAP.get(name)
    if not module_path:
        raise ValueError(f"Unknown dataset source: {name}")
    module = import_module(module_path)
    return module.extract
```
### 4. Update `load_source_dataset`
Replace the big `if/elif` block with:
```python
from extractors.registry import get_extractor

def load_source_dataset(source: str, limit: int | None = 500) -> pd.DataFrame:
    extractor = get_extractor(source)
    df = extractor(limit)
    return df
```
All environment variable handling (`HF_HUB_DISABLE_SYMLINKS`) can be moved into a tiny helper `utils.py` that `load_source_dataset` imports before calling the extractor.

### 5. Implement each extractor (copy‑paste from existing code)
- **medhalt/extractor.py** – copy lines 162‑173 from the current pipeline (adjust imports). 
- **pubmedqa/extractor.py** – copy lines 175‑200 (adjust for the new limit argument). 
- **medquad_hf/extractor.py** – copy lines 202‑222 (including the fallback to GitHub). 
- **medhallu/extractor.py** – copy lines 224‑238. 
- **medhallbench/extractor.py** – copy lines 247‑277. 
- **github_xml/extractor.py** – copy lines 279‑302.
Each file will import `pandas as pd`, `os`, `requests`, `xml.etree.ElementTree as ET`, and `datasets.load_dataset` where needed.

### 6. Add unit tests
`tests/test_extractors.py` will iterate over `EXTRACTOR_MAP` and assert that the returned DataFrame:
- Has the four required columns (`_question`, `_context`, `_answer`, `_label`).
- Contains at most `limit` rows (default 10 for fast CI). 
- No NaN values in those columns.
We can use the `limit=10` argument in tests to keep network traffic low.

### 7. Documentation
Add a `README.md` inside `extractors/` explaining:
- How to add a new dataset (create folder, implement `extract`).
- How the registry works.
- How to run the pipeline (`python cliniguard_pipeline.py --source auto`).

### 8. Backwards compatibility
- The CLI flag `--source auto` will still iterate over the same list of source strings defined in the pipeline (`sources = ["medhalt", "pubmedqa", "medquad", "medhallu", "medhallbench"]`).
- Existing notebooks that import `load_source_dataset` keep working because the function signature is unchanged (the `limit` parameter defaults to 500).

## Verification Plan
1. **Run the full benchmark** after refactor:
   ```powershell
   python cliniguard_pipeline.py --source auto
   ```
   Expect identical AUROC/AP numbers as before (within rounding error). 
2. **Run unit tests**:
   ```powershell
   pip install pytest
   pytest tests/
   ```
   All tests should pass. 
3. **Manual spot‑check** a few rows from each extractor (print `df.head()`), ensure fields map correctly.

---
**Next steps after your approval**
- Break out the new extractor files into the repository. 
- Update `cliniguard_pipeline.py` to use the registry. 
- Add the test suite and documentation. 
- Run the verification steps to confirm parity with the original implementation.

*Please review the open questions and confirm the preferred folder (`extractors/` vs `data_extraction/`). Once approved I will start applying the changes.*
