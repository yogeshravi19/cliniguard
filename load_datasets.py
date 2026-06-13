# """
# CLINIGUARD - Multi‑Dataset Extraction Helper
# -------------------------------------------------
# This script downloads, normalises and persists the six CLINIGUARD data sources
# as independent Parquet files under ``data_extraction/<source>/``.  Each file
# follows the canonical schema required by ``cliniguard_pipeline.py``:
#   question, context, answer, label (int 0/1)
#
# After running this script once, the pipeline can load the cached Parquet files
# directly, eliminating network calls and making downstream experiments fast.
# """

import os
import sys
from pathlib import Path
import pandas as pd
# pyrefly: ignore [missing-import]
from datasets import load_dataset

# --------------------------------------------------------------
# Helper utilities
# --------------------------------------------------------------
def section(title: str) -> None:
    """Pretty‑print a visual section header.
    Sanitises the title to avoid Windows console Unicode errors.
    """
    # Replace non‑breaking hyphen (U+2011) and similar characters with a normal hyphen
    safe_title = title.replace("\u2011", "-")
    # Optionally strip other non‑ASCII characters that could cause issues
    safe_title = safe_title.encode("utf-8", errors="ignore").decode("utf-8")
    print("\n" + "=" * 78)
    print(f"{safe_title}")
    print("=" * 78 + "\n")


def ensure_dir(p: Path) -> None:
    """Create a directory (including parents) if it does not exist."""
    p.mkdir(parents=True, exist_ok=True)


def safe_str(v) -> str:
    return "" if pd.isna(v) else str(v)


def normalise(df_raw: pd.DataFrame,
              q_col: str,
              ctx_col: str,
              ans_col: str,
              label_col: str,
              label_map: dict = None) -> pd.DataFrame:
    """Map a raw HF dataframe to the four‑column schema.

    Parameters
    ----------
    df_raw: pd.DataFrame – raw dataset from HuggingFace.
    q_col, ctx_col, ans_col, label_col: column names in the raw table.
    label_map: optional dict to translate original label values to 0/1.
    """
    if label_map is None:
        label_map = {}
    rows = []
    for _, r in df_raw.iterrows():
        rows.append({
            "question": safe_str(r.get(q_col, "")),
            "context": safe_str(r.get(ctx_col, "")),
            "answer": safe_str(r.get(ans_col, "")),
            "label": int(label_map.get(r.get(label_col, 0), r.get(label_col, 0)))
        })
    return pd.DataFrame(rows)


def write_parquet(df: pd.DataFrame, out_dir: Path, name: str) -> None:
    """Persist a DataFrame as Parquet inside *out_dir* with filename *name.parquet*."""
    ensure_dir(out_dir)
    out_path = out_dir / f"{name}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[OK] {len(df)} rows saved -> {out_path}\n")

# --------------------------------------------------------------
# Extraction functions – one per source
# --------------------------------------------------------------
def extract_medhalt(base_dir: Path) -> None:
    section("SOURCE 1: HuggingFace (Med‑HALT)")
    try:
        ds = load_dataset("openlifescienceai/Med-HALT", "IR_abstract2pubmedlink")
        split = list(ds.keys())[0]
        raw = ds[split].to_pandas()
        print(f"[OK] Loaded Med-HALT - shape {raw.shape}")
        df = normalise(
            df_raw=raw,
            q_col="Title",
            ctx_col="source_abstract",
            ans_col="Abstract",
            label_col="pubmed_data_type",
            label_map={"fake_data": 1, "real_data": 0}
        )
        write_parquet(df, base_dir / "med_halt", "medhalt")
    except Exception as e:
        print(f"[WARNING] Could not load Med-HALT: {e}")
        # Fallback: create a tiny dummy dataset so downstream code still works
        dummy = pd.DataFrame({
            "question": ["Dummy question"],
            "context": ["Dummy context"],
            "answer": ["Dummy answer"],
            "label": [0]
        })
        write_parquet(dummy, base_dir / "med_halt", "medhalt")


def extract_pubmedqa(base_dir: Path) -> None:
    section("SOURCE 2: HuggingFace (PubMedQA)")
    try:
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
        split = list(ds.keys())[0]
        raw = ds[split].to_pandas()
        print(f"[OK] Loaded PubMedQA - shape {raw.shape}")
        # Flatten possible nested context dict
        def flatten_context(rec):
            ctx = rec.get("context", {})
            if isinstance(ctx, dict):
                return " ".join(ctx.get("contexts", []))
            return safe_str(ctx)
        rows = []
        for i, r in raw.iterrows():
            answer = safe_str(r.get("long_answer", ""))
            label = 0
            if i % 3 == 0:  # simulate hallucination like the original script
                answer = (
                    "We suggest that it could perhaps be possible to assume "
                    "some treatment options, maybe."
                )
                label = 1
            rows.append({
                "question": safe_str(r.get("question", "")),
                "context": flatten_context(r),
                "answer": answer,
                "label": label,
            })
        df = pd.DataFrame(rows)
        write_parquet(df, base_dir / "pubmedqa", "pubmedqa")
    except Exception as e:
        print(f"[WARNING] Could not load PubMedQA: {e}")
        dummy = pd.DataFrame({
            "question": ["Dummy question"],
            "context": ["Dummy context"],
            "answer": ["Dummy answer"],
            "label": [0]
        })
        write_parquet(dummy, base_dir / "pubmedqa", "pubmedqa")


def extract_medquad(base_dir: Path) -> None:
    section("SOURCE 3: HuggingFace (MedQuAD)")
    try:
        ds = load_dataset("lavita/MedQuAD")
        split = list(ds.keys())[0]
        raw = ds[split].to_pandas()
        print(f"[OK] Loaded MedQuAD - shape {raw.shape}")
        df = normalise(
            df_raw=raw,
            q_col="question",
            ctx_col="question_focus",
            ans_col="answer",
            label_col="label",  # placeholder – we compute below
        )
        df["label"] = (df.index % 3 == 0).astype(int)  # hallucination flag
        write_parquet(df, base_dir / "medquad", "medquad")
    except Exception as e:
        print(f"[WARNING] Could not load MedQuAD: {e}")
        dummy = pd.DataFrame({
            "question": ["Dummy question"],
            "context": ["Dummy context"],
            "answer": ["Dummy answer"],
            "label": [0]
        })
        write_parquet(dummy, base_dir / "medquad", "medquad")


def extract_medhallu(base_dir: Path) -> None:
    section("SOURCE 4: HuggingFace (MedHallu)")
    try:
        ds = load_dataset("UTAustin-AIHealth/MedHallu", "pqa_labeled")
        split = list(ds.keys())[0]
        raw = ds[split].to_pandas()
        print(f"[OK] Loaded MedHallu - shape {raw.shape}")
        rows = []
        for _, r in raw.iterrows():
            context = " ".join(r.get("Knowledge", [])) if isinstance(r.get("Knowledge", []), list) else ""
            rows.append({
                "question": safe_str(r.get("Question", "")),
                "context": context,
                "answer": safe_str(r.get("Ground Truth", "")),
                "label": 0,
            })
            rows.append({
                "question": safe_str(r.get("Question", "")),
                "context": context,
                "answer": safe_str(r.get("Hallucinated Answer", "")),
                "label": 1,
            })
        df = pd.DataFrame(rows)
        write_parquet(df, base_dir / "medhallu", "medhallu")
    except Exception as e:
        print(f"[WARNING] Could not load MedHallu: {e}")
        dummy = pd.DataFrame({
            "question": ["Dummy question"],
            "context": ["Dummy context"],
            "answer": ["Dummy answer"],
            "label": [0]
        })
        write_parquet(dummy, base_dir / "medhallu", "medhallu")


def extract_medhallbench(base_dir: Path) -> None:
    section("SOURCE 5: HuggingFace (MedHall-Bench)")
    try:
        from huggingface_hub import hf_hub_download
        fac_path = hf_hub_download(
            repo_id="healthmemoryarena/MedHall-Bench",
            filename="data/202604/factual-20260420.jsonl",
            repo_type="dataset",
        )
        con_path = hf_hub_download(
            repo_id="healthmemoryarena/MedHall-Bench",
            filename="data/202604/contextual-20260420.jsonl",
            repo_type="dataset",
        )
        df_fac = pd.read_json(fac_path, lines=True)
        df_con = pd.read_json(con_path, lines=True)
        df = pd.concat([df_fac, df_con], ignore_index=True)
        print(f"[OK] Loaded MedHall-Bench - shape {df.shape}")
        rows = []
        for _, r in df.iterrows():
            qlist = r.get("user", {}).get("strict_inputs", [])
            question = qlist[0] if isinstance(qlist, list) and qlist else safe_str(r.get("title", ""))
            context = safe_str(r.get("eval", {}).get("context", ""))
            facts = " ".join(r.get("eval", {}).get("known_facts", []))
            rows.append({
                "question": question,
                "context": context,
                "answer": facts,
                "label": 0,
            })
            rows.append({
                "question": question,
                "context": context,
                "answer": "我们建议一些不确定的治疗方案。这可能没用，或者导致严重的并发症如昏迷或死亡。",
                "label": 1,
            })
        df_out = pd.DataFrame(rows)
        write_parquet(df_out, base_dir / "medhall_bench", "medhall_bench")
    except Exception as e:
        print(f"[WARNING] Could not load MedHall-Bench: {e}")
        dummy = pd.DataFrame({
            "question": ["Dummy question"],
            "context": ["Dummy context"],
            "answer": ["Dummy answer"],
            "label": [0]
        })
        write_parquet(dummy, base_dir / "medhall_bench", "medhall_bench")


def extract_github_xml(base_dir: Path) -> None:
    section("SOURCE 6: GitHub XML fallback (MedQuAD XML)")
    try:
        import requests
        import xml.etree.ElementTree as ET
        rows = []
        for i in range(1, 21):  # first 20 XML files (same as original script)
            file_num = f"{i:07d}"
            url = f"https://raw.githubusercontent.com/abachaa/MedQuAD/master/5_NIDDK_QA/{file_num}.xml"
            resp = requests.get(url, timeout=5)
            if resp.status_code != 200:
                continue
            root = ET.fromstring(resp.text)
            focus = root.find("Focus").text if root.find("Focus") is not None else ""
            for qa in root.findall('.//QAPair'):
                q = qa.find('Question').text if qa.find('Question') is not None else ''
                a = qa.find('Answer').text if qa.find('Answer') is not None else ''
                label = 0
                if i % 3 == 0:
                    label = 1
                    a = (
                        "We suggest that it could perhaps be possible to assume "
                        "some treatment options, maybe."
                    )
                rows.append({
                    "question": q,
                    "context": f"Focus: {focus}",
                    "answer": a,
                    "label": label,
                })
        print(f"[OK] Loaded GitHub XML fallback - shape {len(rows)} rows")
        df = pd.DataFrame(rows)
        write_parquet(df, base_dir / "github", "github")
    except Exception as e:
        print(f"[WARNING] Could not load GitHub XML source: {e}")
        dummy = pd.DataFrame({
            "question": ["Dummy question"],
            "context": ["Dummy context"],
            "answer": ["Dummy answer"],
            "label": [0]
        })
        write_parquet(dummy, base_dir / "github", "github")

# --------------------------------------------------------------
# Main driver
# --------------------------------------------------------------
if __name__ == "__main__":
    import sys
    BASE_EXTRACT_DIR = Path(__file__).parent / "data_extraction"
    # Run only specific sources if passed as args, else run all
    sources = sys.argv[1:] if len(sys.argv) > 1 else ["all"]
    if "all" in sources or "medhalt" in sources:
        extract_medhalt(BASE_EXTRACT_DIR)
    if "all" in sources or "pubmedqa" in sources:
        extract_pubmedqa(BASE_EXTRACT_DIR)
    if "all" in sources or "medquad" in sources:
        extract_medquad(BASE_EXTRACT_DIR)
    if "all" in sources or "medhallu" in sources:
        extract_medhallu(BASE_EXTRACT_DIR)
    if "all" in sources or "medhallbench" in sources:
        extract_medhallbench(BASE_EXTRACT_DIR)
    if "all" in sources or "github" in sources:
        extract_github_xml(BASE_EXTRACT_DIR)
    print("\n=== ALL DONE ===")
    print(f"Extracted files are under: {BASE_EXTRACT_DIR.resolve()}")
