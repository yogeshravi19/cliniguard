"""
Label Mapping and Medical Safety Category Verification
"""
from pathlib import Path
import pandas as pd

def run_label_verification():
    print("=" * 80)
    print("CLINIGUARD LABEL MAPPING & MEDICAL SAFETY AUDIT")
    print("=" * 80)

    print("\n[1] Documented Output Categories:")
    print("  GREEN / SAFE:      P(hallucination) < 0.20 (or score < 0.35 in legacy pipeline)")
    print("  AMBER / AMBIGUOUS: 0.20 <= P(hallucination) <= 0.70")
    print("  RED / UNSAFE:      P(hallucination) > 0.70")

    print("\n[2] Source Dataset Original Label Semantics:")
    print("  Med-HALT: Binary (real_data = 0 Factual, fake_data = 1 Hallucinated). Real PubMed vs LLM completion.")
    print("  MedHallu: Binary (Ground Truth = 0 Factual, Hallucinated Answer = 1). Expert doctor vs LLM error.")
    print("  Legacy PubMedQA: Original has NO hallucination labels. Legacy script injected synthetic sentence every 3rd row.")
    print("  Legacy MedQuAD: Original has NO hallucination labels. Legacy script assigned index % 3 == 0.")

    print("\n[3] Clinical Validity Assessment:")
    print("  - The labels distinguish lexical/factual source pairs, NOT independent clinician-verified safety.")
    print("  - The AMBER tier is a post-hoc probability threshold band, not an intrinsic annotation in source benchmarks.")

if __name__ == "__main__":
    run_label_verification()
