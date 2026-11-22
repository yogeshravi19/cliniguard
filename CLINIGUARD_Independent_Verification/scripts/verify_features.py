"""
Signal and Feature Verification Script for CLINIGUARD
"""
import sys, math, re
from pathlib import Path
import pandas as pd
import numpy as np

root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

from cliniguard_pipeline import (
    med_isp, c_aas, med_eem, cdt, tokenize,
    DRUG_TERMS_EN, CONTEXT_TERMS_EN, UNCERTAIN_WORDS_EN,
    RE_UNITS_EN, RE_DRUG_EN, RE_CONTEXT_EN, RE_UNCERTAIN_EN
)

def run_feature_verification():
    print("=" * 80)
    print("CLINIGUARD FOUR SIGNAL MATHEMATICAL & BEHAVIORAL AUDIT")
    print("=" * 80)

    # 1. Med-ISP
    print("\n[1] Med-ISP (Medical Information Safety Proxy / Pharmacology Density):")
    rich_text = "The doctor prescribed 500mg metformin orally daily."
    sparse_text = "The quick brown fox jumps over the lazy dog."
    print(f"  Formula: 1.0 - min(hits / max(len(words)*0.05, 1), 1.0)")
    print(f"  Pharma Rich Text:   '{rich_text}' -> Med-ISP: {med_isp(rich_text)} (Low risk)")
    print(f"  Pharma Sparse Text: '{sparse_text}' -> Med-ISP: {med_isp(sparse_text)} (High risk)")
    print(f"  Empty String:       '' -> Med-ISP: {med_isp('')} (Safe fallback to 1.0)")

    # 2. C-AAS
    print("\n[2] C-AAS (Clinical Context Term Alignment):")
    ctx_text = "The adult female patient reported severe acute symptoms and diagnosis of allergy."
    print(f"  Formula: 1.0 - min(hits / max(len(words)*0.04, 1), 1.0)")
    print(f"  Context Rich Text:   '{ctx_text}' -> C-AAS: {c_aas(ctx_text)} (Low risk)")
    print(f"  Context Sparse Text: '{sparse_text}' -> C-AAS: {c_aas(sparse_text)} (High risk)")

    # 3. Med-EEM
    print("\n[3] Med-EEM (Epistemic Entropy Measure / Hedging Frequency):")
    hedge_low = "The patient has hypertension."
    hedge_high = "The patient might possibly perhaps have uncertain symptoms."
    print(f"  Formula: -p*log2(p) - (1-p)*log2(1-p) where p = hedging_hits / n")
    print(f"  Low Hedging Text:  '{hedge_low}' -> Med-EEM: {med_eem(hedge_low)}")
    print(f"  High Hedging Text: '{hedge_high}' -> Med-EEM: {med_eem(hedge_high)}")
    print(f"  Scientific Note: Since hedging frequency p < 0.05 in clinical texts, binary entropy H(p) is strictly monotonic with respect to p. Tree models like LightGBM treat Med-EEM as a linear hedging count.")

    # 4. CDT
    print("\n[4] CDT (Contextual Drift Threshold / Unigram Cosine Distance):")
    q = "What is the recommended dose of metformin for diabetes?"
    a_aligned = "The recommended dose of metformin for diabetes is 500mg daily."
    a_drifted = "Quantum teleportation across interstellar wormholes was discovered."
    print(f"  Formula: 1.0 - cos_sim(prompt_unigrams, answer_unigrams)")
    print(f"  Aligned Answer: '{a_aligned}' -> CDT: {cdt(a_aligned, q)}")
    print(f"  Drifted Answer: '{a_drifted}' -> CDT: {cdt(a_drifted, q)}")

if __name__ == "__main__":
    run_feature_verification()
