import math
import numpy as np
from typing import Set

# Define term sets (same as earlier implementations)
DRUG_TERMS: Set[str] = {
    "mg", "dose", "dosage", "tablet", "capsule", "injection", "oral", "iv", "intravenous",
    "amoxicillin", "ibuprofen", "metformin", "insulin", "aspirin", "atorvastatin", "omeprazole",
    "paracetamol", "acetaminophen", "warfarin", "morphine", "prednisone", "antibiotic", "medication",
    "drug", "prescribe", "contraindication", "side", "effect", "adverse",
}

CONTEXT_TERMS: Set[str] = {
    "patient", "allergy", "allergic", "age", "weight", "pediatric", "adult", "vital",
    "history", "medication", "diagnosis", "symptom", "report", "female", "male",
    "blood pressure", "heart rate", "temperature", "chronic", "acute", "clinical", "contraindication",
}

UNCERTAIN_WORDS: Set[str] = {
    "maybe", "possibly", "might", "could", "uncertain", "unclear", "unknown", "approximately", "seems", "appears", "suggest",
    "perhaps", "likely", "probably", "assume", "think", "believe", "estimate", "roughly", "sometimes", "often",
}

def tokenize(text: str) -> list[str]:
    return text.lower().split() if isinstance(text, str) else []

def med_isp(text: str) -> float:
    words = tokenize(text)
    if not words:
        return 1.0
    hits = sum(1 for w in words if any(term in w for term in DRUG_TERMS))
    density = hits / max(len(words) * 0.05, 1)
    return round(1.0 - min(density, 1.0), 4)

def c_aas(text: str) -> float:
    words = tokenize(text)
    if not words:
        return 1.0
    hits = sum(1 for w in words if any(term in w for term in CONTEXT_TERMS))
    density = hits / max(len(words) * 0.04, 1)
    return round(1.0 - min(density, 1.0), 4)

def med_eem(text: str) -> float:
    words = tokenize(text)
    n = len(words)
    if n == 0:
        return 0.0
    uncertain_hits = sum(1 for w in words if any(term in w for term in UNCERTAIN_WORDS))
    p = uncertain_hits / n
    eps = 1e-9
    H = -(p * math.log2(p + eps) + (1 - p) * math.log2(1 - p + eps))
    return round(min(H * (1 + p), 1.0), 4)

def _wvec(text: str) -> dict[str, int]:
    freq: dict[str, int] = {}
    for w in tokenize(text):
        freq[w] = freq.get(w, 0) + 1
    return freq

def cdt(answer: str, question: str) -> float:
    v1, v2 = _wvec(question), _wvec(answer)
    vocab = set(v1) | set(v2)
    if not vocab:
        return 0.5
    dot = sum(v1.get(w, 0) * v2.get(w, 0) for w in vocab)
    m1 = math.sqrt(sum(x ** 2 for x in v1.values()))
    m2 = math.sqrt(sum(x ** 2 for x in v2.values()))
    if m1 == 0 or m2 == 0:
        return 0.5
    return round(1.0 - dot / (m1 * m2), 4)

def risk_label(score: float) -> str:
    if score < 0.35:
        return "🟢 GREEN (Safe)"
    elif score < 0.65:
        return "🟡 AMBER (Review needed)"
    else:
        return "🔴 RED (High‑risk hallucination)"
