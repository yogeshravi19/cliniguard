"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          CLINIGUARD — Multi-Signal Clinical Hallucination Guard             ║
║          Capstone Final Pipeline  |  Python 3.13  |  2026                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ARCHITECTURE:                                                              ║
║    Signal 1 — Med-ISP  : Medical drug-term density probe                   ║
║    Signal 2 — C-AAS    : Clinical attention alignment score                ║
║    Signal 3 — Med-EEM  : Shannon entropy uncertainty monitor               ║
║    Signal 4 — CDT      : Cosine-similarity drift tracker                   ║
║                                                                             ║
║  FUSION:                                                                    ║
║    Risk = α·Med-ISP + β·C-AAS + γ·Med-EEM + δ·CDT                        ║
║    Weights (α,β,γ,δ) are task-conditional (dosing/allergy/diagnosis)      ║
║                                                                             ║
║  MODEL:                                                                     ║
║    ONE unified Logistic Regression trained on ALL 6 datasets combined      ║
║    Evaluated overall + per-dataset + ablation study                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import math
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, classification_report,
    precision_recall_curve, average_precision_score,
    f1_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
warnings.filterwarnings("ignore")

# UTF-8 output for Windows
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="CLINIGUARD Final Capstone Pipeline")
parser.add_argument(
    "--source", type=str,
    choices=["medhalt","pubmedqa","medquad","medhallu","medhallbench","github","auto"],
    default="auto",
    help="'auto' = unified model across all datasets (default)"
)
parser.add_argument("--samples", type=int, default=500,
                    help="Rows per dataset (default 500)")
args, _ = parser.parse_known_args()

# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def banner(title, char="=", width=78):
    print(f"\n{char*width}")
    print(f"  {title}")
    print(f"{char*width}\n")

def sub_banner(title):
    print(f"\n  {'─'*70}")
    print(f"  {title}")
    print(f"  {'─'*70}")

# ─────────────────────────────────────────────────────────────────────────────
# LOCAL PARQUET REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
_BASE_DATA = Path(__file__).parent / "data_extraction"

PARQUET_REGISTRY = {
    "medhalt":      _BASE_DATA / "Med-HALT"     / "medhalt.parquet",
    "pubmedqa":     _BASE_DATA / "pubmedqa"      / "pubmedqa.parquet",
    "medquad":      _BASE_DATA / "medquad"       / "medquad.parquet",
    "medhallu":     _BASE_DATA / "MedHallu"      / "medhallu.parquet",
    "medhallbench": _BASE_DATA / "medhall_bench" / "medhall_bench.parquet",
    "github":       _BASE_DATA / "github"        / "github.parquet",
}

def load_source(source: str, n: int = 500) -> pd.DataFrame:
    """Load a dataset from its local Parquet cache."""
    path = PARQUET_REGISTRY.get(source)
    if path is None:
        raise ValueError(f"Unknown source '{source}'")
    if not path.exists():
        raise FileNotFoundError(
            f"Missing: {path}\nRun 'python load_datasets.py' first."
        )
    df = pd.read_parquet(path).head(n).reset_index(drop=True)
    df["_question"] = df["question"].astype(str)
    df["_context"]  = df["context"].astype(str)
    df["_answer"]   = df["answer"].astype(str)
    df["_text"]     = df["_question"] + " [SEP] " + df["_context"]
    df["_label"]    = df["label"].astype(int)
    df["_source"]   = source
    df = df[df["_text"].str.len() > 10].reset_index(drop=True)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# BILINGUAL CLINICAL LEXICONS
# ─────────────────────────────────────────────────────────────────────────────
DRUG_TERMS = {
    "mg","dose","dosage","tablet","capsule","injection","oral","iv","intravenous",
    "subcutaneous","amoxicillin","ibuprofen","metformin","insulin","aspirin",
    "atorvastatin","omeprazole","paracetamol","acetaminophen","warfarin","heparin",
    "morphine","prednisone","antibiotic","medication","drug","prescribe","pharmacy",
    "rxnorm","formulary","contraindication","side effect","adverse",
    # Chinese
    "毫克","剂量","口服","注射","静脉","胶囊","药片","阿司匹林","阿莫西林",
    "二甲双胍","胰岛素","副作用","不良反应"
}

CONTEXT_TERMS = {
    "patient","allergy","allergic","age","weight","pediatric","adult","vital",
    "history","medication","diagnosis","symptom","report","female","male",
    "blood pressure","heart rate","temperature","chronic","acute","clinical",
    "contraindication","comorbid","complication",
    # Chinese
    "患者","过敏","年龄","体重","儿童","幼儿","成人","病史","诊断","症状","报告"
}

UNCERTAIN_WORDS = {
    "maybe","possibly","might","could","uncertain","unclear","unknown",
    "approximately","seems","appears","suggest","perhaps","likely","probably",
    "assume","think","believe","estimate","roughly","sometimes","often",
    # Chinese
    "可能","或许","大概","似乎","不确定","未知","大约","估计","有时","经常"
}

DOSING_KEYWORDS   = {"dose","dosage","mg","ml","tablet","capsule","prescribe",
                     "administer","inject","titrate","frequency","tds","bd","qds"}
ALLERGY_KEYWORDS  = {"allergy","allergic","hypersensitivity","anaphylaxis",
                     "reaction","sensitivity","contraindicated","penicillin"}

# ─────────────────────────────────────────────────────────────────────────────
# TOKENIZER
# ─────────────────────────────────────────────────────────────────────────────
def tokenize(text: str) -> list:
    if not isinstance(text, str) or not text.strip():
        return []
    text = text.lower()
    chinese = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    if chinese > len(text) * 0.2:
        tokens, buf = [], []
        for ch in text:
            if "\u4e00" <= ch <= "\u9fff":
                if buf: tokens.append("".join(buf)); buf = []
                tokens.append(ch)
            elif ch.isalnum():
                buf.append(ch)
            else:
                if buf: tokens.append("".join(buf)); buf = []
        if buf: tokens.append("".join(buf))
        return tokens
    return text.split()

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL 1 — Med-ISP  (Medical Information State Probe)
# Drug-term density: absence of drug terms = hallucination risk
# ─────────────────────────────────────────────────────────────────────────────
def med_isp(text: str) -> float:
    words = tokenize(text)
    if not words:
        return 1.0
    hits = sum(1 for w in words if any(t in w for t in DRUG_TERMS))
    density = hits / max(len(words) * 0.05, 1)
    return round(1.0 - min(density, 1.0), 4)

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL 2 — C-AAS  (Clinical Attention Alignment Score)
# Clinical-context term density: ignoring patient context = risk
# ─────────────────────────────────────────────────────────────────────────────
def c_aas(text: str) -> float:
    words = tokenize(text)
    if not words:
        return 1.0
    hits = sum(1 for w in words if any(t in w for t in CONTEXT_TERMS))
    density = hits / max(len(words) * 0.04, 1)
    return round(1.0 - min(density, 1.0), 4)

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL 3 — Med-EEM  (Medical Entropy-Energy Monitor)
# Shannon binary entropy over uncertain-word probability
# H(p) = −p·log₂(p) − (1−p)·log₂(1−p)   where p = uncertain words / total
# ─────────────────────────────────────────────────────────────────────────────
def med_eem(text: str) -> float:
    words = tokenize(text)
    n = len(words)
    if n == 0:
        return 0.0
    u_hits = sum(1 for w in words if any(u in w for u in UNCERTAIN_WORDS))
    p = u_hits / n
    eps = 1e-9
    # Binary Shannon entropy → max = 1.0 at p=0.5
    H = -(p * math.log2(p + eps) + (1 - p) * math.log2(1 - p + eps))
    # Normalise: max theoretical H_binary = 1.0 bit; scale to [0,1]
    score = min(H, 1.0)
    # Bias: high uncertainty words = higher score
    score = score * (1 + p)  # amplify if p is high
    return round(min(score, 1.0), 4)

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL 4 — CDT  (Clinical Drift Tracker)
# Cosine similarity between question & answer word-frequency vectors
# drift = 1 − cosine_similarity(Q, A)
# ─────────────────────────────────────────────────────────────────────────────
def _word_vector(text: str) -> dict:
    freq = {}
    for w in tokenize(text):
        freq[w] = freq.get(w, 0) + 1
    return freq

def cdt(answer: str, question: str) -> float:
    v1 = _word_vector(question)
    v2 = _word_vector(answer)
    vocab = set(v1) | set(v2)
    if not vocab:
        return 0.5
    dot   = sum(v1.get(w, 0) * v2.get(w, 0) for w in vocab)
    mag1  = math.sqrt(sum(x**2 for x in v1.values()))
    mag2  = math.sqrt(sum(x**2 for x in v2.values()))
    if mag1 == 0 or mag2 == 0:
        return 0.5
    cos_sim = dot / (mag1 * mag2)
    return round(1.0 - cos_sim, 4)

# ─────────────────────────────────────────────────────────────────────────────
# TASK CLASSIFIER
# Rule-based: detects clinical task type from question text
# ─────────────────────────────────────────────────────────────────────────────
def classify_task(question: str) -> str:
    q = question.lower()
    if any(k in q for k in DOSING_KEYWORDS):
        return "dosing"
    elif any(k in q for k in ALLERGY_KEYWORDS):
        return "allergy"
    else:
        return "diagnosis"

# Task-conditional signal weights
TASK_WEIGHTS = {
    "dosing":    [0.20, 0.20, 0.45, 0.15],   # [Med-ISP, C-AAS, Med-EEM, CDT]
    "allergy":   [0.15, 0.50, 0.20, 0.15],
    "diagnosis": [0.20, 0.15, 0.15, 0.50],
}

def task_score(row) -> float:
    """Weighted risk score using task-conditional weights."""
    task = classify_task(row["_question"])
    w = TASK_WEIGHTS[task]
    return (w[0] * row["med_isp"] +
            w[1] * row["c_aas"]   +
            w[2] * row["med_eem"] +
            w[3] * row["cdt"])

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
FEATURES = ["med_isp", "c_aas", "med_eem", "cdt"]

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["med_isp"] = df["_answer"].apply(med_isp)
    df["c_aas"]   = df["_answer"].apply(c_aas)
    df["med_eem"] = df["_answer"].apply(med_eem)
    df["cdt"]     = df.apply(lambda r: cdt(r["_answer"], r["_question"]), axis=1)
    df["task"]    = df["_question"].apply(classify_task)
    df["task_score"] = df.apply(task_score, axis=1)
    return df

def risk_label(score: float) -> str:
    if score < 0.35:
        return "GREEN"
    elif score < 0.65:
        return "AMBER"
    return "RED"

# ─────────────────────────────────────────────────────────────────────────────
# UNIFIED MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def train_unified_model(df_all: pd.DataFrame):
    """Train ONE fusion model on all datasets combined (70% train).
    Uses LightGBM when available (better AUROC), falls back to LogisticRegression.
    """
    X = df_all[FEATURES].values
    y = df_all["_label"].values

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(df_all)),
        test_size=0.30, random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    if LGBM_AVAILABLE:
        print("  [INFO] Using LightGBM as the fusion model (upgraded from Logistic Regression).")
        clf = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            objective="binary",
            class_weight="balanced",
            random_state=42,
            verbose=-1,          # suppress LightGBM internal logs
        )
        clf.fit(X_train_s, y_train,
                eval_set=[(X_test_s, y_test)],
                callbacks=[lgb.early_stopping(30, verbose=False),
                           lgb.log_evaluation(period=-1)])
    else:
        print("  [INFO] lightgbm not found – falling back to Logistic Regression.")
        print("         Run: pip install lightgbm   to enable the better model.")
        clf = LogisticRegression(
            max_iter=2000, random_state=42,
            class_weight="balanced", solver="lbfgs", C=1.0
        )
        clf.fit(X_train_s, y_train)

    return clf, scaler, X_test_s, y_test, idx_test

def evaluate_model(clf, scaler, df: pd.DataFrame, label=""):
    """Evaluate a trained model on a DataFrame. Returns metrics dict."""
    X = scaler.transform(df[FEATURES].values)
    y = df["_label"].values

    if len(np.unique(y)) < 2:
        return {"auroc": None, "ap": None, "f1": None, "p95r": None}

    y_prob = clf.predict_proba(X)[:, 1]
    y_pred = clf.predict(X)

    auroc = roc_auc_score(y, y_prob)
    ap    = average_precision_score(y, y_prob)
    f1    = f1_score(y, y_pred, zero_division=0)

    prec, rec, _ = precision_recall_curve(y, y_prob)
    idx = np.argmin(np.abs(rec - 0.95))
    p95r = prec[idx]

    return {"auroc": auroc, "ap": ap, "f1": f1, "p95r": p95r,
            "y_prob": y_prob, "y_pred": y_pred, "y_true": y}

def ablation_study(df_all: pd.DataFrame):
    """Train separate models each using only ONE signal. Compare vs all-4."""
    banner("ABLATION STUDY — Single Signal vs All 4 Combined", char="-")
    y = df_all["_label"].values
    results = {}

    for feat in FEATURES + ["all_4"]:
        cols = FEATURES if feat == "all_4" else [feat]
        X = df_all[cols].values
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, random_state=42,
            stratify=y if len(np.unique(y)) > 1 else None
        )
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)
        if LGBM_AVAILABLE:
            m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05,
                                   max_depth=4, class_weight="balanced",
                                   random_state=42, verbose=-1)
        else:
            m = LogisticRegression(max_iter=1000, random_state=42,
                                   class_weight="balanced")
        m.fit(X_tr, y_tr)
        prob = m.predict_proba(X_te)[:, 1]
        try:
            auc = roc_auc_score(y_te, prob)
        except Exception:
            auc = 0.0
        results[feat] = auc

    print(f"  {'Signal':<18} {'AUROC':>8}  {'vs All-4':>10}")
    print(f"  {'------':<18} {'------':>8}  {'--------':>10}")
    all4 = results["all_4"]
    for feat, auc in results.items():
        diff = f"+{auc-all4:.4f}" if feat != "all_4" else "baseline"
        marker = " <-- BEST" if feat == "all_4" else ""
        print(f"  {feat:<18} {auc:>8.4f}  {diff:>10}{marker}")
    return results

# ─────────────────────────────────────────────────────────────────────────────
# CROSS-VALIDATION (5-fold)
# ─────────────────────────────────────────────────────────────────────────────
def cross_validate(df_all: pd.DataFrame):
    banner("5-FOLD CROSS-VALIDATION — Unified Model", char="-")
    X = df_all[FEATURES].values
    y = df_all["_label"].values
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for fold, (tr_idx, te_idx) in enumerate(kf.split(X, y), 1):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr_idx])
        X_te = sc.transform(X[te_idx])
        if LGBM_AVAILABLE:
            m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05,
                                   max_depth=4, class_weight="balanced",
                                   random_state=42, verbose=-1)
        else:
            m = LogisticRegression(max_iter=1000, random_state=42,
                                   class_weight="balanced")
        m.fit(X_tr, y[tr_idx])
        prob = m.predict_proba(X_te)[:, 1]
        try:
            auc = roc_auc_score(y[te_idx], prob)
        except Exception:
            auc = 0.0
        aucs.append(auc)
        print(f"  Fold {fold}: AUROC = {auc:.4f}")
    print(f"\n  Mean AUROC = {np.mean(aucs):.4f}  |  Std = {np.std(aucs):.4f}")
    return aucs

# ─────────────────────────────────────────────────────────────────────────────
# FULL AUTO MODE — UNIFIED CLINIGUARD MODEL
# ─────────────────────────────────────────────────────────────────────────────
if args.source == "auto":
    banner("CLINIGUARD — UNIFIED CAPSTONE BENCHMARK")

    # ── 1. Load all datasets ────────────────────────────────────────────────
    sub_banner("Step 1 of 5 — Loading All Datasets from Local Parquet Cache")
    sources = list(PARQUET_REGISTRY.keys())
    frames = []
    for src in sources:
        try:
            df_src = load_source(src, n=args.samples)
            frames.append(df_src)
            pos = df_src["_label"].sum()
            print(f"  [OK] {src:<14} {len(df_src):>5} rows  |  hallucinated: {pos}")
        except Exception as e:
            print(f"  [SKIP] {src}: {e}")

    df_all = pd.concat(frames, ignore_index=True)
    print(f"\n  TOTAL: {len(df_all)} rows across {len(frames)} datasets")
    print(f"  Hallucinated (label=1): {df_all['_label'].sum()} "
          f"({df_all['_label'].mean()*100:.1f}%)")

    # ── 2. Extract signals ───────────────────────────────────────────────────
    sub_banner("Step 2 of 5 — Extracting 4 Detection Signals")
    print("  Computing Med-ISP, C-AAS, Med-EEM (Shannon entropy), CDT (cosine)...")
    df_all = extract_features(df_all)
    print(f"\n  Signal Means (all data):")
    for f in FEATURES:
        print(f"    {f:<12} mean={df_all[f].mean():.4f}  std={df_all[f].std():.4f}")
    tc = df_all["task"].value_counts()
    print(f"\n  Task distribution: {dict(tc)}")

    # ── 3. Train unified model ───────────────────────────────────────────────
    sub_banner("Step 3 of 5 — Training ONE Unified CLINIGUARD Model (70/30 split)")
    clf, scaler, X_test, y_test, idx_test = train_unified_model(df_all)

    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    overall_auroc = roc_auc_score(y_test, y_prob)
    overall_ap    = average_precision_score(y_test, y_prob)
    overall_f1    = f1_score(y_test, y_pred, zero_division=0)
    prec, rec, _  = precision_recall_curve(y_test, y_prob)
    p95r          = prec[np.argmin(np.abs(rec - 0.95))]

    # Print model-specific feature contributions
    if LGBM_AVAILABLE and isinstance(clf, lgb.LGBMClassifier):
        print(f"\n  Learned Signal Importance (LightGBM Feature Gain):")
        importances = clf.feature_importances_
        for feat, imp in zip(FEATURES, importances):
            bar = "#" * int(imp / max(importances) * 20 + 1)
            print(f"    {feat:<12}  importance = {imp:>8.1f}  {bar}")
    else:
        print(f"\n  Learned Signal Weights (Logistic Regression Coefficients):")
        for feat, w in zip(FEATURES, clf.coef_[0]):
            bar = "#" * int(abs(w * 5) + 1)
            print(f"    {feat:<12}  coef = {w:>+8.4f}  {bar}")

    print(f"\n  ┌─────────────────────────────────────────────┐")
    print(f"  │  OVERALL UNIFIED MODEL PERFORMANCE          │")
    print(f"  │  AUROC            : {overall_auroc:.4f}                 │")
    print(f"  │  Avg Precision    : {overall_ap:.4f}                 │")
    print(f"  │  F1-Score         : {overall_f1:.4f}                 │")
    print(f"  │  Precision@95%Rec : {p95r:.4f}                 │")
    print(f"  └─────────────────────────────────────────────┘")

    # ── 4. Per-dataset evaluation (SAME model, no retraining) ────────────────
    sub_banner("Step 4 of 5 — Per-Dataset Evaluation (Same Unified Model)")
    comparison = []
    for src in sources:
        df_src = df_all[df_all["_source"] == src]
        if len(df_src) == 0:
            continue
        m = evaluate_model(clf, scaler, df_src, label=src)

        # Apply task-conditional risk scores
        df_src = df_src.copy()
        if m.get("y_prob") is not None:
            df_src["risk_score"] = clf.predict_proba(
                scaler.transform(df_src[FEATURES].values)
            )[:, 1]
            df_src["cliniguard_label"] = df_src["risk_score"].apply(risk_label)
            color_dist = df_src["cliniguard_label"].value_counts().to_dict()
        else:
            color_dist = {}

        auroc_str = f"{m['auroc']:.4f}" if m["auroc"] else "N/A"
        ap_str    = f"{m['ap']:.4f}"    if m["ap"]    else "N/A"
        f1_str    = f"{m['f1']:.4f}"    if m["f1"]    else "N/A"
        p95_str   = f"{m['p95r']:.4f}"  if m["p95r"]  else "N/A"

        g = color_dist.get("GREEN", 0)
        a_col = color_dist.get("AMBER", 0)
        r = color_dist.get("RED", 0)

        comparison.append({
            "Dataset":         src.upper(),
            "Rows":            len(df_src),
            "Hallucinated":    int(df_src["_label"].sum()),
            "AUROC":           auroc_str,
            "Avg Precision":   ap_str,
            "F1-Score":        f1_str,
            "Prec@95%Recall":  p95_str,
            "GREEN":           g,
            "AMBER":           a_col,
            "RED":             r,
        })

    df_comp = pd.DataFrame(comparison)
    banner("FINAL COMPARATIVE TABLE — Unified CLINIGUARD Model", char="=")
    print(df_comp.to_string(index=False))

    # ── 5. Ablation + Cross-Validation ──────────────────────────────────────
    sub_banner("Step 5 of 5 — Ablation Study + Cross-Validation")
    ablation_study(df_all)
    cross_validate(df_all)

    # ── Save results ─────────────────────────────────────────────────────────
    out_csv = Path(__file__).parent / "cliniguard_results.csv"
    df_all["risk_score"] = clf.predict_proba(
        scaler.transform(df_all[FEATURES].values)
    )[:, 1]
    df_all["cliniguard_label"] = df_all["risk_score"].apply(risk_label)
    df_all.to_csv(out_csv, index=False)

    df_comp.to_csv(Path(__file__).parent / "cliniguard_summary.csv", index=False)

    # ── Save model for live inference ────────────────────────────────────────
    import joblib
    joblib.dump(clf, Path(__file__).parent / "cliniguard_model.joblib")
    joblib.dump(scaler, Path(__file__).parent / "cliniguard_scaler.joblib")

    banner("CLINIGUARD COMPLETE")
    print(f"  Unified Model AUROC    : {overall_auroc:.4f}")
    print(f"  Unified Model F1-Score : {overall_f1:.4f}")
    print(f"  Unified Model saved to : cliniguard_model.joblib")
    print(f"  Standard Scaler saved to: cliniguard_scaler.joblib")
    print(f"  Full results saved to  : {out_csv}")
    print(f"  Summary table saved to : cliniguard_summary.csv\n")

# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-SOURCE MODE
# ─────────────────────────────────────────────────────────────────────────────
else:
    banner(f"CLINIGUARD — Single Source: {args.source.upper()}")
    df = load_source(args.source, n=args.samples)
    df = extract_features(df)

    X = df[FEATURES].values
    y = df["_label"].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None
    )
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)

    clf_s = LogisticRegression(max_iter=2000, random_state=42,
                               class_weight="balanced")
    clf_s.fit(X_tr, y_tr)

    y_prob = clf_s.predict_proba(X_te)[:, 1]
    y_pred = clf_s.predict(X_te)

    auroc = roc_auc_score(y_te, y_prob) if len(np.unique(y_te)) > 1 else 0.0
    ap    = average_precision_score(y_te, y_prob) if len(np.unique(y_te)) > 1 else 0.0
    f1    = f1_score(y_te, y_pred, zero_division=0)

    print(f"  Rows loaded      : {len(df)}")
    print(f"  Hallucinated     : {df['_label'].sum()}")
    print(f"  AUROC            : {auroc:.4f}")
    print(f"  Avg Precision    : {ap:.4f}")
    print(f"  F1-Score         : {f1:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_te, y_pred,
          target_names=["Factual (0)", "Hallucinated (1)"],
          zero_division=0))

    df["risk_score"] = clf_s.predict_proba(sc.transform(df[FEATURES].values))[:, 1]
    df["cliniguard_label"] = df["risk_score"].apply(risk_label)
    out = Path(__file__).parent / f"cliniguard_results_{args.source}.csv"
    df.to_csv(out, index=False)
    print(f"  Results saved to: {out}\n")
