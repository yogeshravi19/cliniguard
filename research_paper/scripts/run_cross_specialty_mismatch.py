"""
Pillar 2: Cross-Specialty Extrinsic Clinical Mismatch Experiment
================================================================
Constructs an authentic cross-specialty mismatch benchmark from MedQuAD NIH:
  - 1,000 Matched NIH QA pairs (Label 0: Authentic Clinical Guidance)
  - 1,000 Mismatched NIH QA pairs (Label 1: Cross-Specialty Mismatch / Extrinsic Hallucination,
    where a question about Disease A is paired with NIH guidance for Disease B != A).
Total: 2,000 samples across 1,000 unique clinical questions and disease contexts.

Evaluates:
  1. Zero-shot generalization of CliniGuard (pre-trained on Med-HALT & MedHallu) on cross-specialty mismatches.
  2. 10-seed stratified cross-validation on the cross-specialty benchmark.
  3. Feature ablation & Context Drift (CDT) power.
  4. Clinical Specificity: False-Alarm Rate on pristine NIH clinical guidelines.

Outputs written to research_paper/results/
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    brier_score_loss,
)
from sklearn.model_selection import train_test_split
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cliniguard_pipeline import extract_features, FEATURES

OUT = ROOT / "research_paper" / "results"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
RNG = np.random.RandomState(SEED)


def build_cross_specialty_benchmark(n_questions=1000):
    df_raw = pd.read_parquet(ROOT / "data_extraction" / "medquad" / "medquad.parquet")
    # Clean and filter non-empty
    df_clean = df_raw[(df_raw["question"].str.len() > 15) & (df_raw["answer"].str.len() > 30)].copy()
    df_clean["context"] = df_clean["context"].astype(str).str.strip()

    # Drop duplicate questions
    df_unique = df_clean.drop_duplicates(subset=["question"]).reset_index(drop=True)

    # Sample n_questions with diverse contexts
    selected = df_unique.sample(n=min(n_questions, len(df_unique)), random_state=SEED).reset_index(drop=True)

    matched_rows = []
    mismatched_rows = []

    all_indices = np.arange(len(selected))

    for i, r in selected.iterrows():
        q = r["question"]
        ctx_i = r["context"]
        ans_true = r["answer"]

        # Matched pair: true answer
        matched_rows.append({
            "question": q,
            "disease_context": ctx_i,
            "answer": ans_true,
            "mismatched_disease": ctx_i,
            "label": 0,
            "pair_type": "matched_authentic_nih",
        })

        # Mismatched pair: choose random j where context_j != context_i
        candidates = all_indices[selected["context"] != ctx_i]
        j = RNG.choice(candidates)
        mismatched_rows.append({
            "question": q,
            "disease_context": ctx_i,
            "answer": selected.loc[j, "answer"],
            "mismatched_disease": selected.loc[j, "context"],
            "label": 1,
            "pair_type": "cross_specialty_mismatch",
        })

    benchmark = pd.concat([pd.DataFrame(matched_rows), pd.DataFrame(mismatched_rows)], ignore_index=True)
    # Shuffle
    benchmark = benchmark.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return benchmark


def main():
    print("=" * 78)
    print("PILLAR 2: CROSS-SPECIALTY EXTRINSIC CLINICAL MISMATCH EXPERIMENT")
    print("=" * 78)

    print("\n[1/4] Constructing 2,000-sample Cross-Specialty NIH Benchmark...")
    df_bench = build_cross_specialty_benchmark(n_questions=1000)
    df_bench["_question"] = df_bench["question"]
    df_bench["_context"] = df_bench["disease_context"]
    df_bench["_answer"] = df_bench["answer"]
    df_bench["_label"] = df_bench["label"]

    print(f"Constructed benchmark: {len(df_bench)} samples")
    print(f"Matched pairs (Label 0): {(df_bench['label'] == 0).sum()}")
    print(f"Mismatched pairs (Label 1): {(df_bench['label'] == 1).sum()}")
    print("Sample mismatch example:")
    ex = df_bench[df_bench["label"] == 1].iloc[0]
    print(f"  Question ({ex['disease_context']}): {ex['question']}")
    print(f"  Mismatched Answer ({ex['mismatched_disease']}): {ex['answer'][:120]}...\n")

    print("[2/4] Extracting CliniGuard linguistic signals...")
    df_features = extract_features(df_bench)
    df_features.to_csv(OUT / "Pillar2_cross_specialty_features.csv", index=False)

    # 1. Zero-shot generalization from pre-trained CliniGuard (trained on genuine Med-HALT & MedHallu)
    print("\n[3/4] Evaluating Zero-Shot Generalization of Pre-trained CliniGuard...")
    df_genuine = pd.read_csv(OUT / "features_genuine.csv")
    scaler_pre = StandardScaler().fit(df_genuine[FEATURES].values)
    X_train_pre = scaler_pre.transform(df_genuine[FEATURES].values)
    y_train_pre = df_genuine["_label"].values

    clf_pre = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=5, num_leaves=31,
        class_weight="balanced", random_state=SEED, verbose=-1
    ).fit(X_train_pre, y_train_pre)

    lr_pre = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED).fit(
        X_train_pre, y_train_pre
    )

    X_test_mismatch = scaler_pre.transform(df_features[FEATURES].values)
    y_test_mismatch = df_features["_label"].values

    p_lgb_zero = clf_pre.predict_proba(X_test_mismatch)[:, 1]
    p_lr_zero = lr_pre.predict_proba(X_test_mismatch)[:, 1]

    zero_shot_metrics = {
        "LightGBM_ZeroShot_AUROC": float(roc_auc_score(y_test_mismatch, p_lgb_zero)),
        "LightGBM_ZeroShot_AP": float(average_precision_score(y_test_mismatch, p_lgb_zero)),
        "LogisticRegression_ZeroShot_AUROC": float(roc_auc_score(y_test_mismatch, p_lr_zero)),
        "LogisticRegression_ZeroShot_AP": float(average_precision_score(y_test_mismatch, p_lr_zero)),
        "CDT_Univariate_AUROC": float(roc_auc_score(y_test_mismatch, df_features["cdt"].values)),
        "Med_ISP_Univariate_AUROC": float(roc_auc_score(y_test_mismatch, df_features["med_isp"].values)),
        "C_AAS_Univariate_AUROC": float(roc_auc_score(y_test_mismatch, df_features["c_aas"].values)),
        "Med_EEM_Univariate_AUROC": float(roc_auc_score(y_test_mismatch, df_features["med_eem"].values)),
    }

    # Clinical Specificity on Pristine NIH Guidance (False Alarm Rate)
    # Among true matched NIH answers (Label 0), what fraction is placed in GREEN band (<0.35)?
    matched_mask = (y_test_mismatch == 0)
    p_matched = p_lgb_zero[matched_mask]
    specificity_metrics = {
        "Pristine_NIH_Answers_N": int(matched_mask.sum()),
        "Placed_in_GREEN_LowRisk_Count": int((p_matched < 0.35).sum()),
        "Pristine_Specificity_GREEN_Rate_pct": float(round((p_matched < 0.35).mean() * 100, 2)),
        "False_Alarm_RED_HighRisk_Count": int((p_matched > 0.65).sum()),
        "False_Alarm_RED_Rate_pct": float(round((p_matched > 0.65).mean() * 100, 2)),
    }

    print("\nZero-Shot Results on Cross-Specialty Mismatch:")
    for k, v in zero_shot_metrics.items():
        print(f"  {k}: {v:.4f}")
    print("\nClinical Specificity on Pristine NIH Guidelines:")
    for k, v in specificity_metrics.items():
        print(f"  {k}: {v}")

    # 2. Within-domain 10-seed cross-validation
    print("\n[4/4] 10-Seed Stratified Cross-Validation on Cross-Specialty Benchmark...")
    cv_rows = []
    for s in range(10):
        tr, te = train_test_split(np.arange(len(df_features)), test_size=0.30, random_state=s, stratify=y_test_mismatch)
        sc = StandardScaler().fit(df_features.loc[tr, FEATURES].values)
        Xtr = sc.transform(df_features.loc[tr, FEATURES].values)
        Xte = sc.transform(df_features.loc[te, FEATURES].values)
        y_tr = df_features.loc[tr, "_label"].values
        y_te = df_features.loc[te, "_label"].values

        clf = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=5, num_leaves=31,
            class_weight="balanced", random_state=s, verbose=-1
        ).fit(Xtr, y_tr)
        p_te = clf.predict_proba(Xte)[:, 1]

        cv_rows.append({
            "seed": s,
            "auroc": float(roc_auc_score(y_te, p_te)),
            "ap": float(average_precision_score(y_te, p_te)),
            "f1": float(f1_score(y_te, (p_te >= 0.5).astype(int), zero_division=0)),
            "brier": float(brier_score_loss(y_te, p_te)),
        })

    df_cv = pd.DataFrame(cv_rows)
    cv_summary = {
        "AUROC_mean": float(df_cv["auroc"].mean()),
        "AUROC_std": float(df_cv["auroc"].std()),
        "AP_mean": float(df_cv["ap"].mean()),
        "AP_std": float(df_cv["ap"].std()),
        "F1_mean": float(df_cv["f1"].mean()),
        "F1_std": float(df_cv["f1"].std()),
    }
    print(f"\n10-Seed CV LightGBM Performance: AUROC={cv_summary['AUROC_mean']:.4f} ± {cv_summary['AUROC_std']:.4f}")

    # Export all results
    df_cv.to_csv(OUT / "Pillar2_mismatch_10seed_cv.csv", index=False)
    all_pillar2_results = {
        "zero_shot_metrics": zero_shot_metrics,
        "specificity_metrics": specificity_metrics,
        "cv_summary": cv_summary,
    }
    with open(OUT / "Pillar2_mismatch_metrics.json", "w") as f:
        json.dump(all_pillar2_results, f, indent=2)

    print(f"\n[DONE] Pillar 2 results exported successfully to {OUT}")


if __name__ == "__main__":
    main()
