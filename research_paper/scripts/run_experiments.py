"""
CLINIGUARD - Rigorous Evaluation Suite on Verified Genuine Datasets
===================================================================
Operates strictly on genuine medical hallucination datasets:
  - Med-HALT (4,914 samples, genuine PubMed abstracts vs LLM-hallucinated fake abstracts)
  - MedHallu (2,000 samples, genuine paired Ground Truth vs LLM Hallucinated answers)
Total: 6,914 verified samples.

All outputs written to research_paper/results/.
Run from repository root:
    python research_paper/scripts/run_experiments.py
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    brier_score_loss,
)
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cliniguard_pipeline import extract_features, FEATURES
from load_datasets import load_genuine_datasets

OUT = ROOT / "research_paper" / "results"
OUT.mkdir(parents=True, exist_ok=True)

HEADLINE_SEED = 42
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def log(msg: str):
    print(msg, flush=True)


def p_at_95r(y_true, y_prob):
    """Maximum precision achieved at recall >= 0.95."""
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    valid = prec[rec >= 0.95]
    return float(np.max(valid)) if len(valid) > 0 else 0.0


def compute_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_indices == b
        if np.sum(mask) > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += (np.sum(mask) / n) * np.abs(bin_acc - bin_conf)
    return float(ece)


def metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "ap": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "p_at_95r": p_at_95r(y_true, y_prob),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": compute_ece(y_true, y_prob),
    }


def make_models(seed, with_mlp=True):
    m = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=seed
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            class_weight="balanced",
            random_state=seed,
            verbose=-1,
        ),
    }
    if with_mlp:
        m["MLP-4L(128-64-32-16)"] = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32, 16),
            max_iter=1000,
            random_state=seed,
            early_stopping=True,
        )
        m["MLP-2L(256-128)"] = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            max_iter=1000,
            random_state=seed,
            early_stopping=True,
        )
    return m


def tfidf_predict(tr_txt, y_tr, te_txt):
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000, sublinear_tf=True)
    Xtr = v.fit_transform(tr_txt)
    Xte = v.transform(te_txt)
    c = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, y_tr)
    return c.predict_proba(Xte)[:, 1]


def run_single_split(df, tr, te, seed, feat_cols=FEATURES, with_mlp=True):
    scaler = StandardScaler().fit(df.loc[tr, feat_cols].values)
    Xtr = scaler.transform(df.loc[tr, feat_cols].values)
    Xte = scaler.transform(df.loc[te, feat_cols].values)
    y_tr = df.loc[tr, "_label"].values

    models = make_models(seed, with_mlp=with_mlp)
    preds = {}
    for name, model in models.items():
        model.fit(Xtr, y_tr)
        preds[name] = model.predict_proba(Xte)[:, 1]

    # Baseline 1: Answer length (LR)
    sl = StandardScaler().fit(df.loc[tr, ["log_len"]].values)
    lm = LogisticRegression(class_weight="balanced", max_iter=1000).fit(
        sl.transform(df.loc[tr, ["log_len"]].values), y_tr
    )
    preds["Baseline: answer length (LR)"] = lm.predict_proba(
        sl.transform(df.loc[te, ["log_len"]].values)
    )[:, 1]

    # Baseline 2: TF-IDF on answer text (LR)
    preds["Baseline: TF-IDF answer text (LR)"] = tfidf_predict(
        df.loc[tr, "_answer"], y_tr, df.loc[te, "_answer"]
    )

    return preds


def agg(rows, keys):
    d = pd.DataFrame(rows)
    num_cols = ["auroc", "ap", "f1", "p_at_95r", "brier", "ece"]
    mean_df = d.groupby(keys)[num_cols].mean().add_suffix("_mean")
    std_df = d.groupby(keys)[num_cols].std().add_suffix("_std")
    cnt_df = d.groupby(keys).size().rename("n_runs")
    return mean_df.join(std_df).join(cnt_df).reset_index()


# ==============================================================================
# MAIN EXPERIMENT PIPELINE
# ==============================================================================
def main():
    start_time = time.time()
    log("=" * 78)
    log("CLINIGUARD: RUNNING VERIFIED EXPERIMENTS (GENUINE DATASETS ONLY)")
    log("=" * 78)

    # 1. Load genuine datasets
    df = load_genuine_datasets()
    df["_question"] = df["question"].astype(str)
    df["_context"] = df["context"].astype(str)
    df["_answer"] = df["answer"].astype(str)
    df["_text"] = df["_question"] + " [SEP] " + df["_context"]
    df["_label"] = df["label"].astype(int)
    df["_source"] = df["source"].astype(str)

    log("\nExtracting detection signals (Med-ISP, C-AAS, Med-EEM, CDT)...")
    df = extract_features(df)
    df["n_ans_tokens"] = df["_answer"].apply(lambda t: len(t.split()))
    df["log_len"] = np.log1p(df["n_ans_tokens"])
    df["group"] = pd.factorize(df["_source"] + "|" + df["_question"])[0]

    # Save full feature dataset
    df.to_csv(OUT / "features_genuine.csv", index=False)
    log(f"Saved extracted features: {len(df)} rows to {OUT / 'features_genuine.csv'}")

    # --------------------------------------------------------------------------
    # E0: Dataset Audit
    # --------------------------------------------------------------------------
    log("\n[E0] Dataset Audit (Genuine Sources)...")
    audit_rows = []
    for s, g in df.groupby("_source"):
        audit_rows.append({
            "source": s,
            "n_samples": len(g),
            "n_hallucinated": int(g["_label"].sum()),
            "pct_hallucinated": float(round(g["_label"].mean() * 100, 2)),
            "n_unique_questions": int(g["_question"].nunique()),
            "mean_tokens_label0": float(round(g[g["_label"] == 0]["n_ans_tokens"].mean(), 1)),
            "mean_tokens_label1": float(round(g[g["_label"] == 1]["n_ans_tokens"].mean(), 1)),
            "label_provenance": "Genuine (authentic PubMed vs fake abstract)" if s == "Med-HALT" else "Genuine (expert-paired Ground Truth vs LLM Hallucinated answer)",
        })
    df_audit = pd.DataFrame(audit_rows)
    df_audit.to_csv(OUT / "E0_dataset_audit.csv", index=False)
    log(df_audit.to_string(index=False))

    # --------------------------------------------------------------------------
    # E1: Headline (Seed 42) & Multi-Seed (Seeds 0-9) Evaluation
    # --------------------------------------------------------------------------
    log("\n[E1] Headline & Multi-seed Pooled Evaluation (70/30 stratified splits)...")
    rows1 = []
    P42 = None
    y_te42 = None

    for seed in SEEDS:
        tr, te = train_test_split(
            np.arange(len(df)), test_size=0.30, random_state=seed, stratify=df["_label"].values
        )
        preds = run_single_split(df, tr, te, seed)
        y_te = df.loc[te, "_label"].values
        for mname, p in preds.items():
            res = metrics(y_te, p)
            rows1.append({"seed": seed, "model": mname, **res})

    # Headline seed 42
    tr42, te42 = train_test_split(
        np.arange(len(df)), test_size=0.30, random_state=HEADLINE_SEED, stratify=df["_label"].values
    )
    P42 = run_single_split(df, tr42, te42, HEADLINE_SEED)
    y_te42 = df.loc[te42, "_label"].values

    headline_rows = []
    pred_export = pd.DataFrame({"row": te42, "source": df.loc[te42, "_source"].values, "y": y_te42})
    for mname, p in P42.items():
        res = metrics(y_te42, p)
        headline_rows.append({"seed": HEADLINE_SEED, "model": mname, **res})
        pred_export[mname] = p
    pred_export.to_csv(OUT / "predictions_seed42_test.csv", index=False)

    df_headline = pd.DataFrame(headline_rows)
    df_headline.to_csv(OUT / "E1_headline_seed42.csv", index=False)

    df_all_runs = pd.DataFrame(rows1)
    df_all_runs.to_csv(OUT / "E1_pooled_all_runs.csv", index=False)

    df_summary = agg(rows1, ["model"])
    df_summary.to_csv(OUT / "E1_pooled_10seed_summary.csv", index=False)
    log("\nHeadline Seed 42 Results:")
    log(df_headline[["model", "auroc", "ap", "f1", "p_at_95r", "brier", "ece"]].to_string(index=False))
    log("\n10-Seed Summary (Mean ± Std):")
    log(df_summary[["model", "auroc_mean", "auroc_std", "ap_mean", "f1_mean", "p_at_95r_mean"]].to_string(index=False))

    # Bootstrap 95% CIs and Statistical Significance Tests
    log("\nComputing 2,000-resample paired bootstrap for significance testing...")
    rng = np.random.RandomState(0)
    p_lgb = P42["LightGBM"]
    n_boot = 2000
    boot_indices = [rng.randint(0, len(y_te42), len(y_te42)) for _ in range(n_boot)]

    sig_rows = []
    for mname, p_other in P42.items():
        diffs = []
        lgb_aurocs = []
        other_aurocs = []
        for idx in boot_indices:
            yb = y_te42[idx]
            if len(np.unique(yb)) == 2:
                auc_lgb = roc_auc_score(yb, p_lgb[idx])
                auc_oth = roc_auc_score(yb, p_other[idx])
                lgb_aurocs.append(auc_lgb)
                other_aurocs.append(auc_oth)
                diffs.append(auc_lgb - auc_oth)
        sig_rows.append({
            "model": mname,
            "auroc": float(roc_auc_score(y_te42, p_other)),
            "auroc_95ci_low": float(np.percentile(other_aurocs, 2.5)),
            "auroc_95ci_high": float(np.percentile(other_aurocs, 97.5)),
            "delta_vs_lightgbm": float(np.mean(diffs)),
            "delta_95ci_low": float(np.percentile(diffs, 2.5)),
            "delta_95ci_high": float(np.percentile(diffs, 97.5)),
            "p_value_two_sided": float(2 * min(np.mean(np.array(diffs) <= 0), np.mean(np.array(diffs) >= 0))),
        })
    df_sig = pd.DataFrame(sig_rows)
    df_sig.to_csv(OUT / "E1_significance_tests.csv", index=False)
    log(df_sig[["model", "auroc", "auroc_95ci_low", "auroc_95ci_high", "delta_vs_lightgbm", "delta_95ci_low", "delta_95ci_high", "p_value_two_sided"]].to_string(index=False))

    # --------------------------------------------------------------------------
    # E2: Per-Dataset Performance on Held-Out Test Data
    # --------------------------------------------------------------------------
    log("\n[E2] Per-dataset Performance on Held-out Test Data (Seed 42)...")
    e2_rows = []
    test_sub = df.loc[te42].copy()
    for src, g in test_sub.groupby("_source"):
        idx_src = g.index.values
        y_src = df.loc[idx_src, "_label"].values
        row = {
            "source": src,
            "n_test": len(g),
            "n_test_pos": int(y_src.sum()),
            "n_total": int(len(df[df["_source"] == src])),
        }
        for mname in ["LightGBM", "LogisticRegression", "Baseline: answer length (LR)", "Baseline: TF-IDF answer text (LR)"]:
            p_src = P42[mname][np.isin(te42, idx_src)]
            row[f"auroc_{mname}"] = float(roc_auc_score(y_src, p_src))
            row[f"ap_{mname}"] = float(average_precision_score(y_src, p_src))
        e2_rows.append(row)
    df_e2 = pd.DataFrame(e2_rows)
    df_e2.to_csv(OUT / "E2_per_dataset_heldout.csv", index=False)
    log(df_e2.to_string(index=False))

    # --------------------------------------------------------------------------
    # E3: Question-Grouped 5-Fold Cross-Validation
    # --------------------------------------------------------------------------
    log("\n[E3] Question-Grouped 5-Fold Cross-Validation (StratifiedGroupKFold)...")
    sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=HEADLINE_SEED)
    rows3 = []
    oof_preds = {m: np.zeros(len(df)) for m in ["LightGBM", "LogisticRegression", "MLP-2L(256-128)", "MLP-4L(128-64-32-16)", "Baseline: answer length (LR)", "Baseline: TF-IDF answer text (LR)"]}

    for fold, (tr_idx, te_idx) in enumerate(sgk.split(df, df["_label"], df["group"]), 1):
        p_fold = run_single_split(df, tr_idx, te_idx, seed=fold)
        y_f = df.loc[te_idx, "_label"].values
        for mname, p in p_fold.items():
            oof_preds[mname][te_idx] = p
            rows3.append({"fold": fold, "model": mname, **metrics(y_f, p)})

    df_e3_folds = pd.DataFrame(rows3)
    df_e3_folds.to_csv(OUT / "E3_grouped_cv_all_folds.csv", index=False)

    df_e3_summary = agg(rows3, ["model"])
    df_e3_summary.to_csv(OUT / "E3_grouped_cv_summary.csv", index=False)
    log(df_e3_summary[["model", "auroc_mean", "auroc_std", "ap_mean", "f1_mean", "p_at_95r_mean"]].to_string(index=False))

    # Per-source OOF performance
    oof_src_rows = []
    for src, g in df.groupby("_source"):
        idx_src = g.index.values
        y_src = df.loc[idx_src, "_label"].values
        for mname in oof_preds:
            p_src = oof_preds[mname][idx_src]
            oof_src_rows.append({
                "source": src,
                "model": mname,
                "oof_auroc": float(roc_auc_score(y_src, p_src)),
                "oof_ap": float(average_precision_score(y_src, p_src)),
                "oof_brier": float(brier_score_loss(y_src, p_src)),
            })
    df_oof_src = pd.DataFrame(oof_src_rows)
    df_oof_src.to_csv(OUT / "E3_grouped_cv_oof_per_source.csv", index=False)

    # --------------------------------------------------------------------------
    # E4: Cross-Dataset Domain Transfer (Zero-Shot)
    # --------------------------------------------------------------------------
    log("\n[E4] Cross-Dataset Domain Transfer...")
    transfer_rows = []
    pairs = [("Med-HALT", "MedHallu"), ("MedHallu", "Med-HALT")]
    for tr_src, te_src in pairs:
        tr_idx = df[df["_source"] == tr_src].index.values
        te_idx = df[df["_source"] == te_src].index.values
        p_trans = run_single_split(df, tr_idx, te_idx, seed=HEADLINE_SEED)
        y_test_trans = df.loc[te_idx, "_label"].values
        for mname, p in p_trans.items():
            transfer_rows.append({
                "train_source": tr_src,
                "test_source": te_src,
                "model": mname,
                "auroc": float(roc_auc_score(y_test_trans, p)),
                "ap": float(average_precision_score(y_test_trans, p)),
                "f1": float(f1_score(y_test_trans, (p >= 0.5).astype(int), zero_division=0)),
            })
    df_e4 = pd.DataFrame(transfer_rows)
    df_e4.to_csv(OUT / "E4_cross_dataset_transfer.csv", index=False)
    log(df_e4.to_string(index=False))

    # --------------------------------------------------------------------------
    # E5: Feature Ablation Study
    # --------------------------------------------------------------------------
    log("\n[E5] Feature Ablation Study (10 seeds)...")
    ablation_configs = {
        "All 4 Signals": FEATURES,
        "Drop Med-ISP": [f for f in FEATURES if f != "med_isp"],
        "Drop C-AAS": [f for f in FEATURES if f != "c_aas"],
        "Drop Med-EEM": [f for f in FEATURES if f != "med_eem"],
        "Drop CDT": [f for f in FEATURES if f != "cdt"],
        "Only Med-ISP": ["med_isp"],
        "Only C-AAS": ["c_aas"],
        "Only Med-EEM": ["med_eem"],
        "Only CDT": ["cdt"],
    }
    ablation_rows = []
    for cname, fcols in ablation_configs.items():
        for seed in SEEDS:
            tr, te = train_test_split(
                np.arange(len(df)), test_size=0.30, random_state=seed, stratify=df["_label"].values
            )
            scaler = StandardScaler().fit(df.loc[tr, fcols].values)
            Xtr = scaler.transform(df.loc[tr, fcols].values)
            Xte = scaler.transform(df.loc[te, fcols].values)
            y_tr = df.loc[tr, "_label"].values
            y_te = df.loc[te, "_label"].values

            for mname, m in [("LightGBM", make_models(seed, with_mlp=False)["LightGBM"]),
                             ("LogisticRegression", make_models(seed, with_mlp=False)["LogisticRegression"])]:
                m.fit(Xtr, y_tr)
                p = m.predict_proba(Xte)[:, 1]
                ablation_rows.append({"config": cname, "model": mname, "seed": seed, **metrics(y_te, p)})

    df_abl_all = pd.DataFrame(ablation_rows)
    df_abl_all.to_csv(OUT / "E5_ablation_all_runs.csv", index=False)
    df_abl_sum = agg(ablation_rows, ["config", "model"])
    df_abl_sum.to_csv(OUT / "E5_ablation_summary.csv", index=False)
    log(df_abl_sum[["config", "model", "auroc_mean", "auroc_std", "ap_mean", "f1_mean"]].to_string(index=False))

    # --------------------------------------------------------------------------
    # E6: Univariate Feature Analysis
    # --------------------------------------------------------------------------
    log("\n[E6] Univariate Feature AUROCs...")
    u_rows = []
    for s, g in df.groupby("_source"):
        row = {"source": s, "n": len(g)}
        for f in FEATURES + ["task_score", "log_len"]:
            row[f] = float(roc_auc_score(g["_label"], g[f]))
        u_rows.append(row)
    row_all = {"source": "Combined Genuine Pool", "n": len(df)}
    for f in FEATURES + ["task_score", "log_len"]:
        row_all[f] = float(roc_auc_score(df["_label"], df[f]))
    u_rows.append(row_all)
    df_u = pd.DataFrame(u_rows)
    df_u.to_csv(OUT / "E6_univariate_signal_auroc.csv", index=False)
    log(df_u.to_string(index=False))

    # --------------------------------------------------------------------------
    # E7: Clinical Task Distribution
    # --------------------------------------------------------------------------
    log("\n[E7] Task Distribution Breakdown...")
    tc = df.groupby("task").agg(
        n=("_label", "size"),
        pct_hallucinated=("_label", lambda x: float(round(np.mean(x) * 100, 2))),
    ).reset_index()
    tc.to_csv(OUT / "E7_task_distribution.csv", index=False)
    log(tc.to_string(index=False))

    # --------------------------------------------------------------------------
    # E8: Risk Bands & Calibration Analysis
    # --------------------------------------------------------------------------
    log("\n[E8] Risk Band and Calibration Analysis...")
    band_rows = []
    for mname in ["LightGBM", "LogisticRegression"]:
        p = P42[mname]
        bands = pd.cut(p, bins=[-np.inf, 0.35, 0.65, np.inf], labels=["GREEN", "AMBER", "RED"])
        for b in ["GREEN", "AMBER", "RED"]:
            mask = bands == b
            n_b = int(np.sum(mask))
            h_rate = float(round(np.mean(y_te42[mask]) * 100, 2)) if n_b > 0 else 0.0
            band_rows.append({
                "model": mname,
                "band": b,
                "n_samples": n_b,
                "share_pct": float(round(n_b / len(y_te42) * 100, 2)),
                "hallucination_rate_pct": h_rate,
            })
    df_bands = pd.DataFrame(band_rows)
    df_bands.to_csv(OUT / "E8_band_analysis.csv", index=False)
    log(df_bands.to_string(index=False))

    # Calibration fitting (Platt scaling & Isotonic regression on grouped OOF)
    calib_metrics = {}
    for mname in ["LightGBM", "LogisticRegression"]:
        raw_p = oof_preds[mname]
        raw_ece = compute_ece(df["_label"].values, raw_p)
        raw_brier = float(brier_score_loss(df["_label"].values, raw_p))

        # Fit isotonic & sigmoid calibration
        iso = CalibratedClassifierCV(cv="prefit", method="isotonic")
        sig = CalibratedClassifierCV(cv="prefit", method="sigmoid")

        # Use headline train/test split to evaluate calibration impact
        scaler = StandardScaler().fit(df.loc[tr42, FEATURES].values)
        Xtr42 = scaler.transform(df.loc[tr42, FEATURES].values)
        Xte42 = scaler.transform(df.loc[te42, FEATURES].values)
        y_tr42 = df.loc[tr42, "_label"].values

        clf = make_models(HEADLINE_SEED, with_mlp=False)[mname].fit(Xtr42, y_tr42)
        raw_test_p = clf.predict_proba(Xte42)[:, 1]

        # Fit calibrator on 20% of train set
        tr_sub, val_sub = train_test_split(np.arange(len(tr42)), test_size=0.25, random_state=HEADLINE_SEED)
        clf_cal = make_models(HEADLINE_SEED, with_mlp=False)[mname].fit(Xtr42[tr_sub], y_tr42[tr_sub])
        iso_clf = CalibratedClassifierCV(estimator=clf_cal, method="isotonic", cv="prefit").fit(Xtr42[val_sub], y_tr42[val_sub])
        sig_clf = CalibratedClassifierCV(estimator=clf_cal, method="sigmoid", cv="prefit").fit(Xtr42[val_sub], y_tr42[val_sub])

        iso_test_p = iso_clf.predict_proba(Xte42)[:, 1]
        sig_test_p = sig_clf.predict_proba(Xte42)[:, 1]

        calib_metrics[mname] = {
            "raw_test_ece": compute_ece(y_te42, raw_test_p),
            "raw_test_brier": float(brier_score_loss(y_te42, raw_test_p)),
            "isotonic_test_ece": compute_ece(y_te42, iso_test_p),
            "isotonic_test_brier": float(brier_score_loss(y_te42, iso_test_p)),
            "sigmoid_test_ece": compute_ece(y_te42, sig_test_p),
            "sigmoid_test_brier": float(brier_score_loss(y_te42, sig_test_p)),
        }
    with open(OUT / "E8_calibration_metrics.json", "w") as f:
        json.dump(calib_metrics, f, indent=2)
    log("\nCalibration Metrics:")
    log(json.dumps(calib_metrics, indent=2))

    # --------------------------------------------------------------------------
    # E9: Latency Profiling (2,000 iterations)
    # --------------------------------------------------------------------------
    log("\n[E9] Profiling Inference Latency over 2,000 passes...")
    sample_q = df.loc[0, "_question"]
    sample_c = df.loc[0, "_context"]
    sample_a = df.loc[0, "_answer"]

    # Pre-fit model and scaler
    scaler = StandardScaler().fit(df[FEATURES].values)
    lgb_model = make_models(HEADLINE_SEED, with_mlp=False)["LightGBM"].fit(
        scaler.transform(df[FEATURES].values), df["_label"].values
    )
    lr_model = make_models(HEADLINE_SEED, with_mlp=False)["LogisticRegression"].fit(
        scaler.transform(df[FEATURES].values), df["_label"].values
    )

    t_feat, t_lgb, t_lr = [], [], []
    for _ in range(2000):
        # Feature extraction timing
        t0 = time.perf_counter()
        single_df = pd.DataFrame([{
            "_question": sample_q,
            "_context": sample_c,
            "_answer": sample_a,
            "_label": 0,
        }])
        single_feats = extract_features(single_df)
        x_vec = single_feats[FEATURES].values
        t1 = time.perf_counter()
        t_feat.append((t1 - t0) * 1000)

        # LightGBM pipeline
        t2 = time.perf_counter()
        z_vec = scaler.transform(x_vec)
        lgb_model.predict_proba(z_vec)
        t3 = time.perf_counter()
        t_lgb.append((t3 - t0) * 1000)

        # LR pipeline
        t4 = time.perf_counter()
        lr_model.predict_proba(z_vec)
        t5 = time.perf_counter()
        t_lr.append((t5 - t4 + (t1 - t0)) * 1000)

    latency_data = {
        "hardware_processor": os.environ.get("PROCESSOR_IDENTIFIER", "CPU"),
        "n_repetitions": 2000,
        "sample_answer_tokens": int(len(sample_a.split())),
        "feature_extraction_only_ms": {
            "median": float(np.median(t_feat)),
            "mean": float(np.mean(t_feat)),
            "p95": float(np.percentile(t_feat, 95)),
            "p99": float(np.percentile(t_feat, 99)),
        },
        "end_to_end_lightgbm_ms": {
            "median": float(np.median(t_lgb)),
            "mean": float(np.mean(t_lgb)),
            "p95": float(np.percentile(t_lgb, 95)),
            "p99": float(np.percentile(t_lgb, 99)),
        },
        "end_to_end_logistic_regression_ms": {
            "median": float(np.median(t_lr)),
            "mean": float(np.mean(t_lr)),
            "p95": float(np.percentile(t_lr, 95)),
            "p99": float(np.percentile(t_lr, 99)),
        },
    }
    with open(OUT / "E9_latency.json", "w") as f:
        json.dump(latency_data, f, indent=2)
    log(json.dumps(latency_data, indent=2))

    # Environment meta
    env_data = {
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "n_samples": len(df),
        "n_hallucinated": int(df["_label"].sum()),
        "sources": list(df["_source"].unique()),
        "seeds": SEEDS,
        "headline_seed": HEADLINE_SEED,
        "runtime_seconds": float(round(time.time() - start_time, 2)),
    }
    with open(OUT / "environment.json", "w") as f:
        json.dump(env_data, f, indent=2)

    log(f"\n[DONE] All experiments completed successfully in {time.time() - start_time:.1f}s.")
    log(f"Clean result files written to: {OUT}")


if __name__ == "__main__":
    main()
