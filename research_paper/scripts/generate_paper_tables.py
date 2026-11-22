"""
Automated Table & Manuscript Synchronizer
=========================================
Reads all deterministic CSVs from research_paper/results/
and formats the exact tables and numbers into RESEARCH_PAPER_CLINIGUARD.md.
Eliminates any risk of manual typographical errors or drift.
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "research_paper" / "results"
MD_PATH = ROOT / "research_paper" / "RESEARCH_PAPER_CLINIGUARD.md"


def format_table_3():
    """Pooled 10-Seed Benchmark Performance with both Paired 10-Seed diff and Headline Seed 42."""
    e1_sum = pd.read_csv(RESULTS_DIR / "E1_pooled_10seed_summary.csv").set_index("model")
    e1_sig = pd.read_csv(RESULTS_DIR / "E1_significance_tests.csv").set_index("model")
    
    # Priority order for clean presentation
    model_order = [
        "Baseline: Majority Class",
        "Baseline: All Positive (Always 1)",
        "Baseline: answer length (LR)",
        "Baseline: answer length (LGB)",
        "Baseline: CDT + length (LR)",
        "Baseline: CDT + length (LGB)",
        "LogisticRegression",
        "MLP-2L(256-128)",
        "MLP-4L(128-64-32-16)",
        "LightGBM",
        "CliniGuard + length (LGB)",
        "Baseline: TF-IDF answer text (LR)",
    ]
    
    rows = []
    header = "| Model / Baseline | 10-Seed AUROC ± SD | 10-Seed AP ± SD | 10-Seed F1 (Best F1*) | Difference vs LightGBM (10-Seed 95% CI) | Paired t-test p-value |"
    sep = "|---|:---:|:---:|:---:|:---:|:---:|"
    rows.append(header)
    rows.append(sep)
    
    for m in model_order:
        if m not in e1_sum.index:
            continue
        row_sum = e1_sum.loc[m]
        row_sig = e1_sig.loc[m]
        
        display_name = m
        if m == "LightGBM":
            display_name = "**LightGBM (CliniGuard, 4 signals)**"
        elif m == "LogisticRegression":
            display_name = "Logistic Regression (4 signals)"
            
        auroc_str = f"{row_sum['auroc_mean']:.4f} ± {row_sum['auroc_std']:.4f}"
        ap_str = f"{row_sum['ap_mean']:.4f} ± {row_sum['ap_std']:.4f}"
        
        f1_val = row_sum['f1_mean']
        best_f1_val = row_sum.get('best_f1_mean', f1_val)
        f1_str = f"{f1_val:.4f} ({best_f1_val:.4f}*)" if not np.isnan(best_f1_val) else f"{f1_val:.4f}"
        
        if m == "LightGBM":
            diff_str = "**Reference**"
            p_str = "—"
        else:
            d_mean = row_sig['delta_10seed_mean']
            d_low = row_sig['delta_10seed_ci_low']
            d_high = row_sig['delta_10seed_ci_high']
            sign = "+" if d_mean >= 0 else "−"
            diff_str = f"{sign}{abs(d_mean):.4f} ({d_low:+.4f} to {d_high:+.4f})"
            p_val = row_sig['p_val_paired_ttest']
            p_str = "< 0.001" if p_val < 0.001 else f"{p_val:.3f}"
            
        rows.append(f"| {display_name} | {auroc_str} | {ap_str} | {f1_str} | {diff_str} | {p_str} |")
        
    return "\n".join(rows)


def format_table_4():
    """Question-Grouped 5-Fold Cross-Validation."""
    e3_sum = pd.read_csv(RESULTS_DIR / "E3_grouped_cv_summary.csv").set_index("model")
    
    model_order = [
        "Baseline: Majority Class",
        "Baseline: All Positive (Always 1)",
        "Baseline: answer length (LR)",
        "Baseline: answer length (LGB)",
        "Baseline: CDT + length (LR)",
        "Baseline: CDT + length (LGB)",
        "LogisticRegression",
        "MLP-2L(256-128)",
        "MLP-4L(128-64-32-16)",
        "LightGBM",
        "CliniGuard + length (LGB)",
        "Baseline: TF-IDF answer text (LR)",
    ]
    
    rows = []
    header = "| Model / Baseline | Grouped AUROC (Mean ± SD) | Grouped AP (Mean ± SD) | Grouped F1 (Mean ± SD) | Grouped Best F1* (Mean ± SD) |"
    sep = "|---|:---:|:---:|:---:|:---:|"
    rows.append(header)
    rows.append(sep)
    
    for m in model_order:
        if m not in e3_sum.index:
            continue
        r = e3_sum.loc[m]
        display_name = "**LightGBM (CliniGuard, 4 signals)**" if m == "LightGBM" else m
        if m == "LogisticRegression":
            display_name = "Logistic Regression (4 signals)"
            
        auroc_str = f"{r['auroc_mean']:.4f} ± {r['auroc_std']:.4f}"
        ap_str = f"{r['ap_mean']:.4f} ± {r['ap_std']:.4f}"
        f1_str = f"{r['f1_mean']:.4f} ± {r['f1_std']:.4f}"
        best_f1_str = f"{r.get('best_f1_mean', r['f1_mean']):.4f} ± {r.get('best_f1_std', r['f1_std']):.4f}"
        
        rows.append(f"| {display_name} | {auroc_str} | {ap_str} | {f1_str} | {best_f1_str} |")
        
    return "\n".join(rows)


def format_table_5():
    """Per-Dataset Held-Out Performance."""
    e2 = pd.read_csv(RESULTS_DIR / "E2_per_dataset_heldout.csv")
    
    rows = []
    header = "| Benchmark | Test N | Hallucinated N | LightGBM AUROC | LightGBM F1 (Best F1*) | CDT + Length (LGB) AUROC | CDT + Length (LR) AUROC | Answer Length (LR) AUROC | TF-IDF AUROC | All-Positive F1 |"
    sep = "|---|---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    rows.append(header)
    rows.append(sep)
    
    for _, r in e2.iterrows():
        src = r['source']
        n_test = r['n_test']
        n_pos = r['n_test_pos']
        
        lgb_auc = f"{r.get('auroc_LightGBM', 0):.4f}"
        lgb_f1 = f"{r.get('f1_LightGBM', 0):.4f} ({r.get('best_f1_LightGBM', r.get('f1_LightGBM', 0)):.4f}*)"
        cdt_lgb_auc = f"{r.get('auroc_Baseline: CDT + length (LGB)', 0):.4f}"
        cdt_lr_auc = f"{r.get('auroc_Baseline: CDT + length (LR)', 0):.4f}"
        len_lr_auc = f"{r.get('auroc_Baseline: answer length (LR)', 0):.4f}"
        tfidf_auc = f"{r.get('auroc_Baseline: TF-IDF answer text (LR)', 0):.4f}"
        all_pos_f1 = f"{r.get('f1_Baseline: All Positive (Always 1)', 0):.4f}"
        
        rows.append(f"| **{src}** | {n_test:,} | {n_pos:,} | {lgb_auc} | {lgb_f1} | {cdt_lgb_auc} | {cdt_lr_auc} | {len_lr_auc} | {tfidf_auc} | {all_pos_f1} |")
        
    return "\n".join(rows)


def format_table_6():
    """Cross-Dataset Domain Transfer."""
    e4 = pd.read_csv(RESULTS_DIR / "E4_cross_dataset_transfer.csv")
    
    model_order = [
        "LightGBM",
        "LogisticRegression",
        "MLP-2L(256-128)",
        "MLP-4L(128-64-32-16)",
        "Baseline: CDT + length (LR)",
        "Baseline: CDT + length (LGB)",
        "Baseline: answer length (LR)",
        "Baseline: answer length (LGB)",
        "Baseline: TF-IDF answer text (LR)",
    ]
    
    rows = []
    header = "| Training Source | Evaluation Target | Model | AUROC | AP | F1 (Best F1*) | Accuracy |"
    sep = "|---|---|---|:---:|:---:|:---:|:---:|"
    rows.append(header)
    rows.append(sep)
    
    for tr_src in ["Med-HALT", "MedHallu"]:
        sub_tr = e4[e4['train_source'] == tr_src]
        for m in model_order:
            row_m = sub_tr[sub_tr['model'] == m]
            if len(row_m) == 0:
                continue
            r = row_m.iloc[0]
            te_src = r['test_source']
            disp_m = m
            if m == "LightGBM":
                disp_m = "LightGBM (4 signals)"
            elif m == "LogisticRegression":
                disp_m = "Logistic Regression (4 signals)"
                
            auc_str = f"**{r['auroc']:.4f}**" if r['auroc'] < 0.50 or r['auroc'] > 0.85 else f"{r['auroc']:.4f}"
            ap_str = f"{r['ap']:.4f}"
            best_f1_val = r.get('best_f1', r['f1'])
            f1_str = f"{r['f1']:.4f} ({best_f1_val:.4f}*)" if not np.isnan(best_f1_val) else f"{r['f1']:.4f}"
            acc_str = f"{r['acc']:.4f}"
            
            rows.append(f"| {tr_src} | {te_src} | {disp_m} | {auc_str} | {ap_str} | {f1_str} | {acc_str} |")
            
    return "\n".join(rows)


def format_table_7():
    """Feature Ablation Study."""
    e5 = pd.read_csv(RESULTS_DIR / "E5_ablation_summary.csv")
    
    cfg_order = [
        "All 4 Signals",
        "Drop Med-ISP",
        "Drop Med-EEM",
        "Drop C-AAS",
        "Drop CDT",
        "Only CDT",
        "Only C-AAS",
        "Only Med-EEM",
        "Only Med-ISP",
    ]
    
    rows = []
    header = "| Feature Configuration | LightGBM AUROC ± SD | LightGBM AP ± SD | LightGBM F1 ± SD | Logistic Reg. AUROC ± SD | Logistic Reg. AP ± SD |"
    sep = "|---|:---:|:---:|:---:|:---:|:---:|"
    rows.append(header)
    rows.append(sep)
    
    for cfg in cfg_order:
        sub_cfg = e5[e5['config'] == cfg]
        if len(sub_cfg) == 0:
            continue
        r_lgb = sub_cfg[sub_cfg['model'] == 'LightGBM'].iloc[0]
        r_lr = sub_cfg[sub_cfg['model'] == 'LogisticRegression'].iloc[0]
        
        disp_cfg = f"**{cfg}**" if cfg in ["All 4 Signals", "Drop CDT", "Only CDT"] else cfg
        lgb_auc = f"{r_lgb['auroc_mean']:.4f} ± {r_lgb['auroc_std']:.4f}"
        lgb_ap = f"{r_lgb['ap_mean']:.4f} ± {r_lgb['ap_std']:.4f}"
        lgb_f1 = f"{r_lgb['f1_mean']:.4f} ± {r_lgb['f1_std']:.4f}"
        lr_auc = f"{r_lr['auroc_mean']:.4f} ± {r_lr['auroc_std']:.4f}"
        lr_ap = f"{r_lr['ap_mean']:.4f} ± {r_lr['ap_std']:.4f}"
        
        rows.append(f"| {disp_cfg} | {lgb_auc} | {lgb_ap} | {lgb_f1} | {lr_auc} | {lr_ap} |")
        
    return "\n".join(rows)


def format_table_8():
    """Univariate Signal AUROC."""
    e6 = pd.read_csv(RESULTS_DIR / "E6_univariate_signal_auroc.csv")
    rows = []
    header = "| Dataset Cohort | Sample N | Med-ISP | C-AAS | Med-EEM | CDT | Log Answer Length |"
    sep = "|---|---:|:---:|:---:|:---:|:---:|:---:|"
    rows.append(header)
    rows.append(sep)
    
    for _, r in e6.iterrows():
        src = r['source']
        n = int(r['n'])
        isp = f"{r['med_isp']:.4f}"
        aas = f"{r['c_aas']:.4f}"
        eem = f"{r['med_eem']:.4f}"
        cdt = f"**{r['cdt']:.4f}**"
        len_ = f"{r['log_len']:.4f}"
        rows.append(f"| **{src}** | {n:,} | {isp} | {aas} | {eem} | {cdt} | {len_} |")
        
    return "\n".join(rows)


def format_table_9():
    """Risk Tiers, False Reassurance, and Random Router Lift."""
    e8 = pd.read_csv(RESULTS_DIR / "E8_band_analysis.csv")
    
    rows = []
    header = "| Cohort | Evaluated Model | Operational Tier | Coverage Share (%) | Hallucinations in Tier | Tier Error Rate (%) | False-Reassurance Rate (FRR) | Random Router Expected FRR | Safety Lift vs Random |"
    sep = "|---|---|---|---:|---:|---:|---:|---:|---:|"
    rows.append(header)
    rows.append(sep)
    
    for cohort in ["Pooled Test", "Med-HALT", "MedHallu"]:
        sub_c = e8[e8['source'] == cohort]
        for mname in ["LightGBM", "LogisticRegression", "Baseline: CDT + length (LGB)", "Baseline: answer length (LR)"]:
            sub_m = sub_c[sub_c['model'] == mname]
            if len(sub_m) == 0:
                continue
            for _, r in sub_m.iterrows():
                tier = r['band']
                share = f"{r['share_pct']:.2f}%"
                h_count = f"{int(r['hallucinations_in_band']):,}"
                h_rate = f"{r['hallucination_rate_pct']:.2f}%"
                
                if tier == "GREEN":
                    frr = f"**{r['false_reassurance_rate_pct']:.2f}%**"
                    rand_frr = f"{r.get('random_expected_frr_pct', r['share_pct']):.2f}%"
                    lift = f"{r.get('safety_lift_vs_random', 0):+.2%}"
                else:
                    frr = "—"
                    rand_frr = "—"
                    lift = "—"
                    
                rows.append(f"| {cohort} | {mname} | {tier} | {share} | {h_count} | {h_rate} | {frr} | {rand_frr} | {lift} |")
                
    return "\n".join(rows)


def format_table_tfidf_diag():
    """Top diagnostic features in Med-HALT and MedHallu."""
    p_diag = RESULTS_DIR / "E1_tfidf_top_features.csv"
    if not p_diag.exists():
        return ""
    df_d = pd.read_csv(p_diag)
    
    rows = []
    header = "| Dataset | Rank | Hallucination-Predictive Feature (Positive) | Coef | Truth-Predictive Feature (Negative) | Coef |"
    sep = "|---|---:|---|---:|---|---:|"
    rows.append(header)
    rows.append(sep)
    
    for src in ["Med-HALT", "MedHallu"]:
        sub = df_d[df_d['source'] == src].head(10)
        for _, r in sub.iterrows():
            rows.append(f"| **{src}** | {int(r['rank'])} | `{r['pos_feature_predicts_hallucination']}` | {r['pos_coef']:+.4f} | `{r['neg_feature_predicts_truth']}` | {r['neg_coef']:+.4f} |")
            
    return "\n".join(rows)


if __name__ == "__main__":
    print("Testing table formatters...")
    print("\n--- TABLE 3 ---")
    print(format_table_3())
    print("\n--- TABLE 5 ---")
    print(format_table_5())
