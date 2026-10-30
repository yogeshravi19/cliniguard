"""
Script to display and audit baseline model comparisons across all evaluation protocols.
"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "research_paper" / "results"

print("=" * 80)
print("1. 10-SEED POOLED SUMMARY (70/30 Stratified Splits; N=2,075 per test split)")
print("=" * 80)
df1 = pd.read_csv(RES / "E1_pooled_10seed_summary.csv")
for _, r in df1.iterrows():
    print(f"{r['model']:<32} | AUROC: {r['auroc_mean']:.4f} +/- {r['auroc_std']:.4f} | AP: {r['ap_mean']:.4f} +/- {r['ap_std']:.4f} | F1: {r['f1_mean']:.4f} | P@95R: {r['p_at_95r_mean']:.4f} | Brier: {r['brier_mean']:.4f} | ECE: {r['ece_mean']:.4f}")

print("\n" + "=" * 80)
print("2. HEADLINE SEED 42 SPLIT")
print("=" * 80)
df2 = pd.read_csv(RES / "E1_headline_seed42.csv")
for _, r in df2.iterrows():
    print(f"{r['model']:<32} | AUROC: {r['auroc']:.4f} | AP: {r['ap']:.4f} | F1: {r['f1']:.4f} | P@95R: {r['p_at_95r']:.4f} | Brier: {r['brier']:.4f} | ECE: {r['ece']:.4f}")

print("\n" + "=" * 80)
print("3. STATISTICAL SIGNIFICANCE TESTS (2,000-Resample Paired Bootstrap vs LightGBM)")
print("=" * 80)
df3 = pd.read_csv(RES / "E1_significance_tests.csv")
for _, r in df3.iterrows():
    print(f"{r['model']:<32} | AUROC: {r['auroc']:.4f} | Delta: {r['delta_vs_lightgbm']:+.4f} (95% CI: [{r['delta_95ci_low']:+.4f}, {r['delta_95ci_high']:+.4f}]) | p-val: {r['p_value_two_sided']:.4f}")

print("\n" + "=" * 80)
print("4. QUESTION-GROUPED 5-FOLD CV (Zero Question Leakage; StratifiedGroupKFold)")
print("=" * 80)
df4 = pd.read_csv(RES / "E3_grouped_cv_summary.csv")
for _, r in df4.iterrows():
    print(f"{r['model']:<32} | AUROC: {r['auroc_mean']:.4f} +/- {r['auroc_std']:.4f} | AP: {r['ap_mean']:.4f} +/- {r['ap_std']:.4f} | F1: {r['f1_mean']:.4f} | P@95R: {r['p_at_95r_mean']:.4f}")

print("\n" + "=" * 80)
print("5. ZERO-SHOT CROSS-DATASET DOMAIN TRANSFER")
print("=" * 80)
df5 = pd.read_csv(RES / "E4_cross_dataset_transfer.csv")
for _, r in df5.iterrows():
    print(f"{r['train_source']} -> {r['test_source']:<10} | {r['model']:<30} | AUROC: {r['auroc']:.4f} | AP: {r['ap']:.4f} | F1: {r['f1']:.4f}")
