"""
Independent Metric Verification for Model Comparison and Benchmark Tables
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, brier_score_loss

def run_metric_verification():
    print("=" * 80)
    print("CLINIGUARD INDEPENDENT METRICS AUDIT")
    print("=" * 80)

    root = Path(__file__).resolve().parent.parent.parent

    # 1. Verify Reported Model Comparison (Phase 5 prompt table)
    print("\n[1] Audit of Reported 4-Model Comparison (dl_model_comparison.csv):")
    dl_p = root / "dl_model_comparison.csv"
    if dl_p.exists():
        df_dl = pd.read_csv(dl_p)
        print("  Saved dl_model_comparison.csv:")
        print(df_dl.to_string(index=False))

    # 2. Verify Held-out Test Set (Seed 42)
    print("\n[2] Independent Recomputation from Raw Test Predictions (N=2,075):")
    pred_p = root / "research_paper" / "results" / "predictions_seed42_test.csv"
    if pred_p.exists():
        df_preds = pd.read_csv(pred_p)
        y_true = df_preds["y"].values
        models = [c for c in df_preds.columns if c not in ("row", "source", "y")]

        rows = []
        for m in models:
            y_prob = df_preds[m].values
            auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_prob)) > 1 else 0.5
            ap = average_precision_score(y_true, y_prob)
            y_pred = (y_prob >= 0.5).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            acc = accuracy_score(y_true, y_pred)
            brier = brier_score_loss(y_true, y_prob)
            rows.append({"Model": m, "AUROC": round(auc, 4), "AP": round(ap, 4), "F1": round(f1, 4), "Acc": round(acc, 4), "Brier": round(brier, 4)})
            print(f"  {m[:28]:<28} | AUROC: {auc:.4f} | AP: {ap:.4f} | F1: {f1:.4f} | Acc: {acc:.4f} | Brier: {brier:.4f}")

        # Save to metric comparisons
        out_comp = Path(__file__).resolve().parent.parent / "evidence" / "metric_comparisons" / "recomputed_metrics_seed42.csv"
        pd.DataFrame(rows).to_csv(out_comp, index=False)
        print(f"\n[OK] Saved recalculated metrics -> {out_comp}")

if __name__ == "__main__":
    run_metric_verification()
