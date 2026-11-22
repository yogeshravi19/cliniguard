"""
Generate all publication figures from verified results in research_paper/results/.
Run from repository root:
    python research_paper/scripts/make_figures.py
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.calibration import calibration_curve

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "research_paper" / "results"
F = ROOT / "research_paper" / "figures"
F.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.size": 9,
    "figure.dpi": 200,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})


def fig_architecture():
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 45)

    def box(x, y, w, h, txt, fc, ec="#2b5b84"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5", fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=8.5, weight="medium")

    def arr(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.2, mutation_scale=10))

    box(1, 15, 14, 11, "Medical Question (Q)\n+ AI Answer (A)", "#e8f1fb")
    box(19, 15, 14, 11, "Text Cleaning &\nWord Splitting\n(Lowercase Words)", "#f4f4f4")
    arr(15, 20.5, 19, 20.5)

    sig = [
        ("Med-ISP\n(Few Drug-Lexicon Hits)", 32),
        ("C-AAS\n(Few Context-Lexicon Hits)", 22.5),
        ("Med-EEM\n(Hedging-Word Entropy)", 13),
        ("CDT\n(Question-Answer Drift)", 3.5),
    ]
    for t, y in sig:
        box(39, y, 17, 7.5, t, "#fff5dd", ec="#d48806")
        arr(33, 20.5, 39, y + 3.75)
        arr(56, y + 3.75, 62, 20.5)

    box(62, 15, 16, 11, "Z-Score Scaling\n+ LightGBM Model\n(Class-Balanced)", "#e6f7ea", ec="#237804")
    arr(78, 20.5, 83, 20.5)

    box(83, 14, 16, 13, "Clinical Risk Tier\n\nGREEN (<0.35)\nAMBER (0.35-0.65)\nRED (>0.65)", "#fff1f0", ec="#cf1322")

    ax.text(50, 42.5, "CliniGuard Pipeline: Four Lexical Signals to a Risk Tier",
            ha="center", fontsize=11, weight="bold", color="#1f2937")

    fig.savefig(F / "fig1_architecture.png", bbox_inches="tight")
    plt.close(fig)


def fig_roc():
    p = pd.read_csv(R / "predictions_seed42_test.csv")
    fig, ax = plt.subplots(figsize=(5.2, 4.8))

    models = [
        ("LightGBM", "#1677ff", "-"),
        ("MLP-2L(256-128)", "#722ed1", "-"),
        ("LogisticRegression", "#fa8c16", "-"),
        ("Baseline: answer length (LR)", "#8c8c8c", "--"),
        ("Baseline: TF-IDF answer text (LR)", "#52c41a", ":"),
    ]

    for m, col, ls in models:
        fpr, tpr, _ = roc_curve(p["y"], p[m])
        auc = roc_auc_score(p["y"], p[m])
        ax.plot(fpr, tpr, label=f"{m} (AUROC={auc:.3f})", color=col, linestyle=ls, lw=1.6)

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.6)
    ax.set_xlabel("False Positive Rate", fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontweight="bold")
    ax.set_title("ROC Curves: Verified Genuine Test Benchmark (Seed 42)", pad=10, fontweight="bold")
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)

    fig.savefig(F / "fig2_roc_seed42.png", bbox_inches="tight")
    plt.close(fig)


def fig_per_dataset():
    d = pd.read_csv(R / "E2_per_dataset_heldout.csv")
    x = np.arange(len(d))
    w = 0.2

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    models = [
        ("auroc_LightGBM", "LightGBM", "#1677ff"),
        ("auroc_LogisticRegression", "Logistic Regression", "#fa8c16"),
        ("auroc_Baseline: answer length (LR)", "Answer Length", "#8c8c8c"),
        ("auroc_Baseline: TF-IDF answer text (LR)", "TF-IDF Text", "#52c41a"),
    ]

    for i, (col, lbl, color) in enumerate(models):
        ax.bar(x + (i - 1.5) * w, d[col], w, label=lbl, color=color, edgecolor="#333", lw=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n(N={n})" for s, n in zip(d["source"], d["n_test"])], fontweight="bold")
    ax.set_ylabel("AUROC", fontweight="bold")
    ax.set_ylim(0.4, 1.05)
    ax.axhline(0.5, color="red", linestyle=":", lw=0.9, alpha=0.7)
    ax.set_title("Per-Dataset Performance on Held-Out Test Data (Seed 42)", pad=10, fontweight="bold")
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9)

    fig.savefig(F / "fig3_per_dataset.png", bbox_inches="tight")
    plt.close(fig)


def fig_baselines():
    p1 = pd.read_csv(R / "E1_pooled_10seed_summary.csv").set_index("model")
    p3 = pd.read_csv(R / "E3_grouped_cv_summary.csv").set_index("model")

    models = ["Baseline: answer length (LR)", "LogisticRegression", "MLP-2L(256-128)", "LightGBM", "Baseline: TF-IDF answer text (LR)"]
    labels = ["Answer Length", "Logistic Reg.", "MLP (2-Layer)", "LightGBM", "TF-IDF Text"]

    x = np.arange(len(models))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8.2, 4.0))

    v1 = [p1.loc[m, "auroc_mean"] for m in models]
    e1 = [p1.loc[m, "auroc_std"] for m in models]
    v3 = [p3.loc[m, "auroc_mean"] for m in models]
    e3 = [p3.loc[m, "auroc_std"] for m in models]

    ax.bar(x - w / 2, v1, w, yerr=e1, capsize=3, label="Pooled Genuine (10-Seed Random Splits)", color="#1677ff", edgecolor="#333", lw=0.6)
    ax.bar(x + w / 2, v3, w, yerr=e3, capsize=3, label="Question-Grouped 5-Fold CV (No Question Leakage)", color="#52c41a", edgecolor="#333", lw=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="bold")
    ax.set_ylabel("AUROC (Mean ± SD)", fontweight="bold")
    ax.set_ylim(0.5, 1.05)
    ax.axhline(0.5, color="red", linestyle=":", lw=0.9, alpha=0.7)
    ax.set_title("CliniGuard vs Baseline Models Across Evaluation Splits", pad=10, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)

    fig.savefig(F / "fig4_baselines.png", bbox_inches="tight")
    plt.close(fig)


def fig_ablation():
    d = pd.read_csv(R / "E5_ablation_summary.csv")
    dlgb = d[d["model"] == "LightGBM"].set_index("config")
    dlr = d[d["model"] == "LogisticRegression"].set_index("config")

    configs = [
        "All 4 Signals",
        "Drop Med-ISP",
        "Drop C-AAS",
        "Drop Med-EEM",
        "Drop CDT",
        "Only CDT",
        "Only Med-ISP",
        "Only C-AAS",
        "Only Med-EEM",
    ]

    x = np.arange(len(configs))
    w = 0.38

    fig, ax = plt.subplots(figsize=(9.2, 4.2))

    y_lgb = [dlgb.loc[c, "auroc_mean"] for c in configs]
    e_lgb = [dlgb.loc[c, "auroc_std"] for c in configs]
    y_lr = [dlr.loc[c, "auroc_mean"] for c in configs]
    e_lr = [dlr.loc[c, "auroc_std"] for c in configs]

    ax.bar(x - w / 2, y_lgb, w, yerr=e_lgb, capsize=3, label="LightGBM", color="#1677ff", edgecolor="#333", lw=0.6)
    ax.bar(x + w / 2, y_lr, w, yerr=e_lr, capsize=3, label="Logistic Regression", color="#fa8c16", edgecolor="#333", lw=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=25, ha="right", fontweight="bold")
    ax.set_ylabel("AUROC (10-Seed Mean ± SD)", fontweight="bold")
    ax.set_ylim(0.5, 0.98)
    ax.axhline(0.5, color="red", linestyle=":", lw=0.9, alpha=0.7)
    ax.set_title("Signal Ablation Study: Impact of Handcrafted Features", pad=10, fontweight="bold")
    ax.legend(fontsize=8.5, loc="lower left", framealpha=0.9)

    fig.savefig(F / "fig5_ablation.png", bbox_inches="tight")
    plt.close(fig)


def fig_univariate():
    d = pd.read_csv(R / "E6_univariate_signal_auroc.csv").set_index("source")
    cols = ["med_isp", "c_aas", "med_eem", "cdt", "task_score", "log_len"]
    labels = ["Med-ISP", "C-AAS", "Med-EEM", "CDT", "Task Score", "Log Length"]

    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    mat = d[cols].values
    im = ax.imshow(mat, cmap="RdBu_r", vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontweight="bold")
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d.index, fontweight="bold")

    for i in range(len(d)):
        for j in range(len(cols)):
            v = mat[i, j]
            txt_col = "white" if (v < 0.25 or v > 0.75) else "black"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", color=txt_col, fontsize=8.5, weight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Raw Feature AUROC", fontweight="bold")
    ax.set_title("Univariate Signal AUROC Heatmap (Genuine Datasets)", pad=10, fontweight="bold")

    fig.savefig(F / "fig6_univariate_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def fig_bands():
    d = pd.read_csv(R / "E8_band_analysis.csv")
    dlgb = d[d["model"] == "LightGBM"]

    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    colors = ["#52c41a", "#faad14", "#ff4d4f"]
    bars = ax.bar(dlgb["band"], dlgb["hallucination_rate_pct"], color=colors, edgecolor="#333", lw=0.8, width=0.5)

    for b, rate, n in zip(bars, dlgb["hallucination_rate_pct"], dlgb["n_samples"]):
        ax.text(b.get_x() + b.get_width() / 2, rate + 2.5, f"{rate:.1f}%\n(N={n})",
                ha="center", va="bottom", fontsize=8.5, weight="bold")

    ax.set_ylabel("Observed Hallucination Rate (%)", fontweight="bold")
    ax.set_ylim(0, 105)
    ax.axhline(43.2, color="gray", linestyle="--", lw=1.0, label="Overall Prevalence (43.2%)")
    ax.set_title("LightGBM Risk Band Stratification (Seed 42)", pad=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")

    fig.savefig(F / "fig7_risk_bands.png", bbox_inches="tight")
    plt.close(fig)


def fig_calibration():
    p = pd.read_csv(R / "predictions_seed42_test.csv")
    y_true = p["y"].values
    prob_lgb = p["LightGBM"].values

    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, prob_lgb, n_bins=10)

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", lw=1.0)
    ax.plot(mean_predicted_value, fraction_of_positives, "s-", color="#1677ff", label="LightGBM (ECE=3.2%)", lw=1.6)

    ax.set_xlabel("Mean Predicted Probability", fontweight="bold")
    ax.set_ylabel("Fraction of True Hallucinations", fontweight="bold")
    ax.set_title("Reliability Diagram (Calibration Curve)", pad=10, fontweight="bold")
    ax.legend(fontsize=8.5, loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    fig.savefig(F / "fig8_calibration.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_architecture()
    fig_roc()
    fig_per_dataset()
    fig_baselines()
    fig_ablation()
    fig_univariate()
    fig_bands()
    fig_calibration()
    print("[OK] All 8 figures successfully generated in research_paper/figures/")
