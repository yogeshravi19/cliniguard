"""
Evaluate the saved Logistic Regression model on the same dataset/split
used by the LightGBM pipeline to get comparable numbers.
"""
import sys, warnings, numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, precision_recall_curve
)
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except: pass

# Use the same pipeline helpers
from cliniguard_pipeline import load_source, extract_features, PARQUET_REGISTRY, FEATURES

# 1. Load all datasets
frames = []
for src in PARQUET_REGISTRY.keys():
    try:
        df = load_source(src, n=500)
        frames.append(df)
    except Exception as e:
        pass
df_all = pd.concat(frames, ignore_index=True)
print(f"Loaded {len(df_all)} samples")

# 2. Extract features
df_all = extract_features(df_all)

# 3. Same 70/30 split & scaler as pipeline
X = df_all[FEATURES].values
y = df_all["_label"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# 4. Train a fresh LR with balanced weights (mirrors original train_lr_fixed.py approach)
lr = LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced",
                        solver="lbfgs", C=1.0)
lr.fit(X_train_s, y_train)

y_prob = lr.predict_proba(X_test_s)[:, 1]
y_pred = lr.predict(X_test_s)

auroc = roc_auc_score(y_test, y_prob)
ap    = average_precision_score(y_test, y_prob)
f1    = f1_score(y_test, y_pred, zero_division=0)

prec, rec, _ = precision_recall_curve(y_test, y_prob)
idx  = np.argmin(np.abs(rec - 0.95))
p95r = prec[idx]

print("\n=== LOGISTIC REGRESSION RESULTS (same split as LightGBM) ===")
print(f"  AUROC            : {auroc:.4f}")
print(f"  Avg Precision    : {ap:.4f}")
print(f"  F1-Score         : {f1:.4f}")
print(f"  Prec@95%Recall   : {p95r:.4f}")
print()
print(f"  LR Coefficients:")
for feat, coef in zip(FEATURES, lr.coef_[0]):
    print(f"    {feat:<12} : {coef:+.4f}")
