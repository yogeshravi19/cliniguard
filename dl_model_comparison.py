import sys
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, precision_recall_curve
import lightgbm as lgb

# Import existing pipeline utilities
from cliniguard_pipeline import load_source, extract_features, PARQUET_REGISTRY, FEATURES

warnings.filterwarnings("ignore")

def main():
    print("==========================================================")
    print(" CLINIGUARD — Deep Learning Model Comparison")
    print("==========================================================")

    # 1. Load data
    print("\n[1/4] Loading datasets from parquet cache...")
    frames = []
    for src in PARQUET_REGISTRY.keys():
        try:
            # We use 500 samples per dataset as in the default pipeline
            df_src = load_source(src, n=500)
            frames.append(df_src)
        except Exception as e:
            pass
            
    if not frames:
        print("Error: No data loaded. Make sure the parquet files exist.")
        return
        
    df_all = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df_all)} samples across {len(frames)} datasets.")

    # 2. Extract features
    print("\n[2/4] Extracting signals (Med-ISP, C-AAS, Med-EEM, CDT)...")
    df_all = extract_features(df_all)

    # 3. Train Test Split & Scale
    print("\n[3/4] Preparing data (70/30 split, standard scaling)...")
    X = df_all[FEATURES].values
    y = df_all["_label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 4. Define and Train Models
    print("\n[4/4] Training and Evaluating Models...")
    
    # We compare LightGBM with two Neural Network architectures (MLPs)
    models = {
        "LightGBM (Gradient Boosting)": lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=5, 
            class_weight="balanced", random_state=42, verbose=-1
        ),
        "Deep Neural Network (4 Layers)": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32, 16), activation='relu',
            solver='adam', max_iter=1000, random_state=42, early_stopping=True
        ),
        "Wide Neural Network (2 Layers)": MLPClassifier(
            hidden_layer_sizes=(256, 128), activation='relu',
            solver='adam', max_iter=1000, random_state=42, early_stopping=True
        )
    }

    results = []
    for name, model in models.items():
        print(f"  -> Training {name}...")
        model.fit(X_train_s, y_train)
        
        y_prob = model.predict_proba(X_test_s)[:, 1]
        y_pred = model.predict(X_test_s)
        
        auroc = roc_auc_score(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        idx = np.argmin(np.abs(rec - 0.95))
        p95r = prec[idx]
        
        results.append({
            "Model": name,
            "AUROC": f"{auroc:.4f}",
            "Avg Precision": f"{ap:.4f}",
            "F1-Score": f"{f1:.4f}",
            "Prec@95%Recall": f"{p95r:.4f}"
        })

    # Output comparison
    df_res = pd.DataFrame(results)
    print("\n==========================================================")
    print(" FINAL COMPARISON TABLE")
    print("==========================================================")
    print(df_res.to_string(index=False))
    
    # Save the comparison to CSV
    out_csv = Path(__file__).parent / "dl_model_comparison.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\nComparison results saved to: {out_csv}")

if __name__ == "__main__":
    main()
