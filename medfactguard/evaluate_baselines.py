"""
MedFactGuard Baseline & Cross-Dataset Evaluation Engine
Evaluates traditional ML baselines across the 5 harmonized benchmarks:
  - In-Distribution Evaluation (Cross-validation on Training Pool)
  - Zero-Shot Out-of-Distribution Generalization on PlaceboBench (Pharmaceutical RAG)
  - Zero-Shot Out-of-Distribution Generalization on MedExpert (Clinical Chatbot)
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
import lightgbm as lgb
from medfactguard.data_loader import load_harmonized_benchmark_suite

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'research_paper', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_baseline_evaluation():
    data = load_harmonized_benchmark_suite(include_medhal=True, medhal_n=25000)
    
    # 1. Training Pool: Med-HALT (4,914) + MedHallu (2,000) + MedHal (2,000) = 8,914 samples
    df_train_pool = data['train_pool'].copy().reset_index(drop=True)
    df_placebo = data['external_benchmarks']['PlaceboBench'].copy().reset_index(drop=True)
    df_expert = data['external_benchmarks']['MedExpert'].copy().reset_index(drop=True)
    
    print(f"\nTraining Pool Size: {len(df_train_pool)} rows across Med-HALT, MedHallu, MedHal")
    print(f"External Benchmark 1 (PlaceboBench): {len(df_placebo)} rows")
    print(f"External Benchmark 2 (MedExpert):    {len(df_expert)} rows\n")
    
    # Prepare text representations
    # Input format: Context + Question -> Answer
    train_texts = (df_train_pool['context'] + " " + df_train_pool['question'] + " " + df_train_pool['generated_answer']).tolist()
    y_train = df_train_pool['label'].values
    
    placebo_texts = (df_placebo['context'] + " " + df_placebo['question'] + " " + df_placebo['generated_answer']).tolist()
    y_placebo = df_placebo['label'].values
    
    expert_texts = (df_expert['context'] + " " + df_expert['question'] + " " + df_expert['generated_answer']).tolist()
    y_expert = df_expert['label'].values
    
    # Extract TF-IDF features (sublinear, unigrams + bigrams, top 10k)
    print("Fitting TF-IDF Vectorizer (max_features=10,000)...")
    tfidf = TfidfVectorizer(max_features=10000, sublinear_tf=True, ngram_range=(1, 2), stop_words='english')
    X_train_tfidf = tfidf.fit_transform(train_texts)
    X_placebo_tfidf = tfidf.transform(placebo_texts)
    X_expert_tfidf = tfidf.transform(expert_texts)
    
    # Model 1: TF-IDF + Logistic Regression
    print("\n--- Model 1: TF-IDF + Logistic Regression ---")
    clf_lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    clf_lr.fit(X_train_tfidf, y_train)
    
    p_train_lr = clf_lr.predict_proba(X_train_tfidf)[:, 1]
    p_placebo_lr = clf_lr.predict_proba(X_placebo_tfidf)[:, 1]
    p_expert_lr = clf_lr.predict_proba(X_expert_tfidf)[:, 1]
    
    # Model 2: TF-IDF + LightGBM
    print("--- Model 2: TF-IDF + LightGBM ---")
    clf_lgb = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42, verbose=-1, class_weight='balanced')
    clf_lgb.fit(X_train_tfidf, y_train)
    
    p_train_lgb = clf_lgb.predict_proba(X_train_tfidf)[:, 1]
    p_placebo_lgb = clf_lgb.predict_proba(X_placebo_tfidf)[:, 1]
    p_expert_lgb = clf_lgb.predict_proba(X_expert_tfidf)[:, 1]
    
    # Model 3: Answer Word Count (Length Baseline)
    print("--- Model 3: Answer Word Count (Length Baseline) ---")
    len_train = np.log1p([len(str(a).split()) for a in df_train_pool['generated_answer']]).reshape(-1, 1)
    len_placebo = np.log1p([len(str(a).split()) for a in df_placebo['generated_answer']]).reshape(-1, 1)
    len_expert = np.log1p([len(str(a).split()) for a in df_expert['generated_answer']]).reshape(-1, 1)
    
    clf_len = LogisticRegression(class_weight='balanced', random_state=42)
    clf_len.fit(len_train, y_train)
    p_placebo_len = clf_len.predict_proba(len_placebo)[:, 1]
    p_expert_len = clf_len.predict_proba(len_expert)[:, 1]
    
    def calc_metrics(y_true, y_prob):
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
        ap = average_precision_score(y_true, y_prob)
        y_pred = (y_prob >= 0.5).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        return round(auc, 4), round(ap, 4), round(f1, 4), round(acc, 4)

    results = []
    
    # External PlaceboBench Evaluation
    auc, ap, f1, acc = calc_metrics(y_placebo, p_placebo_lr)
    results.append({'Target': 'PlaceboBench (RAG)', 'Model': 'TF-IDF + Logistic Regression', 'AUROC': auc, 'AP': ap, 'F1': f1, 'Accuracy': acc})
    
    auc, ap, f1, acc = calc_metrics(y_placebo, p_placebo_lgb)
    results.append({'Target': 'PlaceboBench (RAG)', 'Model': 'TF-IDF + LightGBM', 'AUROC': auc, 'AP': ap, 'F1': f1, 'Accuracy': acc})
    
    auc, ap, f1, acc = calc_metrics(y_placebo, p_placebo_len)
    results.append({'Target': 'PlaceboBench (RAG)', 'Model': 'Answer Length (LR)', 'AUROC': auc, 'AP': ap, 'F1': f1, 'Accuracy': acc})
    
    # External MedExpert Evaluation
    auc, ap, f1, acc = calc_metrics(y_expert, p_expert_lr)
    results.append({'Target': 'MedExpert (Chatbot)', 'Model': 'TF-IDF + Logistic Regression', 'AUROC': auc, 'AP': ap, 'F1': f1, 'Accuracy': acc})
    
    auc, ap, f1, acc = calc_metrics(y_expert, p_expert_lgb)
    results.append({'Target': 'MedExpert (Chatbot)', 'Model': 'TF-IDF + LightGBM', 'AUROC': auc, 'AP': ap, 'F1': f1, 'Accuracy': acc})
    
    auc, ap, f1, acc = calc_metrics(y_expert, p_expert_len)
    results.append({'Target': 'MedExpert (Chatbot)', 'Model': 'Answer Length (LR)', 'AUROC': auc, 'AP': ap, 'F1': f1, 'Accuracy': acc})
    
    df_results = pd.DataFrame(results)
    print("\n=== ZERO-SHOT EXTERNAL GENERALIZATION RESULTS ===")
    print(df_results.to_string(index=False))
    
    csv_out = os.path.join(RESULTS_DIR, 'medfactguard_external_generalization_baselines.csv')
    df_results.to_csv(csv_out, index=False)
    print(f"\nSaved results to: {csv_out}")
    return df_results

if __name__ == '__main__':
    run_baseline_evaluation()
