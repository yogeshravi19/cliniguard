import os, joblib, numpy as np, pandas as pd, math
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression

# Paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'cliniguard_all_datasets.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'cliniguard_lr_model.joblib')
SCALER_PATH = os.path.join(BASE_DIR, 'cliniguard_scaler.joblib')

# Token helpers (same as in training pipeline)
DRUG_TERMS = {"mg","dose","dosage","tablet","capsule","injection","oral","iv","intravenous","amoxicillin","ibuprofen","metformin","insulin","aspirin","atorvastatin","omeprazole","paracetamol","acetaminophen","warfarin","morphine","prednisone","antibiotic","medication","drug","prescribe","contraindication","side","effect","adverse"}
CONTEXT_TERMS = {"patient","allergy","allergic","age","weight","pediatric","adult","vital","history","medication","diagnosis","symptom","report","female","male","chronic","acute","clinical","contraindication"}
UNCERTAIN_WORDS = {"maybe","possibly","might","could","uncertain","unclear","unknown","approximately","seems","appears","suggest","perhaps","likely","probably","assume","think","believe","estimate","roughly","sometimes","often"}

def tokenize(t):
    return t.lower().split() if isinstance(t, str) else []

def med_isp(text):
    w = tokenize(text)
    if not w:
        return 1.0
    hits = sum(1 for x in w if any(term in x for term in DRUG_TERMS))
    return 1.0 - min(hits / max(len(w) * 0.05, 1), 1.0)

def c_aas(text):
    w = tokenize(text)
    if not w:
        return 1.0
    hits = sum(1 for x in w if any(term in x for term in CONTEXT_TERMS))
    return 1.0 - min(hits / max(len(w) * 0.04, 1), 1.0)

def med_eem(text):
    w = tokenize(text)
    n = len(w)
    if n == 0:
        return 0.0
    p = sum(1 for x in w if any(term in x for term in UNCERTAIN_WORDS)) / n
    eps = 1e-9
    H = -(p * math.log2(p + eps) + (1 - p) * math.log2(1 - p + eps))
    return min(H * (1 + p), 1.0)

def cdt(answer, question):
    def wvec(t):
        freq = {}
        for token in tokenize(t):
            freq[token] = freq.get(token, 0) + 1
        return freq
    v1, v2 = wvec(question), wvec(answer)
    vocab = set(v1) | set(v2)
    if not vocab:
        return 0.5
    dot = sum(v1.get(tok, 0) * v2.get(tok, 0) for tok in vocab)
    m1 = math.sqrt(sum(val ** 2 for val in v1.values()))
    m2 = math.sqrt(sum(val ** 2 for val in v2.values()))
    if m1 == 0 or m2 == 0:
        return 0.5
    return 1.0 - dot / (m1 * m2)

print('Loading dataset...')
df = pd.read_csv(DATA_PATH)[['question', 'answer', 'label']]
print('Computing feature matrix...')
signals = []
for _, row in df.iterrows():
    ans = row['answer']
    q = row['question']
    signals.append([med_isp(ans), c_aas(ans), med_eem(ans), cdt(ans, q)])
X = np.array(signals)
y = df['label'].astype(int).values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Hyper‑parameter search for C
param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
logreg = LogisticRegression(
    multi_class='multinomial', solver='lbfgs', max_iter=1000,
    class_weight='balanced')
grid = GridSearchCV(logreg, param_grid, cv=3, scoring='accuracy')
grid.fit(X_train, y_train)
print('Best C:', grid.best_params_)

# Retrain on full data (train + validation)
best_model = grid.best_estimator_
best_model.fit(np.vstack([X_train, X_val]), np.hstack([y_train, y_val]))

# Save model and scaler
joblib.dump(best_model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
print('Model and scaler saved to', MODEL_PATH, SCALER_PATH)
