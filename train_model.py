import os, joblib, numpy as np, pandas as pd, math
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
import lightgbm as lgb

# Paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'cliniguard_all_datasets.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'final_website', 'cliniguard_model.joblib')
SCALER_PATH = os.path.join(BASE_DIR, 'final_website', 'cliniguard_scaler.joblib')

# Token helpers
DRUG_TERMS = {"mg","dose","dosage","tablet","capsule","injection","oral","iv","intravenous","amoxicillin","ibuprofen","metformin","insulin","aspirin","atorvastatin","omeprazole","paracetamol","acetaminophen","warfarin","morphine","prednisone","antibiotic","medication","drug","prescribe","contraindication","side","effect","adverse"}
CONTEXT_TERMS = {"patient","allergy","allergic","age","weight","pediatric","adult","vital","history","medication","diagnosis","symptom","report","female","male","chronic","acute","clinical","contraindication"}
UNCERTAIN_WORDS = {"maybe","possibly","might","could","uncertain","unclear","unknown","approximately","seems","appears","suggest","perhaps","likely","probably","assume","think","believe","estimate","roughly","sometimes","often"}
FABRICATED_TERMS = {"unicorn","quantum","glitter","wormhole","interdimensional","cosmic","teleportation","alien","fantasy","myth","mythical","magical"}

def tokenize(t):
    return t.lower().split() if isinstance(t, str) else []

def med_isp(text):
    w = tokenize(text)
    if not w: return 1.0
    hits = sum(1 for x in w if any(t in x for t in DRUG_TERMS))
    return round(1.0 - min(hits / max(len(w)*0.05, 1), 1.0), 4)

def c_aas(text):
    w = tokenize(text)
    if not w: return 1.0
    hits = sum(1 for x in w if any(t in x for t in CONTEXT_TERMS))
    return round(1.0 - min(hits / max(len(w)*0.04, 1), 1.0), 4)

def med_eem(text):
    w = tokenize(text); n = len(w)
    if n == 0: return 0.0
    p = sum(1 for x in w if any(t in x for t in UNCERTAIN_WORDS)) / n
    eps = 1e-9
    H = -(p*math.log2(p+eps) + (1-p)*math.log2(1-p+eps))
    return round(min(H*(1+p), 1.0), 4)

def cdt(answer, question):
    def wvec(t):
        f={}
        for x in tokenize(t): f[x]=f.get(x,0)+1
        return f
    v1,v2 = wvec(question), wvec(answer)
    vocab = set(v1)|set(v2)
    if not vocab: return 0.5
    dot = sum(v1.get(x,0)*v2.get(x,0) for x in vocab)
    m1 = math.sqrt(sum(x**2 for x in v1.values()))
    m2 = math.sqrt(sum(x**2 for x in v2.values()))
    if m1==0 or m2==0: return 0.5
    return round(1.0 - dot/(m1*m2), 4)

def contains_fabricated(text):
    w = set(tokenize(text))
    return any(term in w for term in FABRICATED_TERMS)

# Load data
print('Loading dataset...')
df = pd.read_csv(DATA_PATH)
# keep necessary columns
df = df[['question','answer','label']]
# compute signals
print('Computing signal features...')
signals = []
for idx, row in df.iterrows():
    ans = row['answer']
    q = row['question']
    sig = [
        med_isp(ans),
        c_aas(ans),
        med_eem(ans),
        cdt(ans, q),
    ]
    signals.append(sig)
X = np.array(signals)
# target label (0: GREEN, 1: AMBER, 2: RED) – already numeric in dataset
y = df['label'].astype(int).values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# LightGBM with hyperparameter grid search (small for speed)
param_grid = {
    'num_leaves': [31, 61],
    'learning_rate': [0.1, 0.05],
    'n_estimators': [200, 400],
    'objective': ['multiclass'],
    'num_class': [3]
}
print('Running GridSearchCV...')
lgb_est = lgb.LGBMClassifier()
grid = GridSearchCV(lgb_est, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)
print('Best params:', grid.best_params_)

# Retrain on full training set with best params
best_model = grid.best_estimator_
print('Training final model on full data...')
best_model.fit(np.vstack([X_train, X_val]), np.hstack([y_train, y_val]))

# Save model and scaler
print('Saving model and scaler...')
joblib.dump(best_model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
print('Done! Model saved to', MODEL_PATH)
