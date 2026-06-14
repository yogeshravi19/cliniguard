import os, joblib, numpy as np, pandas as pd, math
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'cliniguard_all_datasets.csv')
MODEL_LGB_PATH = os.path.join(BASE_DIR, 'cliniguard_model.joblib')
MODEL_LR_PATH = os.path.join(BASE_DIR, 'cliniguard_lr_model.joblib')
SCALER_PATH = os.path.join(BASE_DIR, 'cliniguard_scaler.joblib')

# Token helpers (same as training scripts)
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
        f = {}
        for x in tokenize(t):
            f[x] = f.get(x, 0) + 1
        return f
    v1, v2 = wvec(question), wvec(answer)
    vocab = set(v1) | set(v2)
    if not vocab: return 0.5
    dot = sum(v1.get(x,0)*v2.get(x,0) for x in vocab)
    m1 = math.sqrt(sum(x**2 for x in v1.values()))
    m2 = math.sqrt(sum(x**2 for x in v2.values()))
    if m1 == 0 or m2 == 0: return 0.5
    return round(1.0 - dot/(m1*m2), 4)

print('Loading data...')
df = pd.read_csv(DATA_PATH)
df = df[['question','answer','label']]
print('Computing features...')
signals = []
for _, row in df.iterrows():
    ans = row['answer']
    q = row['question']
    sig = [med_isp(ans), c_aas(ans), med_eem(ans), cdt(ans, q)]
    signals.append(sig)
X = np.array(signals)
y = df['label'].astype(int).values

# Load scaler (already fitted on full data during training)
scaler = joblib.load(SCALER_PATH)
X_scaled = scaler.transform(X)

# Split for evaluation (same seed as training)
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Load models
lgb_model = joblib.load(MODEL_LGB_PATH)
lr_model = joblib.load(MODEL_LR_PATH)

# Predict and evaluate
for name, model in [('LightGBM', lgb_model), ('LogisticRegression', lr_model)]:
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    # RED class is label 2
    red_recall = recall_score(y_val, preds, labels=[2], average='macro')
    print(f"{name} – Accuracy: {acc:.4f}, RED recall: {red_recall:.4f}")
