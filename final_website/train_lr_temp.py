import os, joblib, pandas as pd, numpy as np, math
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'cliniguard_all_datasets.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'cliniguard_lr_model.joblib')
SCALER_PATH = os.path.join(BASE_DIR, 'cliniguard_scaler.joblib')

DRUG_TERMS = {"mg","dose","dosage","tablet","capsule","injection","oral","iv","intravenous","amoxicillin","ibuprofen","metformin","insulin","aspirin","atorvastatin","omeprazole","paracetamol","acetaminophen","warfarin","morphine","prednisone","antibiotic","medication","drug","prescribe","contraindication","side","effect","adverse"}
CONTEXT_TERMS = {"patient","allergy","allergic","age","weight","pediatric","adult","vital","history","medication","diagnosis","symptom","report","female","male","chronic","acute","clinical","contraindication"}
UNCERTAIN_WORDS = {"maybe","possibly","might","could","uncertain","unclear","unknown","approximately","seems","appears","suggest","perhaps","likely","probably","assume","think","believe","estimate","roughly","sometimes","often"}

def tokenize(t):
    return t.lower().split() if isinstance(t, str) else []

def med_isp(text):
    w = tokenize(text)
    if not w:
        return 1.0
    hits = sum(1 for x in w if any(t in x for t in DRUG_TERMS))
    return round(1.0 - min(hits / max(len(w) * 0.05, 1), 1.0), 4)

def c_aas(text):
    w = tokenize(text)
    if not w:
        return 1.0
    hits = sum(1 for x in w if any(t in x for t in CONTEXT_TERMS))
    return round(1.0 - min(hits / max(len(w) * 0.04, 1), 1.0), 4)

def med_eem(text):
    w = tokenize(text)
    n = len(w)
    if n == 0:
        return 0.0
    p = sum(1 for x w if any(t in x for t in UNCERTAIN_WORDS)) / n
    eps = 1e-9
    H = -(p * math.log2(p + eps) + (1 - p) * math.log2(1 - p + eps))
    return round(min(H * (1 + p), 1.0), 4)

def cdt(answer, question):
    def wvec(t):
        f = {}
        for x in tokenize(t):
            f[x] = f.get(x, 0) + 1
        return f
    v1, v2 = wvec(question), wvec(answer)
    vocab = set(v1) | set(v2)
    if not vocab:
        return 0.5
    dot = sum(v1.get(x, 0) * v2.get(x, 0) for x in vocab)
    m1 = math.sqrt(sum(x ** 2 for x in v1.values()))
    m2 = math.sqrt(sum(x ** 2 for x in v2.values()))
    if m1 == 0 or m2 == 0:
        return 0.5
    return round(1.0 - dot / (m1 * m2), 4)

# Load dataset
df = pd.read_csv(DATA_PATH)[['question','answer','label']]
signals = []
for _, row in df.iterrows():
    ans = row['answer']
    q = row['question']
    signals.append([med_isp(ans), c_aas(ans), med_eem(ans), cdt(ans, q)])
X = np.array(signals)
y = df['label'].astype(int).values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, class_weight='balanced')
logreg.fit(X_scaled, y)

joblib.dump(logreg, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
print('Logistic Regression model saved to', MODEL_PATH)
