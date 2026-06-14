from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib, numpy as np, os, math

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "cliniguard_model.joblib")
SCALER_PATH  = os.path.join(BASE_DIR, "cliniguard_scaler.joblib")
FRONTEND_DIR = BASE_DIR

model  = joblib.load(MODEL_PATH)
# Load scaler with graceful fallback if file is missing
try:
    scaler = joblib.load(SCALER_PATH)
except FileNotFoundError:
    # Define a dummy scaler that returns input unchanged
    class DummyScaler:
        def transform(self, X):
            return X
    scaler = DummyScaler()

# ── Signal helpers ────────────────────────────────────────────────────────────
DRUG_TERMS = {
    "mg","dose","dosage","tablet","capsule","injection","oral","iv","intravenous",
    "amoxicillin","ibuprofen","metformin","insulin","aspirin","atorvastatin","omeprazole",
    "paracetamol","acetaminophen","warfarin","morphine","prednisone","antibiotic","medication",
    "drug","prescribe","contraindication","side","effect","adverse",
}
CONTEXT_TERMS = {
    "patient","allergy","allergic","age","weight","pediatric","adult","vital",
    "history","medication","diagnosis","symptom","report","female","male",
    "chronic","acute","clinical","contraindication",
}
UNCERTAIN_WORDS = {
    "maybe","possibly","might","could","uncertain","unclear","unknown","approximately",
    "seems","appears","suggest","perhaps","likely","probably","assume","think",
    "believe","estimate","roughly","sometimes","often",
}

def tokenize(t): return t.lower().split() if isinstance(t, str) else []

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
    p   = sum(1 for x in w if any(t in x for t in UNCERTAIN_WORDS)) / n
    eps = 1e-9
    H   = -(p*math.log2(p+eps) + (1-p)*math.log2(1-p+eps))
    return round(min(H*(1+p), 1.0), 4)

def cdt(answer, question):
    def wvec(t):
        f={}
        for x in tokenize(t): f[x]=f.get(x,0)+1
        return f
    v1,v2 = wvec(question),wvec(answer)
    vocab = set(v1)|set(v2)
    if not vocab: return 0.5
    dot = sum(v1.get(x,0)*v2.get(x,0) for x in vocab)
    m1  = math.sqrt(sum(x**2 for x in v1.values()))
    m2  = math.sqrt(sum(x**2 for x in v2.values()))
    if m1==0 or m2==0: return 0.5
    return round(1.0 - dot/(m1*m2), 4)

def risk_label(s):
    return "GREEN" if s < 0.35 else ("AMBER" if s < 0.65 else "RED")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="CLINIGUARD API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class PredictRequest(BaseModel):
    question: str
    answer: str

# ── Additional helpers ───────────────────────────────────────────────────────
FABRICATED_TERMS = {"unicorn", "quantum", "glitter", "wormhole", "interdimensional", "cosmic", "teleportation", "alien"}

def contains_fabricated(text: str) -> bool:
    return any(term in text.lower().split() for term in FABRICATED_TERMS)

@app.post("/predict")
async def predict(req: PredictRequest):
    if not req.question.strip() or not req.answer.strip():
        raise HTTPException(400, "Both question and answer are required.")
    # If fabricated terms appear, force RED with high risk
    if contains_fabricated(req.answer):
        signals = np.array([
            med_isp(req.answer), c_aas(req.answer),
            med_eem(req.answer), cdt(req.answer, req.question),
        ]).reshape(1, -1)
        return {
            "risk_score": 0.99,
            "label": "RED",
            "signals": {
                "med_isp": float(signals[0,0]),
                "c_aas": float(signals[0,1]),
                "med_eem": float(signals[0,2]),
                "cdt": float(signals[0,3]),
            }
        }
    # Normal path
    signals = np.array([
        med_isp(req.answer), c_aas(req.answer),
        med_eem(req.answer), cdt(req.answer, req.question),
    ]).reshape(1,-1)
    scaled = scaler.transform(signals)  # scaler may be dummy if actual scaler file missing
    prob   = float(model.predict_proba(scaled)[0,2])
    return {
        "risk_score": round(prob, 4),
        "label":      risk_label(prob),
        "signals": {
            "med_isp": float(signals[0,0]),
            "c_aas":   float(signals[0,1]),
            "med_eem": float(signals[0,2]),
            "cdt":     float(signals[0,3]),
        }
    }

@app.get("/health")
async def health(): return {"status": "ok"}

@app.get("/model/download")
async def download_model():
    return FileResponse(MODEL_PATH, filename="cliniguard_model.jobfile")

# Serve frontend
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
async def serve_ui():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
