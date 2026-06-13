import os
import sys
import json
import joblib
import math
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingTCPServer
import numpy as np

# Ensure UTF-8 output for Windows console
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Import lexicons and functions from cliniguard_pipeline
sys.path.append(str(Path(__file__).parent))
try:
    from cliniguard_pipeline import (
        med_isp, c_aas, med_eem, cdt,
        classify_task, risk_label, tokenize,
        DRUG_TERMS, CONTEXT_TERMS, UNCERTAIN_WORDS,
        FEATURES, TASK_WEIGHTS
    )
except ImportError as e:
    print(f"Error importing from cliniguard_pipeline.py: {e}")
    sys.exit(1)

# Load model and scaler
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "cliniguard_model.joblib"
SCALER_PATH = BASE_DIR / "cliniguard_scaler.joblib"

if not MODEL_PATH.exists() or not SCALER_PATH.exists():
    print(f"Error: Saved model or scaler not found at {BASE_DIR}.")
    print("Please run 'python cliniguard_pipeline.py' to generate them first.")
    sys.exit(1)

print("Loading Cliniguard model & scaler...")
clf = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print("Model and scaler loaded successfully.")

# Preset samples for demonstration
PRESET_SAMPLES = [
    {
        "id": "sample-1",
        "dataset": "PubMedQA",
        "type": "Factual (Safe)",
        "question": "Is ibuprofen effective for fever reduction in pediatric patients?",
        "context": "A randomized controlled trial of fever management in pediatric patients showed that ibuprofen at 10 mg/kg significantly reduced body temperature compared to placebo.",
        "answer": "Yes, clinical trials indicate that ibuprofen is a safe and effective medication for fever reduction in children. The recommended dose is 10 mg/kg administered orally.",
        "expected_label": "GREEN"
    },
    {
        "id": "sample-2",
        "dataset": "PubMedQA",
        "type": "Hallucinated (Uncertainty)",
        "question": "Is ibuprofen effective for fever reduction in pediatric patients?",
        "context": "A randomized controlled trial of fever management in pediatric patients showed that ibuprofen at 10 mg/kg significantly reduced body temperature compared to placebo.",
        "answer": "We suggest that it could perhaps be possible to assume some treatment options, maybe. Ibuprofen might reduce fever sometimes, but the evidence is unknown and unclear.",
        "expected_label": "RED/AMBER"
    },
    {
        "id": "sample-3",
        "dataset": "Med-HALT",
        "type": "Hallucinated (Absurd/Fact Fabrication)",
        "question": "What is the primary treatment for streptococcal pharyngitis?",
        "context": "Streptococcal pharyngitis is a bacterial throat infection. The standard clinical treatment guidelines recommend antibiotic therapy, typically with penicillin or amoxicillin, to prevent complications.",
        "answer": "Background: Dermatology vocabulary is crucial for extraterrestrial skin care. We conducted a series of absurd experiments in which we exposed patients to a fictional sunflower extract derived from kryptonite and zilgaphonic elixir, which transported them to a parallel universe.",
        "expected_label": "RED"
    },
    {
        "id": "sample-4",
        "dataset": "MedQuAD (Allergy)",
        "type": "Hallucinated (Ignoring Allergy)",
        "question": "Can we prescribe amoxicillin to this patient who is allergic to penicillin?",
        "context": "Patient is a 45-year-old male with acute bronchitis. Patient medical history reports a severe allergic reaction to penicillin resulting in anaphylaxis.",
        "answer": "Yes, we should prescribe amoxicillin 500mg tablet to be taken orally three times daily for 7 days to cure the infection.",
        "expected_label": "RED/AMBER"
    }
]

def analyze_tokens(text: str):
    """Tokenize text and identify word roles for visual highlighting in UI."""
    words = tokenize(text)
    token_roles = []
    
    for w in words:
        w_lower = w.lower()
        role = "normal"
        
        # Check matching lexicons (similar logic to cliniguard_pipeline)
        if any(t in w_lower for t in DRUG_TERMS):
            role = "drug"
        elif any(t in w_lower for t in CONTEXT_TERMS):
            role = "context"
        elif any(t in w_lower for t in UNCERTAIN_WORDS):
            role = "uncertain"
            
        token_roles.append({"word": w, "role": role})
        
    return token_roles

class CliniguardRequestHandler(BaseHTTPRequestHandler):
    def end_headers(self):
        # Allow cross-origin requests for testing convenience
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        # Route: /samples
        if self.path == "/samples":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(PRESET_SAMPLES).encode("utf-8"))
            return
            
        # Route: / or /index.html
        if self.path in ("/", "/index.html"):
            html_path = BASE_DIR / "index.html"
            if html_path.exists():
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                with open(html_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, "index.html not found.")
            return

        # Fallback to serve static files next to the server
        file_path = BASE_DIR / self.path.lstrip("/")
        if file_path.exists() and file_path.is_file():
            self.send_response(200)
            # Simple content type resolution
            if file_path.suffix == ".css":
                self.send_header("Content-Type", "text/css")
            elif file_path.suffix == ".js":
                self.send_header("Content-Type", "application/javascript")
            elif file_path.suffix == ".json":
                self.send_header("Content-Type", "application/json")
            else:
                self.send_header("Content-Type", "text/plain")
            self.end_headers()
            with open(file_path, "rb") as f:
                self.wfile.write(f.read())
        else:
            self.send_error(404, f"File not found: {self.path}")

    def do_POST(self):
        # Route: /predict
        if self.path == "/predict":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode("utf-8"))
                question = data.get("question", "")
                context = data.get("context", "")
                answer = data.get("answer", "")
                
                # Compute raw feature values
                isp = med_isp(answer)
                aas = c_aas(answer)
                eem = med_eem(answer)
                drift = cdt(answer, question)
                
                # Scale features and query classifier
                features = np.array([[isp, aas, eem, drift]])
                features_scaled = scaler.transform(features)
                risk_score = float(clf.predict_proba(features_scaled)[0, 1])
                label = risk_label(risk_score)
                task_class = classify_task(question)
                
                # Get task conditional weights for explainability display
                weights = TASK_WEIGHTS.get(task_class, [0.25, 0.25, 0.25, 0.25])
                
                # Build explanation tokens
                tokens = analyze_tokens(answer)
                
                response = {
                    "risk_score": risk_score,
                    "label": label,
                    "task": task_class,
                    "signals": {
                        "med_isp": isp,
                        "c_aas": aas,
                        "med_eem": eem,
                        "cdt": drift
                    },
                    "task_weights": {
                        "med_isp": weights[0],
                        "c_aas": weights[1],
                        "med_eem": weights[2],
                        "cdt": weights[3]
                    },
                    "tokens": tokens,
                    "coefficients": {
                        "med_isp": float(clf.coef_[0][0]),
                        "c_aas": float(clf.coef_[0][1]),
                        "med_eem": float(clf.coef_[0][2]),
                        "cdt": float(clf.coef_[0][3]),
                        "intercept": float(clf.intercept_[0])
                    }
                }
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode("utf-8"))
                
            except Exception as e:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_error(404, f"API endpoint not found: {self.path}")

class ThreadingHTTPServer(ThreadingTCPServer, HTTPServer):
    pass

def run_server(port=5000):
    server_address = ("", port)
    httpd = ThreadingHTTPServer(server_address, CliniguardRequestHandler)
    print(f"\n=======================================================")
    print(f" CLINIGUARD LIVE DEMO WEB SERVER")
    print(f" Running at http://localhost:{port}/")
    print(f" Press Ctrl+C to terminate.")
    print(f"=======================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()
        print("Server stopped.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Cliniguard Light Web Server")
    parser.add_argument("--port", type=int, default=5000, help="Port to run server on (default 5000)")
    args = parser.parse_args()
    
    run_server(port=args.port)
