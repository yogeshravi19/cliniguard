# 🏥 CLINIGUARD – Medical AI Hallucination Detector

> A safety guardrail for LLM‑generated medical answers that detects hallucinations and unsafe content using four hand‑crafted linguistic signals fused by a LightGBM model.

---

## 📋 Requirements

Before running anything, make sure the following are installed on your machine:

| Requirement | Version | Notes |
|-------------|---------|-------|
| **Python** | 3.9 or higher | [Download here](https://www.python.org/downloads/) |
| **pip** | latest | Comes with Python |
| **Git** | any | Only needed to clone the repo |

### Python Packages

Install all required packages with a single command:

```bash
pip install fastapi uvicorn joblib scikit-learn lightgbm numpy pandas
```

Or, if a `requirements.txt` is present:

```bash
pip install -r requirements.txt
```

---

## 🖥️ How to Run the Website on Any Laptop

> **The repository is already cloned on the laptop? Follow these steps.**

### Step 1 – Pull the latest code

Open a terminal, navigate to the cloned repo folder, and pull any updates:

```bash
cd path/to/cliniguard
git pull
```

### Step 2 – Make sure the model files are present

Check that these three files exist in the `final_website` folder:

```
final_website/cliniguard_model.joblib       ✅ must be present
final_website/cliniguard_scaler.joblib      ✅ must be present
final_website/cliniguard_lr_model.joblib    ✅ must be present
```

If any are missing, copy them from the original machine into the `final_website` folder.

### Step 3 – Install dependencies

```bash
pip install fastapi uvicorn joblib scikit-learn lightgbm numpy pandas
```

### Step 4 – Start the API server

```bash
python app_demo.py
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### Step 5 – Open the website

Open this file in your browser:

```
final_website/index.html
```

The web UI will automatically connect to `http://127.0.0.1:8000/predict`.  
Type a medical question and answer — it will return 🟢 SAFE / 🟡 AMBIGUOUS / 🔴 RED.

---

## 🌐 Access from Other Devices on the Same Wi‑Fi

1. Find the laptop's local IP address:
   - **Windows**: open CMD → `ipconfig` → look for **IPv4 Address** (e.g., `192.168.1.5`)
   - **Mac/Linux**: `ifconfig` or `ip a`

2. The server listens on `0.0.0.0:8000` so it's already accessible on your network.

3. From any phone, tablet, or PC on the same Wi‑Fi, open:
   ```
   http://192.168.1.5:8000
   ```
   *(replace with your actual IP)*

---

## ⚠️ Common Issues & Fixes

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: lightgbm` | Run `pip install lightgbm` |
| `ModuleNotFoundError: fastapi` | Run `pip install fastapi uvicorn` |
| `FileNotFoundError: cliniguard_model.joblib` | Copy the `.joblib` files into the `final_website` folder |
| UI shows "Connection Refused" | The server isn't running — go back to Step 4 |
| Port 8000 already in use | Run `python -m uvicorn server:app --port 8001` |
| `python` not recognised | Make sure Python is added to PATH during installation |

---

## 📁 Repository Structure

```
cliniguard/
├── 📁 data_extraction/          ← Raw benchmark datasets
├── 📁 final_website/            ← Web UI & notebooks
│   └── index.html               ← Open this in your browser
├── 📁 scripts/                  ← Utility scripts
├── 📁 web_portal/               ← FastAPI server files
├── cliniguard_pipeline.py       ← Core: 4 signal functions
├── server.py                    ← FastAPI /predict endpoint
├── app_demo.py                  ← ▶️  Run this to start the server
└── PROJECT_OVERVIEW.md          ← Full project documentation
```

---

## 📖 Full Documentation

See **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)** for the complete:
- End‑to‑end pipeline explanation
- Four signal descriptions
- Model performance metrics
- Step‑by‑step workflow to reproduce from scratch
- Contributor checklist
- Publication guide

---

*Last updated: 2026‑06‑14*
