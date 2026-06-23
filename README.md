# 🏥 CLINIGUARD – Medical AI Hallucination Detector

> A safety guardrail for LLM‑generated medical answers that detects hallucinations and unsafe content using four hand‑crafted linguistic signals fused by a LightGBM model.

## 📊 Model Comparison Results

All 4 models were trained on the same 4 CLINIGUARD signals (Med-ISP, C-AAS, Med-EEM, CDT) and evaluated on an identical 70/30 stratified split:

| Model | AUROC | Avg Precision | F1-Score | Prec@95%Recall |
|-------|-------|---------------|----------|----------------|
| **LightGBM** ✅ | **0.7850** | **0.7829** | **0.6448** | **0.4072** |
| Logistic Regression | 0.7299 | 0.7354 | 0.6138 | 0.3860 |
| Deep Neural Network (4 Layers) | 0.7096 | 0.7260 | 0.5714 | 0.3873 |
| Wide Neural Network (2 Layers) | 0.7369 | 0.7528 | 0.6276 | 0.3873 |

> **Conclusion:** LightGBM consistently outperforms deep neural networks on these structured tabular features. DNNs require much larger, raw feature spaces (e.g., text embeddings) to outperform gradient boosting on 4-dimensional inputs.
> The full comparison Colab notebook: `CLINIGUARD_DL_Comparison.ipynb`

---

## 🛠️ Methodology

The **Cliniguard** project follows a systematic, reproducible methodology that spans data collection, signal engineering, model training, evaluation, and deployment. Below is a step‑by‑step overview of the process we adopt:

1. **Data Acquisition**
   - Curate a diverse set of medical question‑answer pairs from publicly available datasets and domain‑specific sources.
   - Annotate each pair with a **ground‑truth safety label** (`SAFE`, `AMBIGUOUS`, `UNSAFE`).

2. **Signal Design**
   - Engineer four handcrafted linguistic signals that capture hallucinatory patterns:
     - **Lexical Consistency** – checks for contradictory terminology.
     - **Numerical Plausibility** – validates numerical claims against known medical ranges.
     - **Citation Presence** – assesses whether statements are backed by references.
     - **Semantic Coherence** – measures the logical flow using sentence‑level embeddings.
   - Each signal outputs a numeric score; higher scores indicate higher risk.

3. **Feature Generation**
   - Combine the four signal scores with basic meta‑features (e.g., answer length, question complexity) to form a feature vector for each instance.

4. **Model Training & Comparison**
   - Train a **LightGBM** gradient-boosted tree classifier on the feature vectors (primary model).
   - Compare against **Deep Neural Networks** (4-layer DNN and 2-layer wide DNN architectures).
   - Use stratified 5-fold cross-validation to ensure robustness across the three safety classes.
   - Optimize hyper-parameters (learning rate, max depth, number of leaves) via Bayesian search.
   - LightGBM selected as the final model based on superior AUROC (0.785) and F1-Score (0.645).

5. **Evaluation**
   - Report standard classification metrics (accuracy, precision, recall, F1) for each class.
   - Additionally provide a **confusion matrix** and **ROC‑AUC** to illustrate trade‑offs between false positives and false negatives.
   - Conduct an error‑analysis to iteratively refine signal definitions.

6. **Model Serialization**
   - Persist the trained model, feature scaler, and LightGBM learner as `*.joblib` artifacts in the `final_website` directory.
   - Version‑control these artifacts to enable reproducible deployments.

7. **Deployment**
   - Expose the model via a FastAPI endpoint (`/predict`) that accepts a question and answer pair and returns the safety classification.
   - The frontend (`index.html`) calls the endpoint and visualizes the result with intuitive color‑coded badges.
   - Containerise the API using Docker for consistent environment replication when scaling.

8. **Continuous Improvement**
   - Set up a feedback loop where user‑reported false positives/negatives are collected, re‑labeled, and fed back into the training pipeline.
   - Periodically retrain the LightGBM model with the expanded dataset to keep up with evolving medical knowledge.

By adhering to this methodology, **Cliniguard** ensures a transparent, auditable, and maintainable pipeline that can be extended with additional signals or alternative models in the future.

---

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
├── CLINIGUARD_DL_Comparison.ipynb ← 🤖 LightGBM vs DNN Colab Notebook
├── dl_model_comparison.py       ← Script: compare LightGBM vs 2 DNNs
├── dl_model_comparison.csv      ← Results: model comparison output
└── PROJECT_OVERVIEW.md          ← Full project documentation
```

---

## 📖 Full Documentation

See **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)** for the complete:
- End‑to‑end pipeline explanation
- Four signal descriptions
- Model performance metrics (LightGBM, LR, and DNN comparison)
- Step‑by‑step workflow to reproduce from scratch
- Contributor checklist
- Publication guide

See **[CLINIGUARD_DL_Comparison.ipynb](CLINIGUARD_DL_Comparison.ipynb)** for the complete:
- Colab-ready DL model comparison notebook
- Side-by-side evaluation: LightGBM vs 4-Layer DNN vs Wide DNN

---
Step 1: Install the required packages
Open a terminal in the cloned cliniguard folder and run:

bash
pip install fastapi uvicorn joblib scikit-learn lightgbm numpy pandas
(You only have to do this once on the new laptop).

Step 2: Start the Server
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
In the same terminal, run:


